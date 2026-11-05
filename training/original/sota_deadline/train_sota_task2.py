#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset, DataLoader

import model_code_from_original as mc
from sota_models import CTCausalReportGenerator
from training_utils import (
    CheckpointManager,
    get_cosine_with_warmup,
    patient_group_from_volume_name,
)

LABELS = mc.ABNORMALITY_LABELS


class CTFullReportWithLabels(Dataset):
    def __init__(self, df, target_shape, cache_dir):
        self.df = df.reset_index(drop=True)
        self.target_shape = tuple(target_shape)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        volume_name = str(row["VolumeName"])
        subfolder, filename = mc.volume_name_to_hf_parts(volume_name)
        path = hf_hub_download(
            repo_id=mc.REPO_ID,
            repo_type="dataset",
            token=mc.get_hf_token(None) if hasattr(mc, "get_hf_token") else mc.TOKEN,
            filename=filename,
            subfolder=subfolder,
            local_dir=str(self.cache_dir),
            force_download=False,
        )
        x = mc.preprocess_ct_for_torch(path, target_shape=self.target_shape)

        findings = str(row.get("Findings_EN", "") or "")
        impression = str(row.get("Impressions_EN", "") or "")
        report = f"FINDINGS: {findings.strip()}\n\nIMPRESSION: {impression.strip()}"
        y = torch.tensor(row[LABELS].fillna(0).astype(float).to_numpy(), dtype=torch.float32)
        return x, report, y, volume_name


def collate(batch):
    xs, reports, ys, names = zip(*batch)
    return torch.stack(xs), list(reports), torch.stack(ys), list(names)


@torch.no_grad()
def validate(model, dl, device, max_target_length):
    model.eval()
    losses = []
    text_losses = []
    path_losses = []
    for x, reports, y, _ in dl:
        x, y = x.to(device), y.to(device)
        out = model(
            x,
            report_texts=reports,
            pathology_targets=y,
            max_target_length=max_target_length,
        )
        losses.append(float(out["loss"].detach().cpu()))
        text_losses.append(float(out["text_loss"].detach().cpu()))
        path_losses.append(float(out["pathology_loss"].detach().cpu()))
    return float(np.mean(losses)), float(np.mean(text_losses)), float(np.mean(path_losses))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports-csv", required=True)
    ap.add_argument("--labels-csv", required=True)
    ap.add_argument("--out", default="./checkpoints/sota_task2")
    ap.add_argument("--max-rows", type=int, default=10000)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--val-cases", type=int, default=192)
    ap.add_argument("--target-shape", type=int, nargs=3, default=[64,96,96])
    ap.add_argument("--token-grid", type=int, nargs=3, default=[4,6,6])
    ap.add_argument("--visual-tokens", type=int, default=32)
    ap.add_argument("--feature-size", type=int, default=24)
    ap.add_argument("--depths", type=int, nargs=4, default=[1,1,1,1])
    ap.add_argument("--num-heads", type=int, nargs=4, default=[3,6,12,24])
    ap.add_argument("--lm-name", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--max-target-length", type=int, default=256)
    ap.add_argument("--aux-pathology-weight", type=float, default=0.25)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--vision-lr", type=float, default=1e-5)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--cache-dir", default="../vlm3d_original_task_split/ct_rate_volume_cache")
    ap.add_argument("--task1-checkpoint", default=None)
    ap.add_argument("--unfreeze-vision", action="store_true")
    ap.add_argument("--checkpoint-every", type=int, default=2000)
    args = ap.parse_args()

    reports = pd.read_csv(args.reports_csv).copy()
    labels = pd.read_csv(args.labels_csv).copy()
    df = reports.merge(labels[["VolumeName"] + LABELS], on="VolumeName", how="inner")
    df["Findings_EN"] = df["Findings_EN"].fillna("").astype(str)
    df["Impressions_EN"] = df["Impressions_EN"].fillna("").astype(str)
    keep = (
        df["Findings_EN"].str.strip().ne("")
        & df["Impressions_EN"].str.strip().ne("")
    )
    df = df[keep].reset_index(drop=True)

    groups = df["VolumeName"].map(patient_group_from_volume_name)
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=min(args.val_cases / max(len(df), 1), 0.2),
        random_state=17,
    )
    tr_idx, va_idx = next(splitter.split(df, groups=groups))
    train_df = df.iloc[tr_idx].reset_index(drop=True)
    val_df = df.iloc[va_idx].reset_index(drop=True)
    if args.max_rows is not None:
        train_df = train_df.head(args.max_rows)
    val_df = val_df.head(args.val_cases)

    train_ds = CTFullReportWithLabels(train_df, args.target_shape, args.cache_dir)
    val_ds = CTFullReportWithLabels(val_df, args.target_shape, args.cache_dir)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate)

    vision_params = dict(
        feature_size=args.feature_size,
        in_channels=1,
        out_channels=1,
        spatial_dims=3,
        depths=tuple(args.depths),
        num_heads=tuple(args.num_heads),
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.05,
        use_checkpoint=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CTCausalReportGenerator(
        vision_params=vision_params,
        lm_name=args.lm_name,
        token_grid=tuple(args.token_grid),
        num_visual_tokens=args.visual_tokens,
        aux_pathology_weight=args.aux_pathology_weight,
        freeze_vision=not args.unfreeze_vision,
        use_lora=True,
        gradient_checkpointing=True,
    ).to(device)

    # Initialize LazyLinear before optimizer setup.
    with torch.no_grad():
        dummy = torch.zeros(1,1,*args.target_shape, device=device)
        _ = model.encode_visual(dummy)

    if args.task1_checkpoint:
        model.load_task1_encoder(args.task1_checkpoint)

    vision_ids = {id(p) for p in model.vision.parameters()}
    vision_trainable = [p for p in model.vision.parameters() if p.requires_grad]
    other = [p for p in model.parameters() if p.requires_grad and id(p) not in vision_ids]

    groups_opt = [{"params": other, "lr": args.lr}]
    if vision_trainable:
        groups_opt.append({"params": vision_trainable, "lr": args.vision_lr})

    opt = torch.optim.AdamW(groups_opt, weight_decay=args.weight_decay)
    total_steps = max(1, len(train_dl) * args.epochs)
    scheduler = get_cosine_with_warmup(opt, total_steps, args.warmup_ratio)

    ckpt = CheckpointManager(args.out, every_steps=args.checkpoint_every)
    best = float("inf")
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        losses = []
        for x, reports_batch, y, _ in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(
                x,
                report_texts=reports_batch,
                pathology_targets=y,
                max_target_length=args.max_target_length,
            )
            loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            opt.step()
            scheduler.step()
            global_step += 1
            losses.append(float(loss.detach().cpu()))

            ckpt.maybe_step(
                model, opt, scheduler, epoch, global_step, best,
                extra={
                    "vision_params": vision_params,
                    "lm_name": args.lm_name,
                    "target_shape": tuple(args.target_shape),
                    "token_grid": tuple(args.token_grid),
                    "visual_tokens": args.visual_tokens,
                    "labels": LABELS,
                },
            )

        val_loss, val_text, val_path = validate(
            model, val_dl, device, args.max_target_length
        )
        best, improved = ckpt.end_epoch(
            model, opt, scheduler, epoch, global_step, val_loss, best,
            extra={
                "vision_params": vision_params,
                "lm_name": args.lm_name,
                "target_shape": tuple(args.target_shape),
                "token_grid": tuple(args.token_grid),
                "visual_tokens": args.visual_tokens,
                "labels": LABELS,
            },
        )
        print(
            f"epoch={epoch+1} train_loss={np.mean(losses):.5f} "
            f"val_loss={val_loss:.5f} val_text={val_text:.5f} "
            f"val_path={val_path:.5f} best={best:.5f} improved={improved}"
        )

    # Save the LoRA adapter/tokenizer separately as a convenient export.
    export = Path(args.out) / "export"
    export.mkdir(parents=True, exist_ok=True)
    model.tokenizer.save_pretrained(export / "tokenizer")
    if hasattr(model.lm, "save_pretrained"):
        model.lm.save_pretrained(export / "lm_adapter_or_model")
    print("best checkpoint:", Path(args.out) / "best.pt")
    print("export:", export)


if __name__ == "__main__":
    main()

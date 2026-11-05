#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader

import model_code_from_original as mc
from sota_models import InteractingLabelTokenClassifier
from training_utils import (
    CheckpointManager,
    get_cosine_with_warmup,
    patient_group_from_volume_name,
)

LABELS = mc.ABNORMALITY_LABELS


@torch.no_grad()
def validate(model, dl, device, loss_fn):
    model.eval()
    losses = []
    for x, y, _ in dl:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        losses.append(float(loss_fn(logits, y).detach().cpu()))
    return float(np.mean(losses))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-csv", required=True)
    ap.add_argument("--out", default="./checkpoints/sota_task1")
    ap.add_argument("--max-rows", type=int, default=10000)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--val-cases", type=int, default=384)
    ap.add_argument("--target-shape", type=int, nargs=3, default=[64,96,96])
    ap.add_argument("--token-grid", type=int, nargs=3, default=[2,4,4])
    ap.add_argument("--feature-size", type=int, default=24)
    ap.add_argument("--depths", type=int, nargs=4, default=[1,1,1,1])
    ap.add_argument("--num-heads", type=int, nargs=4, default=[3,6,12,24])
    ap.add_argument("--d-model", type=int, default=384)
    ap.add_argument("--label-heads", type=int, default=8)
    ap.add_argument("--label-layers", type=int, default=2)
    ap.add_argument("--vision-lr", type=float, default=1e-5)
    ap.add_argument("--head-lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--cache-dir", default="../vlm3d_original_task_split/ct_rate_volume_cache")
    ap.add_argument("--checkpoint-every", type=int, default=1000)
    ap.add_argument("--use-checkpoint", action="store_true", default=True)
    args = ap.parse_args()

    df = pd.read_csv(args.labels_csv).copy()
    df = df.dropna(subset=["VolumeName"]).reset_index(drop=True)

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
    if args.val_cases is not None:
        val_df = val_df.head(args.val_cases)

    vision_params = dict(
        feature_size=args.feature_size,
        in_channels=1,
        out_channels=1,
        spatial_dims=3,
        depths=tuple(args.depths),
        num_heads=tuple(args.num_heads),
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.1,
        use_checkpoint=args.use_checkpoint,
    )

    train_ds = mc.CTAbnormalityDataset(
        train_df,
        target_shape=tuple(args.target_shape),
        cache_dir=args.cache_dir,
        max_rows=None,
    )
    val_ds = mc.CTAbnormalityDataset(
        val_df,
        target_shape=tuple(args.target_shape),
        cache_dir=args.cache_dir,
        max_rows=None,
    )
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InteractingLabelTokenClassifier(
        vision_params=vision_params,
        num_labels=len(LABELS),
        token_grid=tuple(args.token_grid),
        d_model=args.d_model,
        nhead=args.label_heads,
        label_layers=args.label_layers,
    ).to(device)

    # Initialize LazyLinear before building optimizer groups.
    with torch.no_grad():
        dummy = torch.zeros(1,1,*args.target_shape, device=device)
        _ = model(dummy)

    vision_params_list = list(model.vision.parameters())
    vision_ids = {id(p) for p in vision_params_list}
    head_params = [p for p in model.parameters() if id(p) not in vision_ids]

    opt = torch.optim.AdamW([
        {"params": vision_params_list, "lr": args.vision_lr},
        {"params": head_params, "lr": args.head_lr},
    ], weight_decay=args.weight_decay)

    total_steps = max(1, len(train_dl) * args.epochs)
    scheduler = get_cosine_with_warmup(opt, total_steps, args.warmup_ratio)

    Y = train_df[LABELS].fillna(0).astype(float).to_numpy()
    pos = Y.sum(axis=0)
    neg = len(Y) - pos
    pos_weight = np.sqrt((neg + 1) / (pos + 1))
    pos_weight = np.clip(pos_weight, 1.0, 5.0)
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device)
    )

    ckpt = CheckpointManager(args.out, every_steps=args.checkpoint_every)
    best = float("inf")
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        losses = []
        for x, y, _ in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()
            global_step += 1
            losses.append(float(loss.detach().cpu()))
            ckpt.maybe_step(
                model, opt, scheduler, epoch, global_step, best,
                extra={"vision_params": vision_params, "target_shape": tuple(args.target_shape), "labels": LABELS},
            )

        val_loss = validate(model, val_dl, device, loss_fn)
        best, improved = ckpt.end_epoch(
            model, opt, scheduler, epoch, global_step, val_loss, best,
            extra={"vision_params": vision_params, "target_shape": tuple(args.target_shape), "labels": LABELS},
        )
        print(
            f"epoch={epoch+1} train_loss={np.mean(losses):.5f} "
            f"val_loss={val_loss:.5f} best={best:.5f} improved={improved}"
        )

    print("best checkpoint:", Path(args.out) / "best.pt")
    print("last checkpoint:", Path(args.out) / "last.pt")


if __name__ == "__main__":
    main()

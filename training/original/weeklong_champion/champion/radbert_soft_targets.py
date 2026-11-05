#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoConfig, AutoModel, AutoTokenizer

from common import LABELS, TRAIN_REPORTS, VALID_REPORTS, WORK, full_report_text

RADBERT_ID = "zzxslp/RadBERT-RoBERTa-4m"
CTRATE_REPO = "ibrahimhamamci/CT-RATE"


class RadBertClassifier(nn.Module):
    def __init__(self, base_path: str | Path, n_classes=18):
        super().__init__()
        base_path = str(base_path)
        cfg = AutoConfig.from_pretrained(base_path, local_files_only=True)
        self.model = AutoModel.from_pretrained(base_path, config=cfg, local_files_only=True)
        self.classifier = nn.Linear(self.model.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = getattr(out, "pooler_output", None)
        if pooled is None:
            pooled = out.last_hidden_state[:, 0]
        return self.classifier(pooled)


def download_assets(asset_dir: Path):
    asset_dir.mkdir(parents=True, exist_ok=True)
    base_dir = asset_dir / "RadBERT-RoBERTa-4m"
    if not (base_dir / "config.json").exists():
        snapshot_download(
            RADBERT_ID,
            local_dir=str(base_dir),
            token=os.getenv("HF_TOKEN") or None,
            ignore_patterns=["*.h5", "*.msgpack", "*.onnx"],
        )

    ckpt = asset_dir / "RadBertClassifier.pth"
    if not ckpt.exists():
        errors = []
        for subfolder in ("models", "models/CT-CLIP-Related"):
            try:
                got = hf_hub_download(
                    repo_id=CTRATE_REPO,
                    repo_type="dataset",
                    filename="RadBertClassifier.pth",
                    subfolder=subfolder,
                    token=os.getenv("HF_TOKEN") or None,
                )
                import shutil
                shutil.copy2(got, ckpt)
                break
            except Exception as e:
                errors.append(f"{subfolder}: {e}")
        if not ckpt.exists():
            raise RuntimeError("Could not obtain official RadBERT classifier:\n" + "\n".join(errors))
    return base_dir, ckpt


def load_radbert(asset_dir: Path, device: torch.device):
    base_dir, ckpt = download_assets(asset_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(base_dir), local_files_only=True, do_lower_case=True)
    model = RadBertClassifier(base_dir, n_classes=len(LABELS))
    try:
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
    except Exception:
        state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict) and "model" in state and not any(k.startswith("model.") for k in state):
        state = state["model"]
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return tokenizer, model


@torch.inference_mode()
def infer_df(df: pd.DataFrame, tokenizer, model, device, batch_size=64):
    texts = full_report_text(df).tolist()
    out = np.zeros((len(df), len(LABELS)), dtype=np.float32)
    for start in range(0, len(df), batch_size):
        batch = texts[start:start+batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True,
        )
        logits = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
        out[start:start+len(batch)] = torch.sigmoid(logits).float().cpu().numpy()
        if start == 0 or (start // batch_size) % 50 == 0:
            print(f"RadBERT {start+len(batch):,}/{len(df):,}", flush=True)
    result = pd.DataFrame(out, columns=LABELS)
    result.insert(0, "VolumeName", df["VolumeName"].astype(str).values)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(WORK / "radbert"))
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train_out = out / "train_soft_targets.csv"
    valid_out = out / "valid_soft_targets.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("RadBERT device:", device)
    tok, model = load_radbert(out / "assets", device)

    if not train_out.exists():
        train = pd.read_csv(TRAIN_REPORTS).drop_duplicates("VolumeName").reset_index(drop=True)
        infer_df(train, tok, model, device, args.batch_size).to_csv(train_out, index=False)
    else:
        print("Already exists:", train_out)

    if not valid_out.exists():
        valid = pd.read_csv(VALID_REPORTS).drop_duplicates("VolumeName").reset_index(drop=True)
        infer_df(valid, tok, model, device, args.batch_size).to_csv(valid_out, index=False)
    else:
        print("Already exists:", valid_out)

    print("RADBERT SOFT TARGETS COMPLETE")
    print(train_out)
    print(valid_out)


if __name__ == "__main__":
    main()

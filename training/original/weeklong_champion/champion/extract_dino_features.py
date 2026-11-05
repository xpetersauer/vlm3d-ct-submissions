#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoImageProcessor, AutoModel

HERE = Path(__file__).resolve().parent
PROJECT = Path("/workspace/vlm3d_original_task_split")
sys.path.insert(0, str(PROJECT))
import model_code_from_original as mc

from common import TRAIN_LABELS, VALID_LABELS, WORK, load_labels

DINO_ID = "facebook/dinov2-base"
VIEWS = [
    ("all", -1000.0, 1000.0),
    ("soft", -150.0, 250.0),
    ("lung", -1000.0, 400.0),
]


def local_dino(asset_root: Path):
    d = asset_root / "dinov2-base"
    if not (d / "config.json").exists():
        snapshot_download(DINO_ID, local_dir=str(d), token=os.getenv("HF_TOKEN") or None)
    return d


def case_tokens(x: torch.Tensor, n_slices: int, size: int, mean, std):
    # Existing compact cache stores x == clipped_HU / 1000 at shape [1,D,H,W].
    if x.ndim == 4:
        x = x[0]
    d = x.shape[0]
    inds = torch.linspace(0, d - 1, steps=n_slices).round().long().unique(sorted=True)
    hu = x[inds].float() * 1000.0  # [S,H,W]
    imgs = []
    # Every window becomes a token sequence; this is fixed multi-view representation,
    # not test-time averaging.
    for _, lo, hi in VIEWS:
        v = ((hu - lo) / (hi - lo)).clamp(0, 1)
        v = v[:, None].repeat(1, 3, 1, 1)
        v = F.interpolate(v, size=(size, size), mode="bilinear", align_corners=False)
        v = (v - mean[None, :, None, None]) / std[None, :, None, None]
        imgs.append(v)
    return torch.cat(imgs, dim=0)  # [3*S,3,size,size]


@torch.inference_mode()
def build_split(df: pd.DataFrame, split: str, out: Path, model, processor, device, n_slices=24, image_size=224, image_batch=96):
    n_tokens = len(VIEWS) * n_slices
    dim = int(model.config.hidden_size)
    mmap_path = out / f"{split}_features.f16.mmap"
    done_path = out / f"{split}_done.npy"
    index_path = out / f"{split}_index.csv"

    if not index_path.exists():
        df[["VolumeName"]].to_csv(index_path, index=False)
    else:
        old = pd.read_csv(index_path)
        if old["VolumeName"].astype(str).tolist() != df["VolumeName"].astype(str).tolist():
            raise RuntimeError(f"{index_path} exists but row order differs; refusing to corrupt feature cache")

    shape = (len(df), n_tokens, dim)
    feats = np.memmap(mmap_path, dtype=np.float16, mode="r+" if mmap_path.exists() else "w+", shape=shape)
    done = np.load(done_path) if done_path.exists() else np.zeros(len(df), dtype=np.uint8)
    if len(done) != len(df):
        raise RuntimeError("done-mask length mismatch")

    mean = torch.tensor(processor.image_mean, dtype=torch.float32)
    std = torch.tensor(processor.image_std, dtype=torch.float32)

    for i, name in enumerate(df["VolumeName"].astype(str)):
        if done[i]:
            continue
        x = mc.get_deadline_cached_ct(name, target_shape=(64,96,96), hf_token=None)
        images = case_tokens(x, n_slices, image_size, mean, std)
        chunks = []
        for s in range(0, len(images), image_batch):
            pix = images[s:s+image_batch].to(device=device, dtype=next(model.parameters()).dtype, non_blocking=True)
            o = model(pixel_values=pix)
            z = o.last_hidden_state[:, 0].float().cpu()
            chunks.append(z)
        z = torch.cat(chunks, 0).numpy().astype(np.float16)
        if z.shape != (n_tokens, dim):
            raise RuntimeError(f"Unexpected feature shape {z.shape}, expected {(n_tokens,dim)}")
        feats[i] = z
        feats.flush()
        done[i] = 1
        if i % 25 == 0 or i + 1 == len(df):
            np.save(done_path, done)
            print(f"{split}: {int(done.sum()):,}/{len(done):,} cases complete", flush=True)
    np.save(done_path, done)
    print(f"{split} FEATURE EXTRACTION COMPLETE: {mmap_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(WORK / "dino_features"))
    ap.add_argument("--n-slices", type=int, default=24)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--image-batch", type=int, default=96)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DINO feature device:", device)
    asset = local_dino(WORK / "assets")
    processor = AutoImageProcessor.from_pretrained(str(asset), local_files_only=True)
    model = AutoModel.from_pretrained(str(asset), local_files_only=True).to(device).eval()
    if device.type == "cuda":
        model = model.to(dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)

    train = load_labels(TRAIN_LABELS)
    valid = load_labels(VALID_LABELS)
    build_split(train, "train", out, model, processor, device, args.n_slices, args.image_size, args.image_batch)
    build_split(valid, "valid", out, model, processor, device, args.n_slices, args.image_size, args.image_batch)
    print("DINO FEATURE STAGE COMPLETE")


if __name__ == "__main__":
    main()

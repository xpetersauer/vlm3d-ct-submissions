#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from common import (
    LABELS, TRAIN_LABELS, VALID_LABELS, WORK, alternate_reconstruction_index,
    load_labels, metrics, save_prediction_npz, validation_partition, write_json,
)
from models import AsymmetricLoss, CPUEMA, QueryPoolClassifier


class FeatureDataset(Dataset):
    def __init__(self, mmap_path, n, n_tokens, dim, y, alt=None, soft=None):
        self.path = str(mmap_path)
        self.shape = (n, n_tokens, dim)
        self.mm = np.memmap(self.path, dtype=np.float16, mode="r", shape=self.shape)
        self.y = np.asarray(y, dtype=np.float32)
        self.alt = np.asarray(alt, dtype=np.int64) if alt is not None else np.full(n, -1, dtype=np.int64)
        self.soft = np.asarray(soft, dtype=np.float32) if soft is not None else np.full_like(self.y, np.nan)

    def __len__(self): return len(self.y)

    def __getitem__(self, i):
        x = torch.from_numpy(np.array(self.mm[i], dtype=np.float32, copy=True))
        return x, torch.from_numpy(self.y[i]), torch.from_numpy(self.soft[i]), torch.tensor(int(i), dtype=torch.long)


@torch.inference_mode()
def predict(model, mmap_path, n, n_tokens, dim, device, batch=256):
    mm = np.memmap(mmap_path, dtype=np.float16, mode="r", shape=(n,n_tokens,dim))
    out = np.zeros((n, len(LABELS)), dtype=np.float32)
    for s in range(0, n, batch):
        x = torch.from_numpy(np.array(mm[s:s+batch], dtype=np.float32, copy=True)).to(device)
        out[s:s+len(x)] = torch.sigmoid(model(x)).float().cpu().numpy()
    return out


def aligned_soft(path: Path, names):
    if not path.exists():
        return None
    df = pd.read_csv(path)
    lut = df.set_index("VolumeName")
    missing = [n for n in names if n not in lut.index]
    if missing:
        print(f"Soft targets missing {len(missing):,} cases; report supervision disabled.")
        return None
    return lut.loc[list(names), LABELS].to_numpy(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["bce","asl","full"], required=True)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-tokens", type=int, default=72)
    ap.add_argument("--feature-dim", type=int, default=768)
    ap.add_argument("--report-weight", type=float, default=.20)
    ap.add_argument("--recon-weight", type=float, default=.10)
    ap.add_argument("--recon-every", type=int, default=4)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    featdir = WORK / "dino_features"
    train = load_labels(TRAIN_LABELS); valid = load_labels(VALID_LABELS)
    ytr = train[LABELS].to_numpy(np.float32); yv = valid[LABELS].to_numpy(np.float32)
    names_tr = train["VolumeName"].astype(str).tolist(); names_v = valid["VolumeName"].astype(str).tolist()
    soft = aligned_soft(WORK / "radbert/train_soft_targets.csv", names_tr) if args.variant == "full" else None
    alt = alternate_reconstruction_index(names_tr) if args.variant == "full" else None

    ds = FeatureDataset(featdir / "train_features.f16.mmap", len(train), args.n_tokens, args.feature_dim, ytr, alt=alt, soft=soft)
    g = torch.Generator(); g.manual_seed(args.seed)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, generator=g, num_workers=2, pin_memory=True, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QueryPoolClassifier(args.feature_dim, d_model=384, nhead=8, num_labels=len(LABELS)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    total_steps = max(1, len(dl) * args.epochs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=args.lr*0.05)
    asl = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=.05)
    ema = CPUEMA(model, decay=.999) if args.variant in {"asl","full"} else None

    part = validation_partition(names_v)
    dev = (part == "modeldev") & (np.arange(len(valid)) >= 500)
    best_auc = -1.0; best_state = None; stale = 0; step = 0
    history = []

    for epoch in range(args.epochs):
        model.train(); losses=[]
        for x, y, q, idx_batch in dl:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            if args.variant == "bce":
                loss = F.binary_cross_entropy_with_logits(logits, y)
            else:
                loss = asl(logits, y)

            if args.variant == "full" and soft is not None:
                q = q.to(device, non_blocking=True)
                loss = loss + args.report_weight * F.binary_cross_entropy_with_logits(logits, q)

            if args.variant == "full" and step % args.recon_every == 0:
                idx_np = idx_batch.cpu().numpy().astype(int); alt_np = alt[idx_np]; pos = np.where(alt_np >= 0)[0]
                if len(pos):
                    xa_np = np.array(ds.mm[alt_np[pos]], dtype=np.float32, copy=True)
                    xa = torch.from_numpy(xa_np).to(device, non_blocking=True)
                    pa = torch.sigmoid(model(xa)); pos_t = torch.as_tensor(pos, device=device, dtype=torch.long)
                    p = torch.sigmoid(logits[pos_t])
                    loss = loss + args.recon_weight * F.mse_loss(p, pa)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            if ema is not None: ema.update(model)
            losses.append(float(loss.detach().cpu()))

        eval_model = copy.deepcopy(model).to(device).eval()
        if ema is not None: ema.copy_to(eval_model)
        probs = predict(eval_model, featdir / "valid_features.f16.mmap", len(valid), args.n_tokens, args.feature_dim, device)
        met = metrics(yv[dev], probs[dev])
        row = {"epoch":epoch+1,"train_loss":float(np.mean(losses)),**{k:v for k,v in met.items() if k!="per_label"}}
        history.append(row); print(json.dumps(row), flush=True)
        if met["macro_auroc"] > best_auc + 1e-5:
            best_auc = met["macro_auroc"]; stale=0
            best_state = {k:v.detach().cpu().clone() for k,v in eval_model.state_dict().items()}
            torch.save({
                "model":best_state,"variant":args.variant,"seed":args.seed,"epoch":epoch+1,
                "best_modeldev_auroc":best_auc,"feature_dim":args.feature_dim,"n_tokens":args.n_tokens,
                "dino_id":"facebook/dinov2-base","views":["all","soft","lung"],"n_slices":args.n_tokens//3,
                "labels":LABELS,"report_weight":args.report_weight if soft is not None else 0.0,
                "recon_weight":args.recon_weight if args.variant=="full" else 0.0,
            }, outdir / "best.pt")
        else:
            stale += 1
        del eval_model
        if stale >= args.patience:
            print(f"Early stopping after {epoch+1}; best modeldev AUROC={best_auc:.5f}")
            break

    if best_state is None: raise RuntimeError("No best state was saved")
    model.load_state_dict(best_state); model.eval()
    probs = predict(model, featdir / "valid_features.f16.mmap", len(valid), args.n_tokens, args.feature_dim, device)
    save_prediction_npz(outdir / "valid_predictions.npz", names_v, probs)
    pd.DataFrame(history).to_csv(outdir / "history.csv", index=False)
    allmet = metrics(yv, probs); write_json({k:v for k,v in allmet.items() if k!="per_label"}, outdir / "valid_metrics.json")
    pd.DataFrame(allmet["per_label"]).to_csv(outdir / "valid_per_label.csv", index=False)
    print("DINO HEAD COMPLETE", outdir)


if __name__ == "__main__": main()

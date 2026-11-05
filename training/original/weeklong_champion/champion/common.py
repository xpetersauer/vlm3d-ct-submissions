from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score

LABELS = [
    "Medical material",
    "Arterial wall calcification",
    "Cardiomegaly",
    "Pericardial effusion",
    "Coronary artery wall calcification",
    "Hiatal hernia",
    "Lymphadenopathy",
    "Emphysema",
    "Atelectasis",
    "Lung nodule",
    "Lung opacity",
    "Pulmonary fibrotic sequela",
    "Pleural effusion",
    "Mosaic attenuation pattern",
    "Peribronchial thickening",
    "Consolidation",
    "Bronchiectasis",
    "Interlobular septal thickening",
]

ROOT = Path("/workspace/vlm3d_original_task_split")
WORK = Path("/workspace/vlm3d_weeklong")

TRAIN_LABELS = ROOT / "CT-RATE/dataset/multi_abnormality_labels/train_predicted_labels.csv"
VALID_LABELS = ROOT / "CT-RATE/dataset/multi_abnormality_labels/valid_predicted_labels.csv"
TRAIN_REPORTS = ROOT / "CT-RATE/dataset/radiology_text_reports/train_reports.csv"
VALID_REPORTS = ROOT / "CT-RATE/dataset/radiology_text_reports/validation_reports.csv"


def strip_ext(x: str) -> str:
    name = Path(str(x)).name
    for suffix in (".nii.gz", ".nii", ".npy"):
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return Path(name).stem


def patient_key(x: str) -> str:
    p = strip_ext(x).split("_")
    return "_".join(p[:2]) if len(p) >= 2 else strip_ext(x)


def scan_key(x: str) -> str:
    """CT-RATE naming is split_patient_scan_reconstruction; first 3 fields identify scan."""
    p = strip_ext(x).split("_")
    return "_".join(p[:3]) if len(p) >= 3 else strip_ext(x)


def deterministic_bucket(key: str, seed: str = "vlm3d-weeklong-v1", modulo: int = 10000) -> int:
    h = hashlib.sha1(f"{seed}|{key}".encode()).hexdigest()
    return int(h[:12], 16) % modulo


def validation_partition(volume_names: Sequence[str]) -> np.ndarray:
    """Patient-grouped deterministic split: 40/20/15/15/10 percent."""
    out = []
    for v in volume_names:
        b = deterministic_bucket(patient_key(v)) / 10000.0
        if b < .40:
            out.append("modeldev")
        elif b < .60:
            out.append("selection")
        elif b < .75:
            out.append("graph")
        elif b < .90:
            out.append("calibration")
        else:
            out.append("audit")
    return np.asarray(out, dtype=object)


def load_labels(path: Path | str) -> pd.DataFrame:
    df = pd.read_csv(path).drop_duplicates("VolumeName").reset_index(drop=True)
    for c in LABELS:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(np.float32)
    return df


def full_report_text(df: pd.DataFrame) -> pd.Series:
    f = df.get("Findings_EN", pd.Series([""] * len(df), index=df.index)).fillna("").astype(str)
    i = df.get("Impressions_EN", pd.Series([""] * len(df), index=df.index)).fillna("").astype(str)
    return ("FINDINGS: " + f.str.strip() + "\n\nIMPRESSION: " + i.str.strip()).str.strip()


def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, -40, 40)
    return 1.0 / (1.0 + np.exp(-x))


def logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    return np.log(p / (1 - p))


def binary_entropy(p, eps=1e-6):
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    h = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    return h / math.log(2.0)


def metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float = .5) -> dict:
    y = np.asarray(y_true, dtype=np.int32)
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-6, 1 - 1e-6)
    pred = (p >= threshold).astype(np.int32)
    aucs, f1s = [], []
    per_label = []
    for j, label in enumerate(LABELS):
        yy, pp = y[:, j], p[:, j]
        auc = float("nan")
        if len(np.unique(yy)) > 1:
            auc = float(roc_auc_score(yy, pp))
            aucs.append(auc)
        f1 = float(f1_score(yy, pred[:, j], zero_division=0))
        f1s.append(f1)
        bce = float(-(yy * np.log(pp) + (1 - yy) * np.log(1 - pp)).mean())
        per_label.append({"label": label, "auroc": auc, "f1": f1, "bce": bce, "prevalence": float(yy.mean())})
    bce = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
    return {
        "macro_auroc": float(np.nanmean(aucs)) if aucs else float("nan"),
        "macro_f1": float(np.mean(f1s)),
        "bce": bce,
        "n": int(len(y)),
        "per_label": per_label,
    }


def align_predictions(names: Sequence[str], pred_names: Sequence[str], probs: np.ndarray) -> np.ndarray:
    lut = {strip_ext(n): i for i, n in enumerate(pred_names)}
    idx = []
    for n in names:
        k = strip_ext(n)
        if k not in lut:
            raise KeyError(f"Prediction missing case {n}")
        idx.append(lut[k])
    return np.asarray(probs)[idx]


def load_prediction_npz(path: Path | str, target_names: Sequence[str] | None = None):
    z = np.load(path, allow_pickle=True)
    names = z["names"].astype(str)
    probs = z["probs"].astype(np.float64)
    if target_names is not None:
        probs = align_predictions(target_names, names, probs)
        names = np.asarray(target_names, dtype=str)
    return names, probs


def save_prediction_npz(path: Path | str, names: Sequence[str], probs: np.ndarray):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, names=np.asarray(names, dtype=str), probs=np.asarray(probs, dtype=np.float32), labels=np.asarray(LABELS))


def alternate_reconstruction_index(volume_names: Sequence[str]) -> np.ndarray:
    groups: dict[str, list[int]] = {}
    for i, n in enumerate(volume_names):
        groups.setdefault(scan_key(n), []).append(i)
    alt = np.full(len(volume_names), -1, dtype=np.int64)
    for inds in groups.values():
        if len(inds) > 1:
            for pos, idx in enumerate(inds):
                alt[idx] = inds[(pos + 1) % len(inds)]
    return alt


def write_json(obj, path: Path | str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else str(x)))


def stage_status(stage: str, status: str, detail: str = ""):
    p = WORK / "stage_status.tsv"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(f"{pd.Timestamp.now().isoformat()}\t{stage}\t{status}\t{detail}\n")

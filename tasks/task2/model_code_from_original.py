"""
model_code.py

Safe, importable model/prediction code exported from trial_run_vlm3d_for_chat.ipynb.
This file intentionally avoids top-level training/prediction calls so it can be imported
from task entrypoints and Docker containers.
"""

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import nibabel as nib
from huggingface_hub import hf_hub_download

try:
    from IPython.display import display, Markdown
except Exception:
    display = print
    Markdown = lambda x: x

try:
    from monai.networks.nets import SwinUNETR
except Exception as e:
    SwinUNETR = None
    _MONAI_IMPORT_ERROR = e

try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel
    from transformers.modeling_outputs import BaseModelOutput
except Exception as e:
    AutoTokenizer = None
    AutoModelForSeq2SeqLM = None
    AutoModel = None
    BaseModelOutput = None
    _TRANSFORMERS_IMPORT_ERROR = e

try:
    from sklearn.metrics import roc_auc_score, f1_score
except Exception:
    roc_auc_score = None
    f1_score = None

TOKEN = os.getenv("HF_TOKEN", None)
LOCAL_DIR = os.getenv("LOCAL_DIR", "./CT-RATE")
REPO_ID = os.getenv("CT_RATE_REPO_ID", "ibrahimhamamci/CT-RATE")

CT_RATE_CSV_FILES = {
    "train": {
        "labels": LOCAL_DIR + "/" + "dataset/multi_abnormality_labels/train_predicted_labels.csv",
        "reports": LOCAL_DIR + "/" + "dataset/radiology_text_reports/train_reports.csv",
        "metadata": LOCAL_DIR + "/" + "dataset/metadata/train_metadata.csv",
    },
    "valid": {
        "labels": LOCAL_DIR + "/" + "dataset/multi_abnormality_labels/valid_predicted_labels.csv",
        "reports": LOCAL_DIR + "/" + "dataset/radiology_text_reports/validation_reports.csv",
        "metadata": LOCAL_DIR + "/" + "dataset/metadata/validation_metadata.csv",
    },
}

ABNORMALITY_LABELS = [
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

LABEL_ALIASES = {label.lower().replace(" ", "_"): label for label in ABNORMALITY_LABELS}

def volume_name_to_hf_parts(volume_name, fixed_split=True):
    """
    Example:
        train_1_a_1.nii.gz
    Actual HF path:
        dataset/train_fixed/train_1/train_1_a/train_1_a_1.nii.gz
    """
    stem = volume_name.removesuffix(".nii.gz").removesuffix(".nii")
    split_name, patient_id, scan_id, recon_id = stem.split("_")

    image_split = f"{split_name}_fixed" if fixed_split else split_name
    patient_folder = f"{split_name}_{patient_id}"
    scan_folder = f"{patient_folder}_{scan_id}"

    subfolder = f"dataset/{image_split}/{patient_folder}/{scan_folder}"
    filename = volume_name
    return subfolder, filename

def write_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))

def build_ct_abnormality_results(predictions):
    return {
        "name": "vlm3d",
        "type": "ct-abnormality-classification",
        "version": "0.1.0",
        "predictions": predictions,
    }

def load_volume(path):
    img = nib.load(path)
    volume = img.get_fdata(dtype=np.float32)

    info = {
        "affine": img.affine,
        "header": img.header.copy(),
        "spacing": img.header.get_zooms(),
        "shape": img.shape,
        "path": path,
    }
    return volume, info

def preprocess_ct_for_torch(path, target_shape):
    """Convert a CT volume to a normalized tensor shaped [1, D, H, W]."""
    volume, _ = load_volume(path)

    volume = np.nan_to_num(volume, nan=-1000.0, posinf=1000.0, neginf=-1000.0)
    volume = np.clip(volume, -1000.0, 1000.0)
    # [-1000, 1000] -> [-1, 1]
    volume = (volume + 1000.0) / 1000.0 - 1.0

    x = torch.from_numpy(volume.astype(np.float32))[None, None]
    x = torch.nn.functional.interpolate(x, size=target_shape, mode="trilinear", align_corners=False)
    # return x, but remove the temporary batch dimension
    return x[0]


class CTAbnormalityDataset:
  def __init__(self,
               df,
               target_shape,
               cache_dir="./ct_rate_volume_cache",
               hf_token=None,
               max_rows=None):
    self.df = df.reset_index(drop=True)

    if max_rows is not None:
      self.df = self.df.head(max_rows)

    self.cache_dir = Path(cache_dir)
    self.cache_dir.mkdir(parents=True, exist_ok=True)

    self.target_shape = target_shape
    self.hf_token = hf_token
    self.max_rows = max_rows

  def __len__(self):
    return len(self.df)

  def get_or_download_volume(self, volume_name):
    subfolder, filename = volume_name_to_hf_parts(volume_name)

    local_path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        token=self.hf_token,
        filename=filename,
        subfolder=subfolder,
        local_dir=str(self.cache_dir),
    )

    return Path(local_path)

  def __getitem__(self, idx):
    row = self.df.iloc[idx]
    volume_name = row["VolumeName"]
    volume_path = self.get_or_download_volume(volume_name)
    # preprocess and load ct:
    x = preprocess_ct_for_torch(
            volume_path,
            target_shape=self.target_shape,
        )

    y = row[ABNORMALITY_LABELS].fillna(0).astype(float).to_numpy()
    y = torch.tensor(y, dtype=torch.float32)

    return x, y, volume_name

def list_case_files(input_dir):
    input_dir = Path(input_dir)
    paths = []
    paths.extend(input_dir.glob("*.npy"))
    paths.extend(input_dir.glob("*.nii"))
    paths.extend(input_dir.glob("*.nii.gz"))
    return sorted(paths)

def case_id_from_path(path, include_extension=True):
    path = Path(path)

    if include_extension:
        return path.name

    if path.name.endswith(".nii.gz"):
        return path.name[:-7]

    return path.stem

class CTInferenceDataset(Dataset):
    """
    Dataset for inference only.

    It loads CT files from input_dir and returns:
        x: preprocessed tensor [1, D, H, W]
        case_id: filename or stem
    """

    def __init__(
        self,
        input_dir,
        target_shape,
        max_cases=None,
        include_extension=True,
    ):
        self.paths = list_case_files(input_dir)

        if max_cases is not None:
            self.paths = self.paths[:max_cases]

        self.target_shape = tuple(target_shape)
        self.include_extension = include_extension

        if len(self.paths) == 0:
            raise FileNotFoundError(
                f"No .npy, .nii, or .nii.gz files found in {input_dir}"
            )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]

        x = preprocess_ct_for_torch(
            path,
            target_shape=self.target_shape,
        )

        case_id = case_id_from_path(
            path,
            include_extension=self.include_extension,
        )

        return x, case_id


class SwinUNETRForMultiLabelClassification(nn.Module):

    def __init__(self, swinunetr_params, num_labels):
        super().__init__()
        assert SwinUNETR is not None, f"MONAI SwinUNETR import failed: {_MONAI_IMPORT_ERROR}"
        backbone = SwinUNETR(**swinunetr_params)
        self.encoder = backbone.swinViT
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.LazyLinear(num_labels)

    def forward(self, x):
        hidden = self.encoder(x)

        if isinstance(hidden, (list, tuple)):
            z = hidden[-1]
        else:
            z = hidden

        z = self.pool(z).flatten(1)
        logits = self.head(z)
        return logits


def save_task1_checkpoint(
    model,
    model_params,
    checkpoint_path,
    labels=ABNORMALITY_LABELS,
    target_shape=(64, 128, 128)
):
    """
    Save everything needed to reconstruct the model during Docker inference.

    The Docker container will not have your notebook variables.
    Therefore the checkpoint must include:
        - model weights
        - model_params
        - target_shape
        - label order
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model": model.state_dict(),
            "model_params": model_params,
            "target_shape": target_shape,
            "labels": labels,
        },
        checkpoint_path,
    )

    print(f"Saved checkpoint to: {checkpoint_path}")

    return checkpoint_path

def train_task1(model_params, df, batch_size, epochs, lr, num_workers, checkpoint_path,max_rows,cache_dir, num_labels, target_shape=(64, 128, 128)):
  df = df.reset_index(drop=True)
  ds = CTAbnormalityDataset(df, target_shape = target_shape, max_rows=max_rows,cache_dir=cache_dir)
  dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # TODO: modify swin_unet s.t. it has a classification backbone
  model = SwinUNETRForMultiLabelClassification(model_params, num_labels)
  opt = torch.optim.AdamW(model.parameters(), lr=lr)
  loss_fn = torch.nn.BCEWithLogitsLoss()

  model.to(device)

  for epoch in range(epochs):
      losses = []
      model.train()
      for x, y, _ in dl:
          x, y = x.to(device), y.to(device)
          opt.zero_grad(set_to_none=True)
          loss = loss_fn(model(x), y)
          loss.backward()
          opt.step()
          losses.append(float(loss.detach().cpu()))
          if len(losses) % 250 == 0:
              save_task1_checkpoint(
                  model, model_params, checkpoint_path,
                  labels=ABNORMALITY_LABELS, target_shape=target_shape
              )
              print(f"Task1 safety checkpoint epoch={epoch+1} step={len(losses)}")
      print(f"epoch={epoch + 1} loss={np.mean(losses):.4f}")

  checkpoint_path = save_task1_checkpoint(
      model, model_params, checkpoint_path,
      labels=ABNORMALITY_LABELS, target_shape=target_shape
  )

  return model, checkpoint_path

def save_task1_prediction_outputs(
    predictions,
    output_dir,
    true_df=None,
    results_filename="results.json",
    csv_filename="predictions_file_level.csv",
):
    """
    Save predictions in two formats:

    1. results.json
        Challenge-style submission output.

    2. predictions_file_level.csv
        Debuggable file-level table with one row per CT.

    If true_df is provided, this also merges true labels into the CSV.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Save challenge-style JSON ----
    results_obj = build_ct_abnormality_results(predictions)
    results_path = output_dir / results_filename
    write_json(results_obj, results_path)

    # ---- Save file-level CSV ----
    rows = []

    for item in predictions:
        row = {
            "VolumeName": item["input_image_name"],
        }

        for label, prob in item["probabilities"].items():
            key = label.lower().replace(" ", "_")
            row[f"prob_{key}"] = prob

        rows.append(row)

    pred_df = pd.DataFrame(rows)

    if true_df is not None and "VolumeName" in true_df.columns:
        keep_cols = ["VolumeName"] + [
            label for label in ABNORMALITY_LABELS
            if label in true_df.columns
        ]

        true_small = true_df[keep_cols].copy()

        rename_map = {
            label: f"true_{label.lower().replace(' ', '_')}"
            for label in ABNORMALITY_LABELS
            if label in true_small.columns
        }

        true_small = true_small.rename(columns=rename_map)

        pred_df = pred_df.merge(
            true_small,
            on="VolumeName",
            how="left",
        )

    csv_path = output_dir / csv_filename
    pred_df.to_csv(csv_path, index=False)

    return {
        "results_json": results_path,
        "predictions_csv": csv_path,
        "predictions_df": pred_df,
    }

def validate_task1(
    model,
    valid_df,
    output_dir,
    target_shape,
    batch_size=1,
    num_workers=0,
    cache_dir="./ct_rate_volume_cache",
    hf_token=None,
    threshold=0.5,
    max_rows=None
):
    """
    Run validation on labeled CT-RATE validation data.

    Steps:
        1. Load validation CTs using CTAbnormalityDataset.
        2. Run model inference.
        3. Save challenge-style results.json.
        4. Save file-level predictions CSV.
        5. Compute per-label AUROC/F1 metrics.
        6. Save metrics.csv.

    This mimics Docker output format locally, but also keeps true labels
    for debugging.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ds = CTAbnormalityDataset(
        df=valid_df,
        target_shape=target_shape,
        cache_dir=cache_dir,
        hf_token=hf_token,
        max_rows=max_rows
    )

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    device = next(model.parameters()).device
    loss_fn = torch.nn.BCEWithLogitsLoss()

    model.eval()

    losses = []
    all_probs = []
    all_true = []
    all_names = []

    predictions = []

    with torch.no_grad():
        for x, y, volume_names in dl:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = loss_fn(logits, y)

            probs = torch.sigmoid(logits)

            losses.append(float(loss.detach().cpu()))
            all_probs.append(probs.cpu().numpy())
            all_true.append(y.cpu().numpy())
            all_names.extend(list(volume_names))

            probs_np = probs.cpu().numpy()

            for volume_name, prob_vector in zip(volume_names, probs_np):
                predictions.append(
                    {
                        "input_image_name": volume_name,
                        "probabilities": {
                            label: round(float(prob), 4)
                            for label, prob in zip(ABNORMALITY_LABELS, prob_vector)
                        },
                    }
                )

    all_probs = np.concatenate(all_probs, axis=0)
    all_true = np.concatenate(all_true, axis=0)

    # ---- Save results.json and file-level CSV ----
    saved = save_task1_prediction_outputs(
        predictions=predictions,
        output_dir=output_dir,
        true_df=valid_df,
        results_filename="results.json",
        csv_filename="predictions_file_level.csv",
    )

    # ---- Compute metrics ----
    metric_rows = []

    for j, label in enumerate(ABNORMALITY_LABELS):
        y_true = all_true[:, j]
        y_prob = all_probs[:, j]
        y_pred = (y_prob >= threshold).astype(int)

        if len(np.unique(y_true)) > 1:
            auroc = roc_auc_score(y_true, y_prob)
        else:
            auroc = np.nan

        f1 = f1_score(y_true, y_pred, zero_division=0)

        metric_rows.append(
            {
                "label": label,
                "auroc": auroc,
                "f1": f1,
                "prevalence": float(y_true.mean()),
                "threshold": threshold,
            }
        )

    metrics_df = pd.DataFrame(metric_rows)
    metrics_path = output_dir / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    summary = {
        "loss": float(np.mean(losses)),
        "mean_auroc": float(metrics_df["auroc"].mean(skipna=True)),
        "mean_f1": float(metrics_df["f1"].mean(skipna=True)),
    }

    write_json(summary, output_dir / "summary.json")

    print(f"Saved validation results to: {output_dir}")
    print(summary)

    return {
        "summary": summary,
        "metrics_df": metrics_df,
        "predictions_df": saved["predictions_df"],
        "results_json": saved["results_json"],
        "predictions_csv": saved["predictions_csv"],
        "metrics_csv": metrics_path,
    }


def is_ct_file(path):
    """
    Check whether file is supported CT volume format.
    """
    name = Path(path).name.lower()
    return name.endswith(".npy") or name.endswith(".nii") or name.endswith(".nii.gz")

def list_ct_files(input_dir):
    """
    Recursively find CT image files in a folder.
    """
    input_dir = Path(input_dir)

    paths = [
        p for p in input_dir.rglob("*")
        if p.is_file() and is_ct_file(p)
    ]

    return sorted(paths)

def make_case_to_path(input_dir):
    """
    Build lookup:
        case_001.npy -> path
        case_001     -> path

    This helps match JSON case IDs to actual CT files.
    """
    paths = list_ct_files(input_dir)

    case_to_path = {}

    for p in paths:
        case_to_path[p.name] = p
        case_to_path[strip_ct_extension(p.name)] = p

    return case_to_path

def load_ct_volume(path):
    """
    Load .npy, .nii, or .nii.gz as a squeezed 3D numpy array.
    """
    path = Path(path)

    if path.suffix == ".npy":
        vol = np.load(path).astype(np.float32)

    elif path.name.endswith(".nii.gz") or path.suffix == ".nii":
        import nibabel as nib
        vol = nib.load(str(path)).get_fdata(dtype=np.float32)

    else:
        raise ValueError(f"Unsupported CT file: {path}")

    vol = np.squeeze(vol)

    if vol.ndim != 3:
        raise ValueError(f"Expected 3D CT after squeeze, got shape {vol.shape}")

    return vol.astype(np.float32)

def show_ct_slices(volume_or_path, title=None, vmin=-1000, vmax=500):
    """
    Show axial, coronal, sagittal center slices.
    """
    if isinstance(volume_or_path, (str, Path)):
        vol = load_ct_volume(volume_or_path)
    else:
        vol = np.squeeze(np.asarray(volume_or_path, dtype=np.float32))

    if vol.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {vol.shape}")

    z = vol.shape[0] // 2
    y = vol.shape[1] // 2
    x = vol.shape[2] // 2

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    axes[0].imshow(vol[z, :, :], cmap="gray", vmin=vmin, vmax=vmax)
    axes[0].set_title("Axial")

    axes[1].imshow(vol[:, y, :], cmap="gray", vmin=vmin, vmax=vmax)
    axes[1].set_title("Coronal")

    axes[2].imshow(vol[:, :, x], cmap="gray", vmin=vmin, vmax=vmax)
    axes[2].set_title("Sagittal")

    for ax in axes:
        ax.axis("off")

    if title:
        fig.suptitle(str(title))

    plt.tight_layout()
    plt.show()

def summarize_volume(path):
    """
    Numeric sanity check for one CT volume.
    """
    vol = load_ct_volume(path)

    return {
        "file": Path(path).name,
        "shape": tuple(vol.shape),
        "min": float(np.min(vol)),
        "max": float(np.max(vol)),
        "mean": float(np.mean(vol)),
        "std": float(np.std(vol)),
        "lung_fraction_<-500": float(np.mean(vol < -500)),
        "dense_fraction_>500": float(np.mean(vol > 500)),
    }

def strip_ct_extension(path_or_name):
    """
    Convert:
        case_001.npy -> case_001
        case_001.nii.gz -> case_001
    """
    name = Path(str(path_or_name)).name

    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    if name.endswith(".npy"):
        return name[:-4]

    return name

def read_json_manifest(json_path):
    """
    Read a flexible JSON manifest.

    Supports:
        [{"VolumeName": "..."}]
        [{"image_path": "..."}]
        {"inputs": [...]}
        {"cases": [...]}
        {"images": [...]}
        {"data": [...]}
    """
    json_path = Path(json_path)
    obj = json.loads(json_path.read_text())

    if isinstance(obj, list):
        items = obj
    elif isinstance(obj, dict):
        for key in ["inputs", "cases", "images", "data", "items", "prompts"]:
            if key in obj and isinstance(obj[key], list):
                items = obj[key]
                break
        else:
            items = [obj]
    else:
        raise ValueError(f"Unsupported JSON format in {json_path}")

    # If list of strings, treat them as filenames/paths.
    if len(items) > 0 and isinstance(items[0], str):
        df = pd.DataFrame({"path": items})
    else:
        df = pd.DataFrame(items)

    return df

def find_path_column(df):
    """
    Find a column that likely contains local image paths.
    """
    candidates = [
        "path",
        "filepath",
        "file_path",
        "image_path",
        "ct_path",
        "volume_path",
        "filename",
        "file",
        "input_image_name",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    return None

def manifest_to_local_file_items(df, manifest_parent, include_extension=True):
    """
    Try converting a CSV/JSON manifest into local CT file paths.

    Returns:
        list of {"path": path, "case_id": case_id}

    If paths do not exist locally, returns [].
    """
    path_col = find_path_column(df)

    if path_col is None:
        return []

    items = []
    manifest_parent = Path(manifest_parent)

    for _, row in df.iterrows():
        raw = row[path_col]

        if pd.isna(raw):
            continue

        raw = str(raw)

        candidate_paths = []

        raw_path = Path(raw)

        if raw_path.is_absolute():
            candidate_paths.append(raw_path)
        else:
            candidate_paths.append(manifest_parent / raw_path)
            candidate_paths.append(manifest_parent / "images" / raw_path.name)
            candidate_paths.append(manifest_parent.parent / "images" / raw_path.name)

        existing = None

        for p in candidate_paths:
            if p.exists() and is_ct_file(p):
                existing = p
                break

        if existing is None:
            continue

        if include_extension:
            case_id = existing.name
        else:
            case_id = strip_ct_extension(existing.name)

        items.append(
            {
                "path": existing,
                "case_id": case_id,
            }
        )

    return items

def standardize_manifest_df(df):
    """
    Normalize common manifest column names.

    For CT-RATE CSVs, the important column is VolumeName.
    """
    df = df.copy()

    rename_map = {}

    for c in df.columns:
        cl = c.lower()

        if cl in ["volumename", "volume_name", "volume", "series_name"]:
            rename_map[c] = "VolumeName"

    df = df.rename(columns=rename_map)

    return df

class CTFilePathPredictionDataset(Dataset):
    """
    Prediction dataset for local .npy/.nii/.nii.gz files.
    """

    def __init__(self, items, target_shape):
        self.items = items
        self.target_shape = tuple(target_shape)

        if len(self.items) == 0:
            raise FileNotFoundError("No local CT image files found.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]

        x = preprocess_ct_for_torch(
            item["path"],
            target_shape=self.target_shape,
        )

        return x, item["case_id"]

def build_prediction_dataset_from_any_input(
    input_source,
    target_shape,
    include_extension=True,
    max_cases=None,
    cache_dir="./ct_rate_volume_cache",
    hf_token=None,
):
    """
    Accepts:
        1. directory of CT files
        2. CSV with VolumeName
        3. JSON with VolumeName
        4. CSV/JSON with local image paths

    Returns:
        dataset
    """

    input_source = Path(input_source)

    if not input_source.exists():
        raise FileNotFoundError(f"Input source does not exist: {input_source}")

    # --------------------------------------------------
    # Case 1: directory of CT images
    # --------------------------------------------------
    if input_source.is_dir():
        ct_paths = list_ct_files(input_source)

        if len(ct_paths) == 0:
            # Maybe directory contains a manifest.
            manifests = (
                sorted(input_source.glob("*.json"))
                + sorted(input_source.glob("*.csv"))
            )

            if len(manifests) == 0:
                raise FileNotFoundError(
                    f"No .npy, .nii, .nii.gz, .csv, or .json files found in {input_source}"
                )

            input_source = manifests[0]
        else:
            if max_cases is not None:
                ct_paths = ct_paths[:max_cases]

            items = []

            for p in ct_paths:
                case_id = p.name if include_extension else strip_ct_extension(p.name)
                items.append({"path": p, "case_id": case_id})

            return CTFilePathPredictionDataset(
                items=items,
                target_shape=target_shape,
            )

    # --------------------------------------------------
    # Case 2: CSV manifest
    # --------------------------------------------------
    if input_source.suffix.lower() == ".csv":
        df = pd.read_csv(input_source)
        df = standardize_manifest_df(df)

    # --------------------------------------------------
    # Case 3: JSON manifest
    # --------------------------------------------------
    elif input_source.suffix.lower() == ".json":
        df = read_json_manifest(input_source)
        df = standardize_manifest_df(df)

    else:
        raise ValueError(
            f"Unsupported input_source: {input_source}. "
            "Expected directory, .csv, or .json."
        )

    if max_cases is not None:
        df = df.head(max_cases)

    # --------------------------------------------------
    # Case 4: manifest with local file paths
    # --------------------------------------------------
    local_items = manifest_to_local_file_items(
        df=df,
        manifest_parent=input_source.parent,
        include_extension=include_extension,
    )

    if len(local_items) > 0:
        if max_cases is not None:
            local_items = local_items[:max_cases]

        return CTFilePathPredictionDataset(
            items=local_items,
            target_shape=target_shape,
        )

    # --------------------------------------------------
    # Case 5: CT-RATE style manifest with VolumeName
    # --------------------------------------------------
    if "VolumeName" in df.columns:
        return CTAbnormalityDataset(
            df=df,
            target_shape=target_shape,
            cache_dir=cache_dir,
            hf_token=hf_token,
            max_rows=max_cases,
        )

    raise ValueError(
        "Could not interpret manifest. Expected either local image-path column "
        "or a VolumeName column. "
        f"Available columns: {list(df.columns)}"
    )


def predict_task1(
    input_dir,
    checkpoint_path,
    output_dir,
    batch_size=1,
    num_workers=0,
    max_cases=None,
    target_shape=None,
    include_extension=True,
    cache_dir="./ct_rate_volume_cache",
    hf_token=None,
    output_filename="results.json",
):
    """
    Flexible prediction function for your current Task 3 model.

    Despite the function name, this is:
        CT -> abnormality classification

    input_dir can be:
        1. directory of .npy/.nii/.nii.gz CT volumes
        2. CSV with VolumeName
        3. JSON with VolumeName
        4. CSV/JSON with local image paths
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    labels = ckpt.get("labels", globals().get("ABNORMALITY_LABELS", None))

    if labels is None:
        raise ValueError(
            "Could not find labels in checkpoint or global ABNORMALITY_LABELS."
        )

    model_params = (
        ckpt.get("model_params")
        or ckpt.get("swinunetr_params")
        or ckpt.get("vision_params")
    )

    if model_params is None:
        raise ValueError(
            "Could not find model params in checkpoint. "
            "Expected key: model_params, swinunetr_params, or vision_params."
        )

    if target_shape is None:
        target_shape = tuple(ckpt.get("target_shape", (64, 128, 128)))

    ds = build_prediction_dataset_from_any_input(
        input_source=input_dir,
        target_shape=target_shape,
        include_extension=include_extension,
        max_cases=max_cases,
        cache_dir=cache_dir,
        hf_token=hf_token,
    )

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use whatever classifier class exists in your notebook.
    if "SwinEncoderClassifier" in globals():
        model_cls = globals()["SwinEncoderClassifier"]
    elif "SwinUNETREncoderLinearClassifier" in globals():
        model_cls = globals()["SwinUNETREncoderLinearClassifier"]
    elif "SwinUNETRForMultiLabelClassification" in globals():
        model_cls = globals()["SwinUNETRForMultiLabelClassification"]
    else:
        raise NameError(
            "Could not find model class. Expected SwinEncoderClassifier "
            "or SwinUNETREncoderLinearClassifier or SwinUNETRForMultiLabelClassification to be defined."
        )

    # Different versions of the class may use different argument names.
    try:
        model = model_cls(
            swinunetr_params=model_params,
            num_labels=len(labels),
        )
    except TypeError:
        model = model_cls(
            model_params=model_params,
            num_labels=len(labels),
        )

    # LazyLinear sometimes needs one dummy forward before loading.
    try:
        model.load_state_dict(ckpt["model"], strict=True)
    except RuntimeError as e:
        print("Initial load_state_dict failed. Trying dummy forward for LazyLinear init.")
        print("Original error:", str(e)[:500])

        with torch.no_grad():
            dummy = torch.zeros(1, 1, *tuple(target_shape))
            _ = model(dummy)

        model.load_state_dict(ckpt["model"], strict=True)

    model = model.to(device)
    model.eval()

    predictions = []
    rows = []

    with torch.no_grad():
        for batch in dl:
            # Supports:
            #   CTFilePathPredictionDataset -> x, case_ids
            #   CTAbnormalityDataset        -> x, y, volume_names
            if isinstance(batch, (list, tuple)):
                x = batch[0]
                case_ids = batch[-1]
            else:
                raise ValueError("Unexpected batch format from dataset.")

            x = x.to(device)

            logits = model(x)
            probs = torch.sigmoid(logits).detach().cpu().numpy()

            for case_id, prob_vec in zip(case_ids, probs):
                case_id = str(case_id)

                prob_dict = {
                    label: round(float(prob), 6)
                    for label, prob in zip(labels, prob_vec)
                }

                predictions.append(
                    {
                        "input_image_name": case_id,
                        "probabilities": prob_dict,
                    }
                )

                rows.append(
                    {
                        "input_image_name": case_id,
                        **prob_dict,
                    }
                )

    out_obj = {
        "name": "task3-abnormality-classification",
        "type": "ct-abnormality-classification",
        "version": "0.1.0",
        "predictions": predictions,
    }

    json_path = output_dir / output_filename
    json_path.write_text(json.dumps(out_obj, indent=2))

    csv_path = output_dir / "predictions_file_level.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    print(f"Wrote JSON predictions to: {json_path}")
    print(f"Wrote CSV predictions to: {csv_path}")

    return {
        "json_path": json_path,
        "csv_path": csv_path,
        "predictions": predictions,
        "predictions_df": pd.DataFrame(rows),
    }


class SwinUNETRVisualTokenizer(nn.Module):
    """
    Converts a CT volume into a sequence of visual tokens.

    CT [B, 1, D, H, W]
        -> Swin encoder final feature map [B, C, d, h, w]
        -> adaptive pooling to fixed grid [B, C, gD, gH, gW]
        -> flatten spatial grid into tokens [B, N, C]

    These visual tokens are then projected into the language model hidden size.
    """

    def __init__(self, model_params, token_grid=(2, 4, 4)):
        super().__init__()

        self.swin_backbone = self._build_swinunetr(model_params)
        self.encoder = self.swin_backbone.swinViT
        self.normalize = getattr(self.swin_backbone, "normalize", True)
        self.token_grid = token_grid

    def _build_swinunetr(self, model_params):
        p = dict(model_params)

        try:
            return SwinUNETR(**p)
        except TypeError:
            p.pop("img_size", None)
            return SwinUNETR(**p)

    def forward(self, x):
        try:
            hidden_states = self.encoder(x, self.normalize)
        except TypeError:
            hidden_states = self.encoder(x)

        if isinstance(hidden_states, (list, tuple)):
            z = hidden_states[-1]
        else:
            z = hidden_states

        # z: [B, C, d, h, w]
        z = F.adaptive_avg_pool3d(z, self.token_grid)

        # [B, C, gD, gH, gW] -> [B, N, C]
        z = z.flatten(2).transpose(1, 2)

        return z


class CTVLMReportGenerator(nn.Module):
    """
    CT-CHAT-style report generator.

    This is a practical trainable approximation:
        CT -> Swin visual tokens -> projector -> T5 decoder -> text

    True CT-CHAT would swap:
        Swin encoder -> CT-CLIP vision encoder
        T5 -> Llama/Mistral-style LLM
        full fine-tuning -> projector + LoRA fine-tuning
    """

    def __init__(
        self,
        vision_params,
        lm_name="google/flan-t5-small",
        token_grid=(2, 4, 4),
        freeze_vision=False,
        freeze_lm=False,
    ):
        super().__init__()

        self.lm_name = lm_name
        self.token_grid = token_grid

        self.vision = SwinUNETRVisualTokenizer(
            model_params=vision_params,
            token_grid=token_grid,
        )

        self.lm = AutoModelForSeq2SeqLM.from_pretrained(lm_name)

        # T5 hidden size.
        d_model = self.lm.config.d_model

        # LazyLinear infers CT encoder channel dimension on first pass.
        self.projector = nn.LazyLinear(d_model)

        if freeze_vision:
            for p in self.vision.parameters():
                p.requires_grad = False

        if freeze_lm:
            for p in self.lm.parameters():
                p.requires_grad = False

    def encode_ct(self, x):
        """
        Convert CT into language-model-compatible encoder outputs.
        """
        visual_tokens = self.vision(x)              # [B, N, C]
        visual_tokens = self.projector(visual_tokens)  # [B, N, d_model]

        attention_mask = torch.ones(
            visual_tokens.shape[:2],
            dtype=torch.long,
            device=visual_tokens.device,
        )

        encoder_outputs = BaseModelOutput(
            last_hidden_state=visual_tokens,
        )

        return encoder_outputs, attention_mask

    def forward(self, x, labels):
        """
        Training forward pass.

        labels:
            tokenized target report section, with pad tokens replaced by -100.
        """
        encoder_outputs, attention_mask = self.encode_ct(x)

        out = self.lm(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            labels=labels,
        )

        return out

    @torch.no_grad()
    def generate_text(
        self,
        x,
        tokenizer,
        max_new_tokens=192,
        num_beams=2,
    ):
        """
        Inference: CT -> generated text.
        """
        self.eval()

        encoder_outputs, attention_mask = self.encode_ct(x)

        generated_ids = self.lm.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )

        text = tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )

        return text


class CTReportSectionDataset(Dataset):
    """
    Dataset for training one CT -> text-section model.

    For findings model:
        target_col = "Findings_EN"

    For impression model:
        target_col = "Impressions_EN"
    """

    def __init__(
        self,
        df,
        target_col,
        tokenizer,
        target_shape=(64, 96, 96),
        max_target_length=256,
        cache_dir="./ct_rate_volume_cache",
        hf_token=None,
        max_rows=None,
    ):
        self.df = df.reset_index(drop=True)

        if max_rows is not None:
            self.df = self.df.head(max_rows)

        self.target_col = target_col
        self.tokenizer = tokenizer
        self.target_shape = tuple(target_shape)
        self.max_target_length = max_target_length

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.hf_token = hf_token or os.getenv("HF_TOKEN")

    def __len__(self):
        return len(self.df)

    def get_or_download_volume(self, volume_name):
        subfolder, filename = volume_name_to_hf_parts(volume_name)

        local_path = hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            token=self.hf_token,
            filename=filename,
            subfolder=subfolder,
            local_dir=str(self.cache_dir),
            force_download=False,
        )

        return Path(local_path)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        volume_name = row["VolumeName"]
        volume_path = self.get_or_download_volume(volume_name)

        x = preprocess_ct_for_torch(
            volume_path,
            target_shape=self.target_shape,
        )

        target_text = row.get(self.target_col, "")
        target_text = "" if target_text is None else str(target_text)

        if target_text.strip() == "" or target_text.lower() == "nan":
            raise ValueError(f"No report section available for {volume_name}")
            target_text = "No report section available."

        tokenized = self.tokenizer(
            target_text,
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        labels = tokenized["input_ids"][0]

        # Hugging Face seq2seq convention:
        # label = -100 means "ignore this position in the loss".
        labels = labels.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return x, labels, volume_name


def train_task2_vlm_section(
    vision_params,
    train_df,
    target_col,
    checkpoint_dir,
    lm_name="google/flan-t5-small",
    target_shape=(64, 96, 96),
    token_grid=(2, 4, 4),
    batch_size=1,
    epochs=1,
    lr=2e-5,
    num_workers=0,
    cache_dir="./ct_rate_volume_cache",
    hf_token=None,
    max_rows=None,
    max_target_length=256,
    freeze_vision=False,
    freeze_lm=False,
):
    """
    Fine-tune one CT -> report-section VLM.

    Example:
        target_col="Findings_EN"
        target_col="Impressions_EN"

    Steps:
        1. Load CT volume.
        2. Encode CT into visual tokens.
        3. Project visual tokens into LM hidden dimension.
        4. Train LM to generate target report section with teacher forcing.
        5. Save checkpoint and tokenizer.
    """

    from transformers import AutoTokenizer

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(lm_name)

    ds = CTReportSectionDataset(
        df=train_df,
        target_col=target_col,
        tokenizer=tokenizer,
        target_shape=target_shape,
        max_target_length=max_target_length,
        cache_dir=cache_dir,
        hf_token=hf_token,
        max_rows=max_rows,
    )

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CTVLMReportGenerator(
        vision_params=vision_params,
        lm_name=lm_name,
        token_grid=token_grid,
        freeze_vision=freeze_vision,
        freeze_lm=freeze_lm,
    ).to(device)

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
    )

    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for epoch in range(epochs):
        model.train()
        losses = []

        for x, labels, volume_names in dl:
            x = x.to(device)
            labels = labels.to(device)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(x=x, labels=labels)
                loss = out.loss

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            losses.append(float(loss.detach().cpu()))
            if len(losses) % 250 == 0:
                torch.save({
                    "model": model.state_dict(),
                    "vision_params": vision_params,
                    "lm_name": lm_name,
                    "target_shape": tuple(target_shape),
                    "token_grid": tuple(token_grid),
                    "target_col": target_col,
                    "max_target_length": max_target_length,
                    "model_class": "CTVLMReportGenerator",
                }, checkpoint_dir / "model.pt")
                tokenizer.save_pretrained(checkpoint_dir)
                print(f"Task2 safety checkpoint {target_col} epoch={epoch+1} step={len(losses)}")

        print(
            f"section={target_col} "
            f"epoch={epoch + 1} "
            f"loss={np.mean(losses):.4f}"
        )

    torch.save(
        {
            "model": model.state_dict(),
            "vision_params": vision_params,
            "lm_name": lm_name,
            "target_shape": tuple(target_shape),
            "token_grid": tuple(token_grid),
            "target_col": target_col,
            "max_target_length": max_target_length,
            "model_class": "CTVLMReportGenerator",
        },
        checkpoint_dir / "model.pt",
    )

    tokenizer.save_pretrained(checkpoint_dir)

    print(f"Saved {target_col} VLM to {checkpoint_dir}")

    return model, tokenizer, checkpoint_dir


def build_task2_predictions(generated_reports):
    """
    Challenge-style Task 2 JSON object.

    This matches the mock-style predictions.json structure.
    """
    return [
        {
            "outputs": [
                {
                    "value": {
                        "name": "vlm3d-task2-dual-vlm",
                        "type": "ct-report-generation",
                        "version": {"value": 0.1},
                        "generated_reports": generated_reports,
                    }
                }
            ]
        }
    ]

def predict_task2_dual_vlm(
    input_dir,
    findings_checkpoint_dir,
    impression_checkpoint_dir,
    output_dir,
    max_cases=None,
    max_new_tokens_findings=192,
    max_new_tokens_impression=96,
):
    """
    Inference for Task 2.

    Steps:
        1. Load findings VLM.
        2. Load impression VLM.
        3. For each CT:
            CT -> findings text
            CT -> impression text
        4. Combine into one report.
        5. Save predictions.json.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    findings_model, findings_tokenizer, findings_ckpt = load_task2_vlm_checkpoint(
        findings_checkpoint_dir,
        device=device,
    )

    impression_model, impression_tokenizer, impression_ckpt = load_task2_vlm_checkpoint(
        impression_checkpoint_dir,
        device=device,
    )

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = list_case_files(input_dir)

    if max_cases is not None:
        paths = paths[:max_cases]

    target_shape = tuple(findings_ckpt["target_shape"])

    generated_reports = []

    with torch.no_grad():
        for path in paths:
            x = preprocess_ct_for_torch(
                path,
                target_shape=target_shape,
            )[None].to(device)

            findings = findings_model.generate_text(
                x,
                tokenizer=findings_tokenizer,
                max_new_tokens=max_new_tokens_findings,
            )[0]

            impression = impression_model.generate_text(
                x,
                tokenizer=impression_tokenizer,
                max_new_tokens=max_new_tokens_impression,
            )[0]

            report = (
                "FINDINGS: "
                + findings.strip()
                + "\n\nIMPRESSION: "
                + impression.strip()
            )

            case_id = case_id_from_path(path, include_extension=False)

            generated_reports.append(
                {
                    "input_image_name": case_id,
                    "report": report,
                    "findings": findings.strip(),
                    "impression": impression.strip(),
                }
            )

            print(f"Generated report for {case_id}")

    predictions_obj = build_task2_predictions(generated_reports)

    out_path = output_dir / "predictions.json"
    write_json(predictions_obj, out_path)

    pd.DataFrame(generated_reports).to_csv(
        output_dir / "predictions_file_level.csv",
        index=False,
    )

    print(f"Saved Task 2 predictions to {out_path}")

    return out_path


def load_task2_vlm_checkpoint(checkpoint_dir, device=None):
    checkpoint_dir = Path(checkpoint_dir)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_dir / "model.pt", map_location="cpu")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)

    model = CTVLMReportGenerator(
        vision_params=ckpt["vision_params"],
        lm_name=ckpt["lm_name"],
        token_grid=tuple(ckpt["token_grid"]),
    )

    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    model.eval()

    return model, tokenizer, ckpt


class CTReportToVolumeDataset(Dataset):
    """
    Dataset for Task 1: report -> CT volume.

    Each item returns:
        x: CT tensor [1, D, H, W]
        report_text: free-text report string
        volume_name: CT volume name
    """

    def __init__(
        self,
        df,
        target_shape=(64, 96, 96),
        cache_dir="./ct_rate_volume_cache",
        hf_token=None,
        max_rows=None,
        report_col="Report_EN",
    ):
        self.df = df.reset_index(drop=True)

        if max_rows is not None:
            self.df = self.df.head(max_rows)

        self.target_shape = tuple(target_shape)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.report_col = report_col

    def __len__(self):
        return len(self.df)

    def get_or_download_volume(self, volume_name):
        subfolder, filename = volume_name_to_hf_parts(volume_name)

        local_path = hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            token=self.hf_token,
            filename=filename,
            subfolder=subfolder,
            local_dir=str(self.cache_dir),
            force_download=False,
        )

        return Path(local_path)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        volume_name = row["VolumeName"]
        volume_path = self.get_or_download_volume(volume_name)

        x = preprocess_ct_for_torch(
            volume_path,
            target_shape=self.target_shape,
        )

        report_text = row.get(self.report_col, "")

        if report_text is None or str(report_text).lower() == "nan":
            report_text = ""

        report_text = str(report_text)

        return x, report_text, volume_name

def build_task3_report_df(reports_df):
    """
    Prepare reports dataframe for report -> CT generation.

    Task 1 input should be the most complete report text available.
    We combine Findings_EN and Impressions_EN into Report_EN.
    """

    df = reports_df.copy()

    df["Findings_EN"] = df["Findings_EN"].fillna("").astype(str)
    df["Impressions_EN"] = df["Impressions_EN"].fillna("").astype(str)

    df["Report_EN"] = (
        "FINDINGS: "
        + df["Findings_EN"].str.strip()
        + "\n\nIMPRESSION: "
        + df["Impressions_EN"].str.strip()
    )

    has_report = df["Report_EN"].str.strip().ne("FINDINGS: \n\nIMPRESSION:")

    df = df.loc[has_report].reset_index(drop=True)

    return df


class CTVolumeAutoencoder(nn.Module):
    """
    Simple 3D CT autoencoder.

    Encoder:
        CT [B, 1, D, H, W] -> latent vector [B, latent_dim]

    Decoder:
        latent vector [B, latent_dim] -> CT [B, 1, D, H, W]

    This is a baseline. Later, replace this with a VAE or latent diffusion model.
    """

    def __init__(
        self,
        target_shape=(64, 96, 96),
        latent_dim=512,
        base_channels=16,
    ):
        super().__init__()

        self.target_shape = tuple(target_shape)
        self.latent_dim = latent_dim
        self.base_channels = base_channels

        D, H, W = self.target_shape

        if D % 16 != 0 or H % 16 != 0 or W % 16 != 0:
            raise ValueError(
                "target_shape must be divisible by 16 for this simple autoencoder. "
                f"Got {self.target_shape}."
            )

        self.start_shape = (D // 16, H // 16, W // 16)

        c = base_channels

        self.encoder_conv = nn.Sequential(
            nn.Conv3d(1, c, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(c),
            nn.GELU(),

            nn.Conv3d(c, c * 2, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(c * 2),
            nn.GELU(),

            nn.Conv3d(c * 2, c * 4, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(c * 4),
            nn.GELU(),

            nn.Conv3d(c * 4, c * 8, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(c * 8),
            nn.GELU(),
        )

        flat_dim = c * 8 * self.start_shape[0] * self.start_shape[1] * self.start_shape[2]

        self.to_latent = nn.Linear(flat_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, flat_dim)

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose3d(c * 8, c * 4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm3d(c * 4),
            nn.GELU(),

            nn.ConvTranspose3d(c * 4, c * 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm3d(c * 2),
            nn.GELU(),

            nn.ConvTranspose3d(c * 2, c, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm3d(c),
            nn.GELU(),

            nn.ConvTranspose3d(c, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def encode(self, x):
        z = self.encoder_conv(x)
        z = z.flatten(1)
        z = self.to_latent(z)
        return z

    def decode(self, z):
        c = self.base_channels
        z = self.from_latent(z)
        z = z.view(
            z.shape[0],
            c * 8,
            self.start_shape[0],
            self.start_shape[1],
            self.start_shape[2],
        )
        x_hat = self.decoder_conv(z)
        return x_hat

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


def train_task3_autoencoder(
    train_df,
    checkpoint_path,
    target_shape=(64, 96, 96),
    latent_dim=512,
    base_channels=16,
    batch_size=1,
    epochs=5,
    lr=1e-4,
    num_workers=0,
    cache_dir="./ct_rate_volume_cache",
    hf_token=None,
    max_rows=None,
):
    """
    Stage A training.

    Goal:
        Learn an autoencoder whose decoder can generate plausible CT volumes.
    """

    ds = CTReportToVolumeDataset(
        df=train_df,
        target_shape=target_shape,
        cache_dir=cache_dir,
        hf_token=hf_token,
        max_rows=max_rows,
        report_col="Report_EN",
    )

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CTVolumeAutoencoder(
        target_shape=target_shape,
        latent_dim=latent_dim,
        base_channels=base_channels,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for epoch in range(epochs):
        model.train()
        losses = []

        for x, report_text, volume_names in dl:
            x = x.to(device)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                x_hat, z = model(x)

                # L1 tends to preserve sharper intensity structure than pure MSE.
                recon_loss = F.l1_loss(x_hat, x)

                loss = recon_loss

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            losses.append(float(loss.detach().cpu()))
            if len(losses) % 250 == 0:
                checkpoint_path = Path(checkpoint_path)
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "model": model.state_dict(),
                    "target_shape": tuple(target_shape),
                    "latent_dim": latent_dim,
                    "base_channels": base_channels,
                    "model_class": "CTVolumeAutoencoder",
                }, checkpoint_path)
                print(f"Task3 AE safety checkpoint epoch={epoch+1} step={len(losses)}")

        print(f"AE epoch={epoch + 1} recon_loss={np.mean(losses):.4f}")

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model": model.state_dict(),
            "target_shape": tuple(target_shape),
            "latent_dim": latent_dim,
            "base_channels": base_channels,
            "model_class": "CTVolumeAutoencoder",
        },
        checkpoint_path,
    )

    print(f"Saved autoencoder checkpoint to {checkpoint_path}")

    return model, checkpoint_path


class ReportToCTLatentMapper(nn.Module):
    """
    Text encoder + MLP mapper.

    Input:
        report text tokens

    Output:
        predicted CT latent vector [B, latent_dim]
    """

    def __init__(
        self,
        text_encoder_name="distilbert-base-uncased",
        latent_dim=512,
        freeze_text_encoder=True,
    ):
        super().__init__()

        self.text_encoder_name = text_encoder_name
        self.text_encoder = AutoModel.from_pretrained(text_encoder_name)

        hidden_size = self.text_encoder.config.hidden_size

        self.mapper = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, latent_dim),
        )

        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad = False

    def mean_pool_text(self, outputs, attention_mask):
        """
        Mean-pool token embeddings using the attention mask.
        """
        token_embeddings = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()

        pooled = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

        return pooled

    def forward(self, input_ids, attention_mask):
        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        text_embedding = self.mean_pool_text(outputs, attention_mask)

        z_pred = self.mapper(text_embedding)

        return z_pred


def load_task3_autoencoder_checkpoint(checkpoint_path, device=None):
    checkpoint_path = Path(checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    model = CTVolumeAutoencoder(
        target_shape=tuple(ckpt["target_shape"]),
        latent_dim=ckpt["latent_dim"],
        base_channels=ckpt["base_channels"],
    )

    model.load_state_dict(ckpt["model"])

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    return model, ckpt

def train_task3_report_to_latent(
    train_df,
    autoencoder_checkpoint_path,
    mapper_checkpoint_dir,
    text_encoder_name="distilbert-base-uncased",
    target_shape=(64, 96, 96),
    latent_dim=512,
    batch_size=1,
    epochs=5,
    lr=1e-4,
    num_workers=0,
    cache_dir="./ct_rate_volume_cache",
    hf_token=None,
    max_rows=None,
    max_text_length=256,
    freeze_text_encoder=True,
    recon_loss_weight=0.25,
):
    """
    Stage B training.

    Goal:
        Learn report -> CT latent.

    Loss has two parts:
        latent_loss:
            predicted latent should match the autoencoder's CT latent

        recon_loss:
            decoded predicted latent should reconstruct the paired CT

    The reconstruction term encourages predicted latents to decode into useful CT volumes.
    """

    mapper_checkpoint_dir = Path(mapper_checkpoint_dir)
    mapper_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    autoencoder, ae_ckpt = load_task3_autoencoder_checkpoint(
        autoencoder_checkpoint_path,
        device=device,
    )

    for p in autoencoder.parameters():
        p.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(text_encoder_name)

    mapper = ReportToCTLatentMapper(
        text_encoder_name=text_encoder_name,
        latent_dim=latent_dim,
        freeze_text_encoder=freeze_text_encoder,
    ).to(device)

    ds = CTReportToVolumeDataset(
        df=train_df,
        target_shape=target_shape,
        cache_dir=cache_dir,
        hf_token=hf_token,
        max_rows=max_rows,
        report_col="Report_EN",
    )

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    opt = torch.optim.AdamW(
        [p for p in mapper.parameters() if p.requires_grad],
        lr=lr,
    )

    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for epoch in range(epochs):
        mapper.train()
        losses = []

        for x, report_texts, volume_names in dl:
            x = x.to(device)

            tokenized = tokenizer(
                list(report_texts),
                padding=True,
                truncation=True,
                max_length=max_text_length,
                return_tensors="pt",
            )

            input_ids = tokenized["input_ids"].to(device)
            attention_mask = tokenized["attention_mask"].to(device)

            opt.zero_grad(set_to_none=True)

            with torch.no_grad():
                z_true = autoencoder.encode(x)

            with torch.cuda.amp.autocast(enabled=use_amp):
                z_pred = mapper(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                latent_loss = F.mse_loss(z_pred, z_true)

                x_pred = autoencoder.decode(z_pred)
                recon_loss = F.l1_loss(x_pred, x)

                loss = latent_loss + recon_loss_weight * recon_loss

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            losses.append(float(loss.detach().cpu()))
            if len(losses) % 250 == 0:
                torch.save({
                    "model": mapper.state_dict(),
                    "text_encoder_name": text_encoder_name,
                    "latent_dim": latent_dim,
                    "target_shape": tuple(target_shape),
                    "autoencoder_checkpoint": str(autoencoder_checkpoint_path),
                    "max_text_length": max_text_length,
                    "freeze_text_encoder": freeze_text_encoder,
                    "model_class": "ReportToCTLatentMapper",
                }, mapper_checkpoint_dir / "mapper.pt")
                tokenizer.save_pretrained(mapper_checkpoint_dir)
                print(f"Task3 mapper safety checkpoint epoch={epoch+1} step={len(losses)}")

        print(f"Mapper epoch={epoch + 1} loss={np.mean(losses):.4f}")

    torch.save(
        {
            "model": mapper.state_dict(),
            "text_encoder_name": text_encoder_name,
            "latent_dim": latent_dim,
            "target_shape": tuple(target_shape),
            "autoencoder_checkpoint": str(autoencoder_checkpoint_path),
            "max_text_length": max_text_length,
            "freeze_text_encoder": freeze_text_encoder,
            "model_class": "ReportToCTLatentMapper",
        },
        mapper_checkpoint_dir / "mapper.pt",
    )

    tokenizer.save_pretrained(mapper_checkpoint_dir)

    print(f"Saved report-to-latent mapper to {mapper_checkpoint_dir}")

    return mapper, tokenizer, mapper_checkpoint_dir


def normalized_to_hu(x):
    """
    Convert normalized [-1, 1] CT tensor back to approximate HU range [-1000, 1000].
    """
    x = np.asarray(x, dtype=np.float32)
    hu = (x + 1.0) * 1000.0 - 1000.0
    return hu.astype(np.float32)

def clean_case_id(x):
    x = str(x)

    if x.endswith(".nii.gz"):
        return x[:-7]
    if x.endswith(".nii"):
        return x[:-4]
    if x.endswith(".npy"):
        return x[:-4]

    return x

def row_to_report_prompt(row):
    """
    Convert one CT-RATE report row into a text prompt.
    """
    if "Report_EN" in row and pd.notna(row["Report_EN"]):
        return str(row["Report_EN"])

    findings = ""
    impression = ""

    if "Findings_EN" in row and pd.notna(row["Findings_EN"]):
        findings = str(row["Findings_EN"])

    if "Impressions_EN" in row and pd.notna(row["Impressions_EN"]):
        impression = str(row["Impressions_EN"])

    if findings.strip() or impression.strip():
        return (
            "FINDINGS: "
            + findings.strip()
            + "\n\nIMPRESSION: "
            + impression.strip()
        )

    for col in ["report", "prompt", "text", "input_text", "radiology_report"]:
        if col in row and pd.notna(row[col]):
            return str(row[col])

    return ""

def read_task3_prompts(input_path_or_df):
    """
    Supports:
        1. CT-RATE report CSV
        2. challenge/mock prompts JSON
        3. dataframe
        4. directory containing prompts.json

    Returns:
        [
            {"case_id": "...", "report": "..."},
            ...
        ]
    """

    # Already-loaded dataframe
    if isinstance(input_path_or_df, pd.DataFrame):
        df = input_path_or_df.copy()
        prompts = []

        for i, row in df.iterrows():
            if "VolumeName" in df.columns:
                case_id = clean_case_id(row["VolumeName"])
            else:
                case_id = f"case_{i + 1:03d}"

            prompts.append({
                "case_id": case_id,
                "report": row_to_report_prompt(row),
            })

        return prompts

    input_path = Path(input_path_or_df)

    if input_path.is_dir():
        if (input_path / "prompts.json").exists():
            input_path = input_path / "prompts.json"
        else:
            jsons = sorted(input_path.glob("*.json"))
            csvs = sorted(input_path.glob("*.csv"))

            if jsons:
                input_path = jsons[0]
            elif csvs:
                input_path = csvs[0]
            else:
                raise FileNotFoundError(f"No .json or .csv found in {input_path}")

    if not input_path.exists():
        raise FileNotFoundError(f"Path does not exist: {input_path}")

    # CSV path
    if input_path.suffix.lower() == ".csv":
        df = pd.read_csv(input_path)
        return read_task3_prompts(df)

    # JSON path
    if input_path.suffix.lower() == ".json":
        text = input_path.read_text(errors="ignore").strip()
        obj = json.loads(text)

        if isinstance(obj, list):
            raw_items = obj
        elif isinstance(obj, dict):
            raw_items = obj.get("prompts") or obj.get("inputs") or [obj]
        else:
            raise ValueError("Unsupported JSON prompt format.")

        prompts = []

        for i, item in enumerate(raw_items):
            if isinstance(item, str):
                case_id = f"case_{i + 1:03d}"
                report = item
            else:
                case_id = (
                    item.get("case_id")
                    or item.get("id")
                    or item.get("VolumeName")
                    or item.get("input_image_name")
                    or item.get("output_name")
                    or f"case_{i + 1:03d}"
                )

                report = (
                    item.get("report")
                    or item.get("prompt")
                    or item.get("text")
                    or item.get("input_text")
                    or item.get("radiology_report")
                    or ""
                )

            prompts.append({
                "case_id": clean_case_id(case_id),
                "report": str(report),
            })

        return prompts

    raise ValueError(f"Unsupported input type: {input_path}")

def load_task3_mapper_checkpoint(mapper_checkpoint_dir, device=None):
    mapper_checkpoint_dir = Path(mapper_checkpoint_dir)
    ckpt = torch.load(mapper_checkpoint_dir / "mapper.pt", map_location="cpu")

    tokenizer = AutoTokenizer.from_pretrained(mapper_checkpoint_dir)

    mapper = ReportToCTLatentMapper(
        text_encoder_name=ckpt["text_encoder_name"],
        latent_dim=ckpt["latent_dim"],
        freeze_text_encoder=ckpt["freeze_text_encoder"],
    )

    mapper.load_state_dict(ckpt["model"])

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mapper = mapper.to(device)
    mapper.eval()

    return mapper, tokenizer, ckpt

def predict_task3_report_to_ct(
    input_path,
    autoencoder_checkpoint_path,
    mapper_checkpoint_dir,
    output_dir,
    output_format="npy",
    max_cases=None,
):
    """
    Task 3 inference.

    Reads reports/prompts and writes generated CT volumes.

    output_format:
        "npy"    -> writes case_id.npy
        "nii.gz" -> writes case_id.nii.gz with identity affine
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    autoencoder, ae_ckpt = load_task3_autoencoder_checkpoint(
        autoencoder_checkpoint_path,
        device=device,
    )

    mapper, tokenizer, mapper_ckpt = load_task3_mapper_checkpoint(
        mapper_checkpoint_dir,
        device=device,
    )

    prompts = read_task3_prompts(input_path)

    if max_cases is not None:
        prompts = prompts[:max_cases]

    max_text_length = mapper_ckpt["max_text_length"]

    written_paths = []

    with torch.no_grad():
        for item in prompts:
            case_id = item["case_id"]
            report = item["report"]

            tokenized = tokenizer(
                [report],
                padding=True,
                truncation=True,
                max_length=max_text_length,
                return_tensors="pt",
            )

            input_ids = tokenized["input_ids"].to(device)
            attention_mask = tokenized["attention_mask"].to(device)

            z_pred = mapper(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            x_pred = autoencoder.decode(z_pred)

            # [1, 1, D, H, W] -> [D, H, W]
            volume_norm = x_pred[0, 0].detach().cpu().numpy()
            volume_hu = normalized_to_hu(volume_norm)

            if output_format == "npy":
                out_path = output_dir / f"{case_id}.npy"
                np.save(out_path, volume_hu.astype(np.float32))

            elif output_format == "nii.gz":
                import nibabel as nib

                out_path = output_dir / f"{case_id}.nii.gz"
                img = nib.Nifti1Image(volume_hu.astype(np.float32), affine=np.eye(4))
                nib.save(img, str(out_path))

            else:
                raise ValueError(f"Unsupported output_format: {output_format}")

            written_paths.append(out_path)
            print(f"Wrote generated CT: {out_path}")

    return written_paths
# ---------------------------------------------------------------------
# Small robustness overrides for standalone use.
# ---------------------------------------------------------------------

def load_volume(path):
    """
    Load .npy, .nii, or .nii.gz and return (volume, info).
    Overrides the notebook version, which only loaded NIfTI.
    """
    path = Path(path)
    if path.suffix.lower() == ".npy":
        volume = np.load(path).astype(np.float32)
        info = {"affine": np.eye(4), "header": None, "spacing": None, "shape": volume.shape, "path": path}
        return volume, info

    if path.name.endswith(".nii.gz") or path.suffix.lower() == ".nii":
        img = nib.load(str(path))
        volume = img.get_fdata(dtype=np.float32)
        info = {
            "affine": img.affine,
            "header": img.header.copy(),
            "spacing": img.header.get_zooms(),
            "shape": img.shape,
            "path": path,
        }
        return volume, info

    raise ValueError(f"Unsupported CT file: {path}")


def preprocess_ct_for_torch(path, target_shape):
    """Convert a CT volume path to a normalized tensor shaped [1, D, H, W]."""
    volume, _ = load_volume(path)
    volume = np.squeeze(volume)
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D CT after squeeze, got shape {volume.shape}")

    volume = np.nan_to_num(volume, nan=-1000.0, posinf=1000.0, neginf=-1000.0)
    volume = np.clip(volume, -1000.0, 1000.0)
    volume = (volume + 1000.0) / 1000.0 - 1.0
    x = torch.from_numpy(volume.astype(np.float32))[None, None]
    x = torch.nn.functional.interpolate(x, size=tuple(target_shape), mode="trilinear", align_corners=False)
    return x[0]


def get_hf_token(hf_token=None):
    return hf_token or TOKEN or os.getenv("HF_TOKEN")


def _ctab_get_or_download_volume(self, volume_name):
    subfolder, filename = volume_name_to_hf_parts(str(volume_name))
    local_path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        token=get_hf_token(self.hf_token),
        filename=filename,
        subfolder=subfolder,
        local_dir=str(self.cache_dir),
    )
    return Path(local_path)

try:
    CTAbnormalityDataset.get_or_download_volume = _ctab_get_or_download_volume
except NameError:
    pass


# ============================================================
# === DEADLINE COMPACT TENSOR CACHE ===
# ============================================================
# Each raw NIfTI is downloaded only long enough to preprocess to
# the requested target_shape, then the raw file is deleted.
# The persistent cache is fp16 [1,D,H,W], ~1-2 MB/case at 64x96x96.
# ============================================================

import uuid as _deadline_uuid

_DEADLINE_TENSOR_ROOT = Path(os.getenv("VLM3D_TENSOR_CACHE", "/workspace/vlm3d_tensor_cache"))
_DEADLINE_RAW_STAGE = Path(os.getenv("VLM3D_RAW_STAGE", "/workspace/vlm3d_raw_staging"))


def _deadline_case_stem(volume_name):
    name = Path(str(volume_name)).name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    if name.endswith(".npy"):
        return name[:-4]
    return Path(name).stem


def _deadline_tensor_path(volume_name, target_shape):
    shape_name = "x".join(str(int(x)) for x in target_shape)
    root = _DEADLINE_TENSOR_ROOT / shape_name
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{_deadline_case_stem(volume_name)}.pt"


def _deadline_load_tensor(path):
    try:
        x = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        x = torch.load(path, map_location="cpu")
    return x.float()


def get_deadline_cached_ct(volume_name, target_shape, hf_token=None):
    target_shape = tuple(int(x) for x in target_shape)
    tensor_path = _deadline_tensor_path(volume_name, target_shape)
    if tensor_path.exists() and tensor_path.stat().st_size > 1000:
        return _deadline_load_tensor(tensor_path)

    subfolder, filename = volume_name_to_hf_parts(str(volume_name))
    _DEADLINE_RAW_STAGE.mkdir(parents=True, exist_ok=True)
    token = hf_token or globals().get("TOKEN") or os.getenv("HF_TOKEN")

    raw_path = Path(hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        token=token,
        filename=filename,
        subfolder=subfolder,
        local_dir=str(_DEADLINE_RAW_STAGE),
        force_download=False,
    ))

    try:
        x = preprocess_ct_for_torch(raw_path, target_shape=target_shape)
        tmp = tensor_path.parent / f".{tensor_path.stem}.{os.getpid()}.{_deadline_uuid.uuid4().hex}.tmp"
        torch.save(x.detach().cpu().half().contiguous(), tmp)
        os.replace(tmp, tensor_path)
    finally:
        try:
            raw_path.unlink()
        except FileNotFoundError:
            pass

    return x.float()


def _deadline_ctab_getitem(self, idx):
    row = self.df.iloc[idx]
    volume_name = row["VolumeName"]
    x = get_deadline_cached_ct(volume_name, self.target_shape, self.hf_token)
    y = pd.to_numeric(row[ABNORMALITY_LABELS], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    y = torch.tensor(y, dtype=torch.float32)
    return x, y, volume_name


def _deadline_report_section_getitem(self, idx):
    row = self.df.iloc[idx]
    volume_name = row["VolumeName"]
    x = get_deadline_cached_ct(volume_name, self.target_shape, self.hf_token)
    target_text = row.get(self.target_col, "")
    target_text = "" if target_text is None else str(target_text)
    if target_text.strip() == "" or target_text.lower() == "nan":
        raise ValueError(f"No report section available for {volume_name}")
    tokenized = self.tokenizer(
        target_text,
        max_length=self.max_target_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    labels = tokenized["input_ids"][0].clone()
    labels[labels == self.tokenizer.pad_token_id] = -100
    return x, labels, volume_name


def _deadline_report_to_volume_getitem(self, idx):
    row = self.df.iloc[idx]
    volume_name = row["VolumeName"]
    x = get_deadline_cached_ct(volume_name, self.target_shape, self.hf_token)
    report_text = row.get(self.report_col, "")
    if report_text is None or str(report_text).lower() == "nan":
        report_text = ""
    return x, str(report_text), volume_name


CTAbnormalityDataset.__getitem__ = _deadline_ctab_getitem
CTReportSectionDataset.__getitem__ = _deadline_report_section_getitem
CTReportToVolumeDataset.__getitem__ = _deadline_report_to_volume_getitem

# === END DEADLINE COMPACT TENSOR CACHE ===


# VLM3D_ROBUST_CT_PREPROCESS_CROSS_TASK_V1
def preprocess_ct_for_torch(path, target_shape):
    """Promote single-slice CTs to 3-D and sanitize malformed numeric values."""
    import numpy as _np
    import torch as _torch
    from pathlib import Path as _Path
    try:
        volume, _ = load_volume(path)
        volume = _np.asarray(volume, dtype=_np.float32)
    except Exception as exc:
        print(
            f"[ct-input-warning] {_Path(path).name}: load failed ({exc}); "
            "using an air-valued placeholder",
            flush=True,
        )
        volume = _np.full((1, 1, 1), -1000.0, dtype=_np.float32)
    original_shape = tuple(volume.shape)
    if volume.size == 0:
        volume = _np.full((1, 1, 1), -1000.0, dtype=_np.float32)
    else:
        volume = _np.squeeze(volume)
        while volume.ndim < 3:
            volume = _np.expand_dims(volume, axis=0)
        while volume.ndim > 3:
            axis = volume.ndim - 1
            volume = _np.take(volume, volume.shape[axis] // 2, axis=axis)
    if tuple(volume.shape) != original_shape:
        print(
            f"[ct-input-warning] {_Path(path).name}: adapted "
            f"{original_shape} -> {tuple(volume.shape)}",
            flush=True,
        )
    volume = _np.nan_to_num(volume, nan=-1000.0, posinf=1000.0, neginf=-1000.0)
    volume = _np.clip(volume, -1000.0, 1000.0)
    volume = (volume + 1000.0) / 1000.0 - 1.0
    tensor = _torch.from_numpy(_np.ascontiguousarray(volume))[None, None]
    tensor = _torch.nn.functional.interpolate(
        tensor, size=tuple(target_shape), mode="trilinear", align_corners=False
    )
    return tensor[0]


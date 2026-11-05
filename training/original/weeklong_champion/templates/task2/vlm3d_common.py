"""
Shared helpers for VLM3D task scripts.
"""

from pathlib import Path
import json
import os
import shutil
import subprocess

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
    return path


def read_json(path):
    return json.loads(Path(path).read_text())


def strip_ct_extension(x):
    name = Path(str(x)).name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    if name.endswith(".npy"):
        return name[:-4]
    return name


def normalize_case_key(x):
    return strip_ct_extension(str(x))


def save_dataframe(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def is_ct_file(path):
    name = Path(path).name.lower()
    return name.endswith(".npy") or name.endswith(".nii") or name.endswith(".nii.gz")


def list_ct_files(input_dir):
    input_dir = Path(input_dir)
    return sorted([p for p in input_dir.rglob("*") if p.is_file() and is_ct_file(p)])


def _read_manifest(path):
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".json":
        obj = json.loads(path.read_text())
        if isinstance(obj, list):
            items = obj
        elif isinstance(obj, dict):
            items = obj.get("inputs") or obj.get("cases") or obj.get("images") or obj.get("data") or obj.get("items") or obj.get("prompts") or [obj]
        else:
            raise ValueError(f"Unsupported JSON structure: {path}")
        return pd.DataFrame(items if not (items and isinstance(items[0], str)) else {"path": items})
    raise ValueError(f"Unsupported manifest: {path}")


def resolve_ct_paths_from_input(input_source, max_cases=None, cache_dir="./ct_rate_volume_cache", hf_token=None):
    """
    Resolve CT volumes from either a folder, CSV/JSON local-path manifest, or CT-RATE VolumeName CSV/JSON.
    Returns list of (case_id, path).
    """
    from huggingface_hub import hf_hub_download
    from model_code_from_original import volume_name_to_hf_parts, REPO_ID, TOKEN

    src = Path(input_source)
    out = []

    if src.is_dir():
        for p in list_ct_files(src):
            out.append((strip_ct_extension(p.name), p))
        return out[:max_cases] if max_cases is not None else out

    df = _read_manifest(src)
    if max_cases is not None:
        df = df.head(max_cases)

    for col in ["path", "filepath", "file_path", "image_path", "ct_path", "volume_path", "filename", "file", "input_image_name"]:
        if col in df.columns:
            rows = []
            for _, row in df.iterrows():
                raw = str(row[col])
                rp = Path(raw)
                candidates = [rp] if rp.is_absolute() else [src.parent / rp, src.parent / "images" / rp.name, src.parent.parent / "images" / rp.name]
                found = next((p for p in candidates if p.exists() and is_ct_file(p)), None)
                if found is not None:
                    rows.append((strip_ct_extension(found.name), found))
            if rows:
                return rows

    vol_col = None
    for c in df.columns:
        if c.lower() in ["volumename", "volume_name", "volume"]:
            vol_col = c
            break
    if vol_col is None:
        raise ValueError(f"Could not resolve CTs from {input_source}. Columns={list(df.columns)}")

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    token = hf_token or TOKEN or os.getenv("HF_TOKEN")

    for volume_name in df[vol_col].astype(str):
        p = Path(volume_name)
        if p.exists() and is_ct_file(p):
            out.append((strip_ct_extension(p.name), p))
            continue
        subfolder, filename = volume_name_to_hf_parts(volume_name)
        local_path = hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            token=token,
            filename=filename,
            subfolder=subfolder,
            local_dir=str(cache_dir),
            force_download=False,
        )
        out.append((strip_ct_extension(volume_name), Path(local_path)))
    return out


def save_ct_slices_png(volume_or_path, output_path, title=None, vmin=-1000, vmax=500):
    from model_code_from_original import load_ct_volume
    if isinstance(volume_or_path, (str, Path)):
        vol = load_ct_volume(volume_or_path)
    else:
        vol = np.squeeze(np.asarray(volume_or_path, dtype=np.float32))
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {vol.shape}")
    z, y, x = vol.shape[0] // 2, vol.shape[1] // 2, vol.shape[2] // 2
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].imshow(vol[z], cmap="gray", vmin=vmin, vmax=vmax); axes[0].set_title("Axial")
    axes[1].imshow(vol[:, y, :], cmap="gray", vmin=vmin, vmax=vmax); axes[1].set_title("Coronal")
    axes[2].imshow(vol[:, :, x], cmap="gray", vmin=vmin, vmax=vmax); axes[2].set_title("Sagittal")
    for ax in axes:
        ax.axis("off")
    if title:
        fig.suptitle(str(title))
    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_bar_plot(series, output_path, title, xlabel, xlim=None):
    series = pd.Series(series).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(series))))
    ax.barh(series.index.astype(str), series.values)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def extract_classification_predictions(obj):
    if isinstance(obj, dict) and "predictions" in obj:
        return obj["predictions"]
    if isinstance(obj, list):
        for item in obj:
            if not isinstance(item, dict):
                continue
            if "predictions" in item:
                return item["predictions"]
            for out in item.get("outputs", []):
                value = out.get("value", {})
                if "predictions" in value:
                    return value["predictions"]
    raise ValueError("Could not find predictions in JSON.")


def classification_json_to_df(predictions_json):
    rows = []
    for item in extract_classification_predictions(read_json(predictions_json)):
        case_id = item.get("input_image_name") or item.get("case_id") or item.get("VolumeName") or "unknown_case"
        probs = item.get("probabilities", item.get("scores", {}))
        row = {"input_image_name": str(case_id)}
        row.update(probs)
        rows.append(row)
    return pd.DataFrame(rows)


def classification_metrics(pred_df, labels_csv, threshold=0.5):
    from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
    from model_code_from_original import ABNORMALITY_LABELS
    labels_df = pd.read_csv(labels_csv)
    pred_df = pred_df.copy()
    labels_df = labels_df.copy()
    pred_df["case_key"] = pred_df["input_image_name"].map(normalize_case_key)
    labels_df["case_key"] = labels_df["VolumeName"].map(normalize_case_key)
    merged = pred_df.merge(labels_df, on="case_key", how="inner", suffixes=("_pred", "_true"))

    rows = []
    for label in ABNORMALITY_LABELS:
        pred_col = (
            f"{label}_pred" if f"{label}_pred" in merged.columns
            else label if label in merged.columns
            else f"prob_{label.lower().replace(' ', '_')}"
        )
        true_col = f"{label}_true" if f"{label}_true" in merged.columns else label
        if pred_col not in merged.columns or true_col not in merged.columns:
            continue
        y_true = merged[true_col].fillna(0).astype(int).values
        y_score = merged[pred_col].fillna(0).astype(float).values
        y_pred = (y_score >= threshold).astype(int)
        try:
            auroc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else np.nan
        except Exception:
            auroc = np.nan
        rows.append({
            "label": label,
            "auroc": auroc,
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "accuracy": accuracy_score(y_true, y_pred),
            "support": int(y_true.sum()),
        })
    out = pd.DataFrame(rows)
    if len(out):
        macro = {"label": "macro_average"}
        for c in ["auroc", "f1", "precision", "recall", "accuracy"]:
            macro[c] = float(out[c].mean(skipna=True))
        macro["support"] = int(out["support"].sum())
        out = pd.concat([out, pd.DataFrame([macro])], ignore_index=True)
    return out, merged


def extract_generated_reports(obj):
    if isinstance(obj, dict):
        if "generated_reports" in obj:
            return obj["generated_reports"]
        if "predictions" in obj:
            return obj["predictions"]
    if isinstance(obj, list):
        for item in obj:
            if not isinstance(item, dict):
                continue
            if "generated_reports" in item:
                return item["generated_reports"]
            if "predictions" in item:
                return item["predictions"]
            for out in item.get("outputs", []):
                value = out.get("value", {})
                if "generated_reports" in value:
                    return value["generated_reports"]
                if "predictions" in value:
                    return value["predictions"]
    raise ValueError("Could not find generated reports in JSON.")


def generated_reports_json_to_df(predictions_json):
    return pd.DataFrame(extract_generated_reports(read_json(predictions_json)))


def build_reference_report_df(reports_csv):
    df = pd.read_csv(reports_csv).copy()
    findings = df["Findings_EN"].fillna("").astype(str) if "Findings_EN" in df.columns else ""
    impressions = df["Impressions_EN"].fillna("").astype(str) if "Impressions_EN" in df.columns else ""
    df["reference_report"] = "FINDINGS: " + findings.str.strip() + "\n\nIMPRESSION: " + impressions.str.strip()
    df["case_key"] = df["VolumeName"].map(normalize_case_key)
    return df


def _lcs_len(a, b):
    prev = [0] * (len(b) + 1)
    for x in a:
        curr = [0]
        for j, y in enumerate(b, start=1):
            curr.append(prev[j - 1] + 1 if x == y else max(prev[j], curr[-1]))
        prev = curr
    return prev[-1]


def report_text_metrics(pred_df, reference_reports_csv):
    ref = build_reference_report_df(reference_reports_csv)
    pred_df = pred_df.copy()
    pred_df["case_key"] = pred_df["input_image_name"].map(normalize_case_key)
    merged = pred_df.merge(ref[["case_key", "reference_report"]], on="case_key", how="inner")
    rows = []
    for _, row in merged.iterrows():
        pred = str(row.get("report", ""))
        ref_text = str(row.get("reference_report", ""))
        ptok, rtok = pred.lower().split(), ref_text.lower().split()
        pset, rset = set(ptok), set(rtok)
        overlap = len(pset & rset)
        prec = overlap / max(len(pset), 1)
        rec = overlap / max(len(rset), 1)
        tok_f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        lcs = _lcs_len(ptok, rtok)
        rouge_p = lcs / max(len(ptok), 1)
        rouge_r = lcs / max(len(rtok), 1)
        rouge_l_f1 = 2 * rouge_p * rouge_r / max(rouge_p + rouge_r, 1e-8)
        rows.append({
            "input_image_name": row["input_image_name"],
            "bleu1_proxy": prec,
            "meteor": np.nan,
            "rouge_l_f1": rouge_l_f1,
            "token_precision": prec,
            "token_recall": rec,
            "token_f1": tok_f1,
            "pred_len": len(ptok),
            "ref_len": len(rtok),
        })
    return pd.DataFrame(rows), merged


def ct_generation_proxy_metrics(generated_dir):
    from model_code_from_original import list_ct_files, summarize_volume
    return pd.DataFrame([summarize_volume(p) for p in list_ct_files(generated_dir)])


def save_official_metric_note(output_dir, task_name, official_metrics, local_metrics):
    obj = {
        "task": task_name,
        "official_metrics": official_metrics,
        "local_metrics_computed_here": local_metrics,
        "note": "Hidden-test official metrics requiring private labels/references or official feature extractors are not computable inside the submission container. This script reports local/proxy metrics when local references are provided.",
    }
    return write_json(obj, Path(output_dir) / "metrics_manifest.json")


def write_example_table(rows, output_dir, filename="examples.csv"):
    return save_dataframe(pd.DataFrame(rows), Path(output_dir) / filename)


def make_requirements_text():
    return """numpy
pandas
matplotlib
scikit-learn
nibabel
torch
monai
transformers
huggingface_hub
sentencepiece
accelerate
"""


def make_dockerfile_text(entrypoint, checkpoint_cmd_args):
    return f'''FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY model_code.py /app/model_code.py
COPY vlm3d_common.py /app/vlm3d_common.py
COPY {entrypoint} /app/{entrypoint}
COPY checkpoints /app/checkpoints

ENV PYTHONUNBUFFERED=1
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

CMD ["python", "/app/{entrypoint}", "--mode", "predict", "--input", "/input", "--output", "/output"{checkpoint_cmd_args}]
'''


def build_docker_context(task_name, entrypoint, checkpoint_sources, checkpoint_cmd_args, output_root="docker_contexts"):
    context_dir = Path(output_root) / task_name
    if context_dir.exists():
        shutil.rmtree(context_dir)
    context_dir.mkdir(parents=True, exist_ok=True)
    for filename in ["model_code.py", "vlm3d_common.py", entrypoint]:
        shutil.copy2(filename, context_dir / filename)
    (context_dir / "requirements.txt").write_text(make_requirements_text())
    (context_dir / "Dockerfile").write_text(make_dockerfile_text(entrypoint, checkpoint_cmd_args))
    ckpt_root = context_dir / "checkpoints"
    ckpt_root.mkdir(parents=True, exist_ok=True)
    for src, dst in checkpoint_sources.items():
        src = Path(src)
        dst = ckpt_root / dst
        if src.is_dir():
            shutil.copytree(src, dst)
        elif src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        else:
            raise FileNotFoundError(f"Missing checkpoint source: {src}")
    return context_dir


def docker_build(context_dir, image_tag):
    subprocess.run(["docker", "build", "-t", image_tag, str(context_dir)], check=True)


def docker_save_gzip(image_tag, tar_gz_path):
    tar_gz_path = Path(tar_gz_path)
    tar_gz_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(f"docker save {image_tag} | gzip > {tar_gz_path}", shell=True, check=True)
    return tar_gz_path

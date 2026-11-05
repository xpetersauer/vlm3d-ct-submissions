#!/usr/bin/env python3
"""
Task 1 from your original notebook: CT -> abnormality probabilities.
Official VLM3D/Forithmus track: ct-abnormality-classification / abnclass.

This file starts from the same task code in trial_run_vlm3d_chat_07282026.ipynb,
but wraps it as a runnable Python script with train, predict, local_eval, and
Forithmus submission modes.

Forithmus contract:
  input:  /input/*.nii.gz
  output: /output/results.json
  schema: {"predictions": [{"input_image_name": ..., "probabilities": {18 labels...}}]}
"""

from pathlib import Path
import argparse
import json
import os
import shutil
import zipfile

import pandas as pd

import model_code_from_original as mc
from vlm3d_common import (
    ensure_dir,
    classification_json_to_df,
    classification_metrics,
    save_dataframe,
    save_bar_plot,
    resolve_ct_paths_from_input,
    save_ct_slices_png,
    write_example_table,
    save_official_metric_note,
)

OFFICIAL_TASK = "ct-abnormality-classification"
OFFICIAL_OUTPUT_FILE = "results.json"


def officialize_abnclass_results(results_json):
    """Rewrite the notebook-style JSON into the current Forithmus-style wrapper."""
    path = Path(results_json)
    obj = json.loads(path.read_text())
    predictions = obj.get("predictions", obj if isinstance(obj, list) else None)
    if predictions is None:
        raise ValueError(f"Could not find predictions in {path}")

    fixed = []
    for item in predictions:
        case_id = item.get("input_image_name") or item.get("case_id") or item.get("VolumeName")
        probs = item.get("probabilities") or item.get("scores") or {}
        # Make sure every official label exists. Missing labels fail Forithmus schema/eval.
        full_probs = {label: float(probs.get(label, 0.0)) for label in mc.ABNORMALITY_LABELS}
        fixed.append({"input_image_name": str(mc.strip_ct_extension(case_id)), "probabilities": full_probs})

    official = {
        "name": "Generated probabilities",
        "type": "Abnormality Classification",
        "version": {"major": 1, "minor": 0},
        "predictions": fixed,
    }
    path.write_text(json.dumps(official, indent=2))
    return path


def train(args):
    train_labels = pd.read_csv(args.train_labels_csv)

    model_params = dict(
        feature_size=args.feature_size,
        in_channels=1,
        out_channels=len([c for c in train_labels.columns if c != "VolumeName"]),
        spatial_dims=3,
        depths=tuple(args.depths),
        num_heads=tuple(args.num_heads),
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.1,
        use_checkpoint=args.use_checkpoint,
    )

    model, checkpoint_path = mc.train_task1(
        model_params=model_params,
        num_labels=len([c for c in train_labels.columns if c != "VolumeName"]),
        df=train_labels,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        num_workers=args.num_workers,
        max_rows=args.max_rows,
        checkpoint_path=args.checkpoint,
        cache_dir=args.cache_dir,
        target_shape=tuple(args.target_shape),
    )

    print(f"Saved Task 1 checkpoint: {checkpoint_path}")
    return checkpoint_path


def predict(args):
    out = mc.predict_task1(
        input_dir=args.input,
        checkpoint_path=args.checkpoint,
        output_dir=args.output,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_cases=args.max_cases,
        target_shape=tuple(args.target_shape),
        include_extension=False,
        cache_dir=args.cache_dir,
        hf_token=args.hf_token,
        output_filename=OFFICIAL_OUTPUT_FILE,
    )
    officialize_abnclass_results(out["json_path"])
    print(f"Wrote Forithmus-compatible output: {out['json_path']}")
    return out


def local_eval(args):
    output_dir = ensure_dir(args.output)
    out = predict(args)
    pred_df = classification_json_to_df(out["json_path"])
    save_dataframe(pred_df, output_dir / "predictions_file_level_readable.csv")

    save_official_metric_note(
        output_dir,
        task_name="ct_abnormality_classification",
        official_metrics=["CRG", "macro AUROC", "macro F1"],
        local_metrics=["AUROC", "F1", "Precision", "Recall", "Accuracy"] if args.labels_csv else ["schema check", "example visualizations"],
    )

    if args.labels_csv:
        metrics_df, _ = classification_metrics(pred_df, args.labels_csv, threshold=args.threshold)
        save_dataframe(metrics_df, output_dir / "metrics_classification.csv")
        if metrics_df is not None and (not metrics_df.empty) and ("label" in metrics_df.columns):
            per_label = metrics_df[metrics_df["label"] != "macro_average"].set_index("label")
            if len(per_label):
                if "auroc" in per_label.columns:
                    save_bar_plot(per_label["auroc"].dropna(), output_dir / "figures" / "auroc_by_label.png", "Task 1 AUROC by abnormality", "AUROC", xlim=(0, 1))
                if "f1" in per_label.columns:
                    save_bar_plot(per_label["f1"].dropna(), output_dir / "figures" / "f1_by_label.png", "Task 1 F1 by abnormality", "F1", xlim=(0, 1))
            print(metrics_df.tail(1))
        else:
            print("Task 1 output was written, but local metrics were skipped because this one-case smoke test did not produce a non-empty metrics table.")

    # Save examples for quick visual inspection.
    example_rows = []
    try:
        ct_paths = dict(resolve_ct_paths_from_input(args.input, max_cases=args.n_show, cache_dir=args.cache_dir, hf_token=args.hf_token))
    except Exception as e:
        print(f"Could not resolve input CTs for example images: {e}")
        ct_paths = {}

    label_cols = [c for c in pred_df.columns if c != "input_image_name"]
    for i in range(min(args.n_show, len(pred_df))):
        row = pred_df.iloc[i]
        case_id = str(row["input_image_name"])
        probs = row[label_cols].astype(float).sort_values(ascending=False).head(args.top_k)
        bar_path = save_bar_plot(probs, output_dir / "examples" / f"{case_id}_top_probs.png", f"Top {args.top_k}: {case_id}", "Probability", xlim=(0, 1))
        ct_png = None
        ct_path = ct_paths.get(case_id) or ct_paths.get(Path(case_id).stem)
        if ct_path is not None:
            ct_png = save_ct_slices_png(ct_path, output_dir / "examples" / f"{case_id}_input_ct.png", title=f"Input CT: {case_id}")
        example_rows.append({"case_id": case_id, "input_ct_png": str(ct_png) if ct_png else None, "top_probs_png": str(bar_path)})
    write_example_table(example_rows, output_dir)
    return out


def make_weights(args):
    weights_dir = ensure_dir(args.weights_dir)
    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)
    shutil.copy2(ckpt, weights_dir / "task1_swinunetr.pt")
    zip_path = Path(args.weights_zip)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.write(weights_dir / "task1_swinunetr.pt", "task1_swinunetr.pt")
    print(f"Wrote {zip_path}; contents are at ZIP root.")
    return zip_path


def forithmus_process(args):
    args.input = os.environ.get("FORITHMUS_INPUT", "/input")
    args.output = os.environ.get("FORITHMUS_OUTPUT", "/output")
    args.checkpoint = os.environ.get("TASK1_CHECKPOINT", "/weights/task1_swinunetr.pt")
    args.max_cases = None
    return predict(args)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "predict", "local_eval", "make_weights", "forithmus"], default="predict")
    p.add_argument("--input", default="/input")
    p.add_argument("--output", default="/output")
    p.add_argument("--checkpoint", default="/weights/task1_swinunetr.pt")
    p.add_argument("--train-labels-csv", default=mc.CT_RATE_CSV_FILES["train"]["labels"])
    p.add_argument("--labels-csv", default=None)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-cases", type=int, default=None)
    p.add_argument("--max-rows", type=int, default=10)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--target-shape", type=int, nargs=3, default=[64, 128, 128])
    p.add_argument("--cache-dir", default="./ct_rate_volume_cache")
    p.add_argument("--hf-token", default=os.getenv("HF_TOKEN"))
    p.add_argument("--n-show", type=int, default=5)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--feature-size", type=int, default=24)
    p.add_argument("--depths", type=int, nargs=4, default=[2, 2, 2, 2])
    p.add_argument("--num-heads", type=int, nargs=4, default=[3, 6, 12, 24])
    p.add_argument("--use-checkpoint", action="store_true", default=False)
    p.add_argument("--weights-dir", default="weights_task1")
    p.add_argument("--weights-zip", default="weights_task1.zip")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "train":
        train(args)
    elif args.mode == "predict":
        predict(args)
    elif args.mode == "local_eval":
        local_eval(args)
    elif args.mode == "make_weights":
        make_weights(args)
    elif args.mode == "forithmus":
        forithmus_process(args)

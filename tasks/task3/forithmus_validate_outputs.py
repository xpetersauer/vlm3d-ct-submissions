#!/usr/bin/env python3
"""
Local schema validators for the three VLM3D/Forithmus outputs.
These are not the official evaluator, but mirror the public output contracts so
format errors are caught before `forithmus test` / paid submission.
"""

from pathlib import Path
import argparse
import json

ABNORMALITY_LABELS = [
    "Medical material", "Arterial wall calcification", "Cardiomegaly",
    "Pericardial effusion", "Coronary artery wall calcification", "Hiatal hernia",
    "Lymphadenopathy", "Emphysema", "Atelectasis", "Lung nodule",
    "Lung opacity", "Pulmonary fibrotic sequela", "Pleural effusion",
    "Mosaic attenuation pattern", "Peribronchial thickening", "Consolidation",
    "Bronchiectasis", "Interlobular septal thickening",
]


def _read_json(path):
    return json.loads(Path(path).read_text())


def validate_abnclass(output_dir):
    output_dir = Path(output_dir)
    path = output_dir / "results.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    obj = _read_json(path)
    if not isinstance(obj, dict):
        raise TypeError("Task 1 results.json must be a dict")
    preds = obj.get("predictions")
    if not isinstance(preds, list) or len(preds) == 0:
        raise ValueError("Task 1 needs non-empty predictions list")
    for i, item in enumerate(preds):
        if "input_image_name" not in item:
            raise ValueError(f"Task 1 prediction {i} missing input_image_name")
        probs = item.get("probabilities")
        if not isinstance(probs, dict):
            raise ValueError(f"Task 1 prediction {i} missing probabilities dict")
        missing = [l for l in ABNORMALITY_LABELS if l not in probs]
        if missing:
            raise ValueError(f"Task 1 prediction {i} missing labels: {missing}")
        for label in ABNORMALITY_LABELS:
            v = float(probs[label])
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"Task 1 {item['input_image_name']} {label}={v}, expected [0,1]")
    print(f"✓ abnclass schema OK: {len(preds)} predictions in {path}")


def _extract_reportgen(obj):
    if isinstance(obj, list):
        for item in obj:
            if not isinstance(item, dict):
                continue
            for out in item.get("outputs", []):
                value = out.get("value", {})
                if "generated_reports" in value:
                    return value["generated_reports"]
    if isinstance(obj, dict) and "generated_reports" in obj:
        return obj["generated_reports"]
    raise ValueError("Could not find generated_reports in reportgen predictions.json")


def validate_reportgen(output_dir):
    output_dir = Path(output_dir)
    path = output_dir / "predictions.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    reports = _extract_reportgen(_read_json(path))
    if not isinstance(reports, list) or len(reports) == 0:
        raise ValueError("Task 2 needs non-empty generated_reports list")
    for i, item in enumerate(reports):
        if "input_image_name" not in item:
            raise ValueError(f"Task 2 report {i} missing input_image_name")
        if "report" not in item or not isinstance(item["report"], str):
            raise ValueError(f"Task 2 report {i} missing report string")
    print(f"✓ reportgen schema OK: {len(reports)} reports in {path}")


def validate_ctgen(output_dir):
    output_dir = Path(output_dir)
    paths = sorted(output_dir.rglob("*.nii.gz"))
    if not paths:
        raise FileNotFoundError(f"Task 3 expected loose .nii.gz files in {output_dir}")
    # Avoid importing nibabel as a hard dependency for schema-only validation.
    for p in paths:
        if p.name.endswith(".nii.gz") is False:
            raise ValueError(f"Unexpected ctgen output name: {p.name}")
    print(f"✓ ctgen schema OK: {len(paths)} .nii.gz volumes in {output_dir}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("task", choices=["task1", "abnclass", "task2", "reportgen", "task3", "ctgen"])
    p.add_argument("output_dir")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.task in ["task1", "abnclass"]:
        validate_abnclass(args.output_dir)
    elif args.task in ["task2", "reportgen"]:
        validate_reportgen(args.output_dir)
    else:
        validate_ctgen(args.output_dir)

#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math
from pathlib import Path

LABELS = [
    "Medical material", "Arterial wall calcification", "Cardiomegaly",
    "Pericardial effusion", "Coronary artery wall calcification",
    "Hiatal hernia", "Lymphadenopathy", "Emphysema", "Atelectasis",
    "Lung nodule", "Lung opacity", "Pulmonary fibrotic sequela",
    "Pleural effusion", "Mosaic attenuation pattern",
    "Peribronchial thickening", "Consolidation", "Bronchiectasis",
    "Interlobular septal thickening",
]
VOLUME_SUFFIXES = (".nii.gz", ".nii", ".mha", ".npy")

def case_id(value):
    name = Path(str(value)).name
    lower = name.lower()
    for suffix in VOLUME_SUFFIXES:
        if lower.endswith(suffix):
            return name[:-len(suffix)]
    return name

def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Invalid JSON at {path}: {exc}") from exc

def atomic_json(path, payload):
    tmp = path.with_name("." + path.name + ".contract.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)

def input_volume_ids(input_dir):
    return [
        case_id(path.name) for path in sorted(input_dir.rglob("*"))
        if path.is_file() and path.name.lower().endswith((".nii.gz", ".nii", ".mha"))
    ]

def check_ids(actual, expected, task):
    if len(actual) != len(set(actual)):
        raise RuntimeError(f"{task}: duplicate output identifiers")
    if expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        if missing or extra or len(actual) != len(expected):
            raise RuntimeError(
                f"{task}: expected={len(expected)} actual={len(actual)} "
                f"missing={missing[:10]} extra={extra[:10]}"
            )

def unwrap(raw):
    if isinstance(raw, list) and len(raw) == 1:
        try:
            return raw[0]["outputs"][0]["value"]
        except (KeyError, IndexError, TypeError):
            pass
    return raw

def task1(input_dir, output_dir):
    path = output_dir / "results.json"
    if not path.is_file():
        raise RuntimeError(f"Task 1 missing {path}")
    inner = unwrap(read_json(path))
    predictions = inner.get("predictions") if isinstance(inner, dict) else None
    if not isinstance(predictions, list):
        raise RuntimeError("Task 1 results.json has no predictions list")
    normalized = []
    for index, item in enumerate(predictions):
        name = case_id(item.get("input_image_name", ""))
        scores = item.get("probabilities")
        if not name or not isinstance(scores, dict):
            raise RuntimeError(f"Task 1 malformed prediction index {index}")
        missing = [label for label in LABELS if label not in scores]
        if missing:
            raise RuntimeError(f"Task 1 {name} missing labels: {missing}")
        clean = {}
        for label in LABELS:
            value = float(scores[label])
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise RuntimeError(f"Task 1 invalid probability {name}/{label}={value}")
            clean[label] = value
        normalized.append({"input_image_name": name, "probabilities": clean})
    check_ids([x["input_image_name"] for x in normalized], input_volume_ids(input_dir), "Task 1")
    value = {
        "name": "Generated probabilities", "type": "Abnormality Classification",
        "version": "1.0", "predictions": normalized,
    }
    atomic_json(path, [{"outputs": [{"value": value}]}])
    print(f"[contract] TASK1 PASS: {len(normalized)} predictions", flush=True)

def task2(input_dir, output_dir):
    path = output_dir / "predictions.json"
    if not path.is_file():
        raise RuntimeError(f"Task 2 missing {path}")
    inner = unwrap(read_json(path))
    if isinstance(inner, list):
        reports = inner
    elif isinstance(inner, dict):
        reports = inner.get("generated_reports")
    else:
        reports = None
    if not isinstance(reports, list):
        raise RuntimeError("Task 2 predictions.json has no generated_reports list")
    normalized = []
    for index, item in enumerate(reports):
        name = case_id(item.get("input_image_name", ""))
        report = item.get("report")
        if not name or not isinstance(report, str) or not report.strip():
            raise RuntimeError(f"Task 2 malformed/empty report index {index}")
        normalized.append({"input_image_name": name, "report": report.strip()})
    check_ids([x["input_image_name"] for x in normalized], input_volume_ids(input_dir), "Task 2")
    value = {
        "name": "Generated reports", "type": "Report generation",
        "version": "1.0", "generated_reports": normalized,
    }
    atomic_json(path, [{"outputs": [{"value": value}]}])
    print(f"[contract] TASK2 PASS: {len(normalized)} reports", flush=True)

def task3(input_dir, output_dir):
    import nibabel as nib
    import numpy as np
    prompt_path = input_dir / "prompts.json"
    if not prompt_path.is_file():
        candidates = sorted(input_dir.glob("*.json"))
        if not candidates:
            raise RuntimeError("Task 3 has no prompt JSON")
        prompt_path = candidates[0]
    raw = read_json(prompt_path)
    if isinstance(raw, dict):
        raw = raw.get("prompts") or raw.get("inputs")
    if not isinstance(raw, list):
        raise RuntimeError("Task 3 prompts must be a list")
    expected = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise RuntimeError(f"Task 3 malformed prompt index {index}")
        value = item.get("input_image_name") or item.get("case_id") or item.get("id") or item.get("VolumeName")
        if value is None:
            raise RuntimeError(f"Task 3 prompt {index} has no case identifier")
        expected.append(case_id(value))
    paths = sorted(output_dir.glob("*.nii.gz"))
    actual = [case_id(path.name) for path in paths]
    check_ids(actual, expected, "Task 3")
    for path in paths:
        if path.stat().st_size < 1024:
            raise RuntimeError(f"Task 3 suspiciously small NIfTI: {path}")
        try:
            image = nib.load(str(path))
            if len(image.shape) != 3 or any(int(x) <= 0 for x in image.shape):
                raise ValueError(f"invalid shape {image.shape}")
            if not np.isfinite(image.affine).all():
                raise ValueError("non-finite affine")
            zooms = image.header.get_zooms()[:3]
            if len(zooms) != 3 or not all(math.isfinite(float(x)) and float(x) > 0 for x in zooms):
                raise ValueError(f"invalid spacing {zooms}")
            if not np.isfinite(np.asanyarray(image.dataobj)).all():
                raise ValueError("non-finite voxels")
        except Exception as exc:
            raise RuntimeError(f"Task 3 invalid NIfTI {path}: {exc}") from exc
    print(f"[contract] TASK3 PASS: {len(paths)} exact, finite 3-D NIfTIs", flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("task1", "task2", "task3"), required=True)
    parser.add_argument("--input", type=Path, default=Path("/input"))
    parser.add_argument("--output", type=Path, default=Path("/output"))
    args = parser.parse_args()
    {"task1": task1, "task2": task2, "task3": task3}[args.task](args.input, args.output)

if __name__ == "__main__":
    main()

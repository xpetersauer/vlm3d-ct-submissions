#!/usr/bin/env python3
"""Join Task-1 prediction probabilities to a held-out label CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


SUFFIXES = (".nii.gz", ".nii", ".mha", ".npy", ".npz")


def case_id(value) -> str:
    name = Path(str(value)).name
    for suffix in SUFFIXES:
        if name.lower().endswith(suffix):
            return name[:-len(suffix)]
    return Path(name).stem


def probability_rows(node):
    rows = []
    if isinstance(node, dict):
        probabilities = node.get("probabilities")
        identifier = node.get("input_image_name") or node.get("case_id") or node.get("id")
        if isinstance(probabilities, dict) and identifier is not None:
            rows.append((case_id(identifier), probabilities))
        for value in node.values():
            rows.extend(probability_rows(value))
    elif isinstance(node, list):
        for value in node:
            rows.extend(probability_rows(value))
    return rows


def normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-rows", type=int, default=20)
    args = parser.parse_args()
    import pandas as pd
    predictions = probability_rows(json.loads(args.predictions.read_text()))
    if not predictions:
        raise SystemExit("No Task-1 probability dictionaries found")
    label_frame = pd.read_csv(args.labels)
    id_column = next((name for name in ("VolumeName", "input_image_name", "case_id") if name in label_frame), None)
    if id_column is None:
        raise SystemExit(f"No case-id column in {args.labels}")
    label_frame["_case_id"] = label_frame[id_column].map(case_id)
    label_frame = label_frame.set_index("_case_id")
    prediction_labels = list(predictions[0][1])
    columns_by_normalized = {normalized(column): column for column in label_frame.columns}
    mapping = {}
    for label in prediction_labels:
        key = normalized(label)
        if key in columns_by_normalized:
            mapping[label] = columns_by_normalized[key]
    if len(mapping) != len(prediction_labels):
        missing = [label for label in prediction_labels if label not in mapping]
        raise SystemExit(f"Label CSV does not match prediction labels: {missing}")
    rows = []
    for identifier, probability in predictions:
        if identifier not in label_frame.index:
            continue
        truth = label_frame.loc[identifier]
        row = {"VolumeName": identifier}
        for label, column in mapping.items():
            # Preserve the exact emitted label spelling so deployment can
            # match the probability dictionary without an ambiguous alias.
            row[f"true_{label}"] = float(truth[column])
            row[f"prob_{label}"] = float(probability[label])
        rows.append(row)
    if len(rows) < args.minimum_rows:
        raise SystemExit(f"Only {len(rows)} Task-1 cases joined; refusing unstable calibration")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"[task1-table] wrote {len(rows)} cases x {len(mapping)} labels to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

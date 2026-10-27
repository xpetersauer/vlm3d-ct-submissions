#!/usr/bin/env python3
"""Apply held-out Task-1 calibration to every probability dictionary."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def logit(value: float) -> float:
    value = min(max(float(value), 1e-7), 1.0 - 1e-7)
    return math.log(value / (1.0 - value))


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def apply(node, parameters: dict[str, dict]) -> int:
    changed = 0
    if isinstance(node, dict):
        probabilities = node.get("probabilities")
        if isinstance(probabilities, dict):
            for label, config in parameters.items():
                if label not in probabilities:
                    continue
                raw = float(probabilities[label])
                temperature = max(float(config["temperature"]), 1e-4)
                bias = float(config["bias"])
                threshold = min(max(float(config["threshold"]), 1e-6), 1.0 - 1e-6)
                calibrated_logit = logit(raw) / temperature + bias - logit(threshold)
                probabilities[label] = round(sigmoid(calibrated_logit), 7)
                changed += 1
        for value in node.values():
            changed += apply(value, parameters)
    elif isinstance(node, list):
        for value in node:
            changed += apply(value, parameters)
    return changed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=Path("/output/results.json"))
    parser.add_argument("--calibration", type=Path, default=Path("/weights/task1_metric_calibration.json"))
    args = parser.parse_args()
    if not args.calibration.is_file():
        print(f"[task1-calibration] no calibration file at {args.calibration}; output unchanged", flush=True)
        return 0
    payload = json.loads(args.predictions.read_text())
    config = json.loads(args.calibration.read_text())
    changed = apply(payload, config["parameters"])
    if changed == 0:
        raise RuntimeError("Task-1 calibration found no probability values; refusing a silent no-op")
    temporary = args.predictions.with_name(f".{args.predictions.name}.calibrated.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(args.predictions)
    print(f"[task1-calibration] transformed {changed} label probabilities", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Package only the Task-3 checkpoint selected by official FVD_CTNet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-hours", type=float, default=11.5)
    args = parser.parse_args()
    payload = json.loads(args.metrics.read_text())
    rows = payload.get("candidates", payload if isinstance(payload, list) else [])
    by_name = {str(row["name"]): row for row in rows}
    args.output.mkdir(parents=True, exist_ok=True)
    recipe = args.output / "task3_fvd_winner.json"
    subprocess.run([
        sys.executable, str(Path(__file__).with_name("metric_alignment.py")), "task3-select",
        "--metrics", str(args.metrics), "--output", str(recipe), "--max-hours", str(args.max_hours),
    ], check=True)
    selected = json.loads(recipe.read_text())["winner"]
    row = by_name[selected]
    weights = Path(row["weights_zip"])
    source = Path(row["source_dir"])
    if not weights.is_file() or not source.is_dir():
        raise SystemExit(f"Winner artifacts missing: weights={weights} source={source}")
    shutil.copy2(weights, args.output / "task3_weights_fvd_selected.zip")
    shutil.copytree(source, args.output / "task3_source_baseline", dirs_exist_ok=True)
    print(json.dumps({
        "status": "PASS",
        "winner": selected,
        "weights": str(args.output / "task3_weights_fvd_selected.zip"),
        "source": str(args.output / "task3_source_baseline"),
        "mac_requirement": "Run the cross-task hardener/MHA filename guard before Docker export.",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

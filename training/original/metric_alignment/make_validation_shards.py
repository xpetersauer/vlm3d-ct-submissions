#!/usr/bin/env python3
"""Create deterministic symlink shards for resume-safe Task-1/2 validation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=64)
    args = parser.parse_args()
    files = sorted(path for path in args.input.iterdir() if path.is_file() and path.name.lower().endswith((".nii.gz", ".nii", ".mha", ".npy", ".npz")))
    if not files:
        raise SystemExit(f"No validation CTs in {args.input}")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for start in range(0, len(files), args.size):
        number = start // args.size
        shard = args.output / f"shard_{number:04d}"
        input_dir = shard / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        selected = files[start:start + args.size]
        for source in selected:
            target = input_dir / source.name
            if not target.exists():
                os.symlink(source.resolve(), target)
        rows.append({"name": shard.name, "cases": len(selected), "input": str(input_dir)})
    manifest = {"total_cases": len(files), "shard_size": args.size, "shards": rows}
    (args.output / "SHARDS.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

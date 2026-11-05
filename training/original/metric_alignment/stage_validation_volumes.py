#!/usr/bin/env python3
"""Stage cached CT tensors as resumable HU NIfTIs for validation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


SUFFIXES = (".nii.gz", ".nii", ".mha", ".npy", ".npz", ".pt", ".pth")


def stem(value) -> str:
    name = Path(str(value)).name
    for suffix in SUFFIXES:
        if name.lower().endswith(suffix):
            return name[:-len(suffix)]
    return Path(name).stem


def load(path: Path):
    lower = path.name.lower()
    if lower.endswith(".npy"):
        return np.load(path, allow_pickle=False)
    if lower.endswith(".npz"):
        archive = np.load(path, allow_pickle=False)
        for key in ("volume", "image", "tensor", "arr_0"):
            if key in archive:
                return archive[key]
        return archive[archive.files[0]]
    if lower.endswith((".pt", ".pth")):
        import torch
        try:
            value = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            value = torch.load(path, map_location="cpu")
        if isinstance(value, dict):
            value = next((value[key] for key in ("volume", "image", "tensor", "x", "data") if key in value), value)
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        return value
    import nibabel as nib
    return nib.load(str(path)).get_fdata(dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--cases-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-cases", type=int, default=0)
    args = parser.parse_args()
    import nibabel as nib
    import pandas as pd
    index = {}
    for path in args.cache.rglob("*"):
        if path.is_file() and path.name.lower().endswith(SUFFIXES):
            identifier = stem(path)
            if identifier in index and index[identifier] != path:
                raise SystemExit(f"Duplicate cache id: {identifier}")
            index[identifier] = path
    frame = pd.read_csv(args.cases_csv)
    column = next((name for name in ("VolumeName", "input_image_name", "case_id") if name in frame), None)
    if column is None:
        raise SystemExit(f"No case-id column in {args.cases_csv}")
    identifiers = [stem(value) for value in frame[column]]
    if args.max_cases > 0:
        identifiers = identifiers[: args.max_cases]
    missing = [identifier for identifier in identifiers if identifier not in index]
    if missing:
        raise SystemExit(f"Cache missing {len(missing)} selected cases; first={missing[0]}")
    args.output.mkdir(parents=True, exist_ok=True)
    for number, identifier in enumerate(identifiers, start=1):
        target = args.output / f"{identifier}.nii.gz"
        if target.is_file() and target.stat().st_size > 1024:
            continue
        array = np.asarray(load(index[identifier]), dtype=np.float32).squeeze()
        if array.ndim != 3:
            raise ValueError(f"{identifier}: expected 3-D cache tensor, got {array.shape}")
        array = np.nan_to_num(array, nan=-1000.0, posinf=1000.0, neginf=-1000.0)
        low, high = np.percentile(array, [0.1, 99.9])
        if low >= -1.25 and high <= 1.25:
            array = array * 1000.0
        hu = np.rint(np.clip(array, -1000.0, 1000.0)).astype(np.int16)
        image = nib.Nifti1Image(hu, np.eye(4))
        image.set_data_dtype(np.int16)
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp.nii.gz")
        nib.save(image, str(temporary))
        os.replace(temporary, target)
        if number % 50 == 0 or number == len(identifiers):
            print(f"[validation-cache] {number}/{len(identifiers)} staged", flush=True)
    manifest = {"status": "PASS", "cases": len(identifiers), "cache_files": len(index), "output": str(args.output)}
    (args.output / "STAGING_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

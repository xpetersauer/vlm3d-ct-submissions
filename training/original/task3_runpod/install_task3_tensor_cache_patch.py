#!/usr/bin/env python3
"""Teach the baseline Task-3 dataset to reuse local cached CT tensors.

The original notebook dataset always calls ``hf_hub_download`` for each case.
This patch first resolves a case by stem below the configured cache directory
and supports NIfTI, NPY, NPZ, PT, and PTH files.  Cached tensors that already
live in [-1, 1] are not normalized a second time.
"""

from __future__ import annotations

import ast
from pathlib import Path
import shutil
import sys
import textwrap


MARKER = "TASK3_LOCAL_TENSOR_CACHE_V1"


APPEND = textwrap.dedent(
    r'''

    # TASK3_LOCAL_TENSOR_CACHE_V1
    _TASK3_CACHE_INDEXES = {}
    _TASK3_ORIGINAL_GET_OR_DOWNLOAD = CTReportToVolumeDataset.get_or_download_volume


    def _task3_case_stem(value):
        name = Path(str(value)).name
        lower = name.lower()
        for suffix in (".nii.gz", ".nii", ".mha", ".npy", ".npz", ".pt", ".pth"):
            if lower.endswith(suffix):
                return name[:-len(suffix)]
        return Path(name).stem


    def _task3_cache_index(root):
        root = Path(root).resolve()
        key = str(root)
        cached = _TASK3_CACHE_INDEXES.get(key)
        if cached is not None:
            return cached
        supported = (".nii.gz", ".nii", ".mha", ".npy", ".npz", ".pt", ".pth")
        index = {}
        collisions = set()
        for path in root.rglob("*"):
            if not path.is_file() or not path.name.lower().endswith(supported):
                continue
            stem = _task3_case_stem(path.name)
            previous = index.get(stem)
            if previous is None:
                index[stem] = path
            elif previous.resolve() != path.resolve():
                collisions.add(stem)
        if collisions:
            first = sorted(collisions)[0]
            raise RuntimeError(f"Ambiguous cached CT stem {first!r} below {root}")
        _TASK3_CACHE_INDEXES[key] = index
        print(f"[task3-cache] indexed {len(index)} local volumes below {root}", flush=True)
        return index


    def _task3_get_or_download_volume(self, volume_name):
        case_id = _task3_case_stem(volume_name)
        local = _task3_cache_index(self.cache_dir).get(case_id)
        if local is not None:
            return local
        try:
            return _TASK3_ORIGINAL_GET_OR_DOWNLOAD(self, volume_name)
        except Exception as error:
            raise FileNotFoundError(
                f"CT {case_id!r} was absent from local cache {self.cache_dir} and "
                f"the Hugging Face fallback failed: {type(error).__name__}: {error}"
            ) from error


    def _task3_extract_array(path):
        path = Path(path)
        lower = path.name.lower()
        if lower.endswith(".npy"):
            return np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
        if lower.endswith(".npz"):
            archive = np.load(path, allow_pickle=False)
            for key in ("volume", "image", "tensor", "arr_0"):
                if key in archive:
                    return np.asarray(archive[key], dtype=np.float32)
            if len(archive.files) == 1:
                return np.asarray(archive[archive.files[0]], dtype=np.float32)
            raise ValueError(f"Could not identify a volume array in {path}")
        if lower.endswith((".pt", ".pth")):
            try:
                obj = torch.load(path, map_location="cpu", weights_only=True)
            except TypeError:
                obj = torch.load(path, map_location="cpu")
            if isinstance(obj, dict):
                for key in ("volume", "image", "tensor", "x", "data"):
                    if key in obj:
                        obj = obj[key]
                        break
            if torch.is_tensor(obj):
                obj = obj.detach().cpu().numpy()
            return np.asarray(obj, dtype=np.float32)
        volume, _ = load_volume(path)
        return np.asarray(volume, dtype=np.float32)


    def _task3_cached_volume_to_tensor(path, target_shape):
        volume = np.squeeze(_task3_extract_array(path))
        if volume.ndim != 3:
            raise ValueError(f"Expected cached 3-D CT, got {volume.shape} from {path}")
        volume = np.nan_to_num(volume, nan=-1000.0, posinf=1000.0, neginf=-1000.0)
        finite = volume[np.isfinite(volume)]
        low = float(np.percentile(finite, 0.1)) if finite.size else -1000.0
        high = float(np.percentile(finite, 99.9)) if finite.size else 1000.0
        if low >= -1.25 and high <= 1.25:
            # The shared 56-GB cache already contains model-space tensors.
            normalized = np.clip(volume, -1.0, 1.0)
        else:
            normalized = np.clip(volume, -1000.0, 1000.0)
            normalized = (normalized + 1000.0) / 1000.0 - 1.0
        x = torch.from_numpy(normalized.astype(np.float32, copy=False))[None, None]
        if tuple(volume.shape) != tuple(target_shape):
            x = torch.nn.functional.interpolate(
                x, size=tuple(target_shape), mode="trilinear", align_corners=False
            )
        return x[0]


    def _task3_cached_dataset_getitem(self, idx):
        row = self.df.iloc[idx]
        volume_name = row["VolumeName"]
        volume_path = self.get_or_download_volume(volume_name)
        x = _task3_cached_volume_to_tensor(volume_path, self.target_shape)
        report_text = row.get(self.report_col, "")
        if report_text is None or str(report_text).lower() == "nan":
            report_text = ""
        return x, str(report_text), volume_name


    CTReportToVolumeDataset.get_or_download_volume = _task3_get_or_download_volume
    CTReportToVolumeDataset.__getitem__ = _task3_cached_dataset_getitem
    '''
).rstrip() + "\n"


def install(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    if MARKER in source:
        print(f"OK already patched: {path}")
        return
    ast.parse(source, filename=str(path))
    backup = path.with_name(path.name + ".before_tensor_cache.bak")
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(source.rstrip() + "\n" + APPEND, encoding="utf-8")
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    print(f"PATCHED local tensor-cache support: {path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} /path/to/model_code_from_original.py")
    install(Path(sys.argv[1]).resolve())

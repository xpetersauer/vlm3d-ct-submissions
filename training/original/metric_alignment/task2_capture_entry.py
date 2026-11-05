#!/usr/bin/env python3
"""Run the shipped Task-2 process and capture candidates/reranker features.

This is a validation-only entrypoint.  It leaves the generator and current
reranker decisions intact while recording the exact numeric matrix passed to
``reranker.predict``.  Reference token metrics can then supervise a replacement
model without leaking references into hidden-test inference.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re

import numpy as np

from task2_diversity import install_decode_capture, install_generation_patch


VOLUME_SUFFIXES = (".nii.gz", ".nii", ".mha", ".npy", ".npz")


def case_id(value: str) -> str:
    name = Path(str(value)).name
    for suffix in VOLUME_SUFFIXES:
        if name.lower().endswith(suffix):
            return name[:-len(suffix)]
    return Path(name).stem


def references(path: Path) -> dict[str, str]:
    import pandas as pd
    frame = pd.read_csv(path)
    id_column = next((name for name in ("VolumeName", "input_image_name", "case_id") if name in frame), None)
    if id_column is None:
        raise ValueError(f"No case-id column in {path}")
    result = {}
    for _, row in frame.iterrows():
        findings = next((str(row[name]) for name in ("Findings_EN", "Findings", "findings") if name in frame and str(row[name]).lower() != "nan"), "")
        impression = next((str(row[name]) for name in ("Impressions_EN", "Impression_EN", "Impression", "impression") if name in frame and str(row[name]).lower() != "nan"), "")
        report = "\n\n".join(part for part in (
            f"FINDINGS: {findings.strip()}" if findings.strip() else "",
            f"IMPRESSION: {impression.strip()}" if impression.strip() else "",
        ) if part)
        if not report:
            report = next((str(row[name]) for name in ("Report_EN", "report") if name in frame), "")
        result[case_id(str(row[id_column]))] = report
    return result


class LoggingReranker:
    def __init__(self, wrapped, decoded: list[str], calls: list[dict]):
        self.wrapped = wrapped
        self.decoded = decoded
        self.calls = calls
        self.cursor = 0

    def _capture(self, values, scores):
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix[None, :]
        count = matrix.shape[0]
        available = self.decoded[self.cursor:]
        candidates = available[-count:] if len(available) >= count else available
        self.cursor = len(self.decoded)
        if len(candidates) != count:
            raise RuntimeError(
                f"Task-2 capture saw {count} reranker rows but {len(candidates)} newly decoded candidates"
            )
        score_array = np.asarray(scores, dtype=np.float64)
        if score_array.ndim == 2:
            score_array = score_array[:, -1]
        score_array = score_array.reshape(-1)
        self.calls.append({
            "features": matrix.tolist(),
            "candidates": candidates,
            "original_scores": score_array.tolist(),
        })
        return scores

    def predict(self, values):
        return self._capture(values, self.wrapped.predict(values))

    def predict_proba(self, values):
        return self._capture(values, self.wrapped.predict_proba(values))

    def __getattr__(self, name):
        return getattr(self.wrapped, name)


def main() -> int:
    capture_path = Path(os.environ["TASK2_CAPTURE_JSONL"])
    reference_path = Path(os.environ["TASK2_REFERENCE_CSV"])
    input_dir = Path(os.environ.get("VLM3D_INPUT", "/input"))
    raw_events = str(capture_path.with_suffix(".events.jsonl"))
    Path(raw_events).unlink(missing_ok=True)
    decoded: list[str] = []
    calls: list[dict] = []
    install_generation_patch(capture_path=raw_events)
    install_decode_capture(raw_events, decoded)

    import process
    original_loader = process.load_reranker

    def load_reranker(device):
        loaded = original_loader(device)
        if not isinstance(loaded, tuple) or len(loaded) != 4:
            raise RuntimeError("Expected load_reranker to return (config, reranker, tokenizer, RadBERT)")
        config, reranker, tokenizer, radbert = loaded
        print(f"[task2-capture] existing reranker={type(reranker).__module__}.{type(reranker).__name__}", flush=True)
        return config, LoggingReranker(reranker, decoded, calls), tokenizer, radbert

    process.load_reranker = load_reranker
    process.main()

    ids = sorted(case_id(path.name) for path in input_dir.iterdir() if path.is_file() and path.name.lower().endswith(VOLUME_SUFFIXES))
    reference_by_id = references(reference_path)
    if len(calls) != len(ids):
        raise RuntimeError(
            f"Task-2 capture expected one reranker call per CT: calls={len(calls)} cases={len(ids)}"
        )
    capture_path.parent.mkdir(parents=True, exist_ok=True)
    with capture_path.open("w", encoding="utf-8") as handle:
        for identifier, call in zip(ids, calls):
            reference = reference_by_id.get(identifier, "")
            if not reference.strip():
                raise RuntimeError(f"No reference report for {identifier}")
            row = {"case_id": identifier, "reference": reference, **call}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    unique = [
        len({re.sub(r"\s+", " ", value.strip().lower()) for value in call["candidates"]}) / len(call["candidates"])
        for call in calls
    ]
    print(
        f"[task2-capture] wrote {len(calls)} cases to {capture_path}; "
        f"mean unique-candidate ratio={float(np.mean(unique)):.3f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

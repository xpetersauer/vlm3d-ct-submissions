#!/usr/bin/env python3
"""Build Task-1/2 source overlays and metric-aligned weight ZIPs.

The original deliverables are never modified.  Apply the emitted overlays on
the Mac, then run the existing cross-task hardener and Docker preflight.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import zipfile


def locate(root: Path) -> Path:
    candidates = [root, root / "vlm3d_weeklong_deliverables"]
    candidates += list(root.rglob("vlm3d_weeklong_deliverables"))
    for candidate in candidates:
        if all((candidate / name).is_dir() for name in ("task1_source", "task2_source")):
            return candidate.resolve()
    raise SystemExit(f"Could not locate task1_source/task2_source under {root}")


def insert_docker_copy(path: Path, line: str, marker: str) -> None:
    source = path.read_text()
    if marker in source:
        return
    locations = [position for token in ("\nUSER ", "\nENTRYPOINT ", "\nCMD ") if (position := source.find(token)) >= 0]
    position = min(locations) if locations else len(source)
    source = source[:position].rstrip() + f"\n\n# {marker}\n{line}\n" + source[position:]
    path.write_text(source)


def patch_task1_entrypoint(path: Path) -> None:
    source = path.read_text()
    marker = "VLM3D_TASK1_METRIC_CALIBRATION_V1"
    command = (
        "python /opt/app/task1_calibrate_output.py "
        "--predictions /output/results.json "
        "--calibration /weights/task1_metric_calibration.json"
    )
    lines = source.splitlines()
    process_candidates = [
        index
        for index, line in enumerate(lines)
        if not line.lstrip().startswith("#")
        and re.search(r"(?<![\w.])(?:/opt/app/)?process\.py(?![\w.])", line)
    ]
    if len(process_candidates) != 1:
        raise SystemExit(
            f"Expected one executable Task-1 process invocation in {path}; "
            f"found {len(process_candidates)}"
        )

    process_line = lines[process_candidates[0]]
    indent = process_line[: len(process_line) - len(process_line.lstrip())]
    # An exec here would make every following calibration/contract command
    # unreachable.  Preserve the exact interpreter/arguments, but remove exec.
    process_line = re.sub(r"^(\s*)exec\s+", r"\1", process_line, count=1)

    # Canonicalize both fresh and previously patched entrypoints.  In
    # particular, repair the historical bad order:
    # process -> exec contract -> calibration (unreachable).
    filtered: list[str] = []
    insertion = None
    for index, line in enumerate(lines):
        if index == process_candidates[0]:
            insertion = len(filtered)
            continue
        if marker in line:
            continue
        if "task1_calibrate_output.py" in line:
            continue
        if "forithmus_contract_guard.py" in line and "--task task1" in line:
            continue
        if "task1 output gate" in line:
            continue
        filtered.append(line)
    if insertion is None:
        raise AssertionError("Task-1 insertion point disappeared")

    block = [
        process_line,
        f"{indent}# {marker}",
        f"{indent}{command}",
        f"{indent}exec python /opt/app/forithmus_contract_guard.py --task task1",
    ]
    filtered[insertion:insertion] = block
    if not any(line.strip() == "set -e" for line in filtered):
        shebang = 1 if filtered and filtered[0].startswith("#!") else 0
        filtered.insert(shebang, "set -e")
    path.write_text("\n".join(filtered).rstrip() + "\n")


def patch_task2_entrypoint(path: Path) -> None:
    source = path.read_text()
    marker = "VLM3D_TASK2_DIVERSE_ENTRY_V1"
    if marker in source:
        return
    # Hardened images have used several equivalent launch forms, including
    # ``exec /opt/conda/bin/python -u /opt/app/process.py``. Replace only the
    # script/module target so wrappers, interpreter paths, flags, arguments and
    # shell redirections remain intact. Requiring exactly one executable line
    # prevents a silent multi-command patch.
    lines = source.splitlines()
    candidates: list[int] = []
    for index, line in enumerate(lines):
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#"):
            continue
        has_script = re.search(r"(?<![\w.])(?:/opt/app/)?process\.py(?![\w.])", line)
        has_module = re.search(r"(?:^|\s)-m\s+process(?=$|\s|[;&|])", line)
        if has_script or has_module:
            candidates.append(index)
    if not candidates:
        already_patched = [
            index
            for index, line in enumerate(lines)
            if not line.lstrip().startswith("#") and "task2_diverse_entry.py" in line
        ]
        if len(already_patched) == 1:
            path.write_text(source.rstrip() + f"\n# {marker}\n")
            return
    if len(candidates) != 1:
        rendered = "\n".join(f"  {i + 1}: {lines[i]}" for i in candidates) or "  <none>"
        raise SystemExit(
            f"Expected one executable Task-2 process invocation in {path}; "
            f"found {len(candidates)}:\n{rendered}"
        )
    index = candidates[0]
    line, count = re.subn(
        r"(?<![\w.])(?:/opt/app/)?process\.py(?![\w.])",
        "/opt/app/task2_diverse_entry.py",
        lines[index],
    )
    if count == 0:
        line, count = re.subn(
            r"((?:^|\s))-m\s+process(?=$|\s|[;&|])",
            r"\1/opt/app/task2_diverse_entry.py",
            lines[index],
        )
    if count != 1:
        raise SystemExit(f"Could not safely patch Task-2 invocation in {path}: {lines[index]}")
    lines[index] = line
    source = "\n".join(lines) + ("\n" if source.endswith("\n") else "")
    source = source.rstrip() + f"\n# {marker}\n"
    path.write_text(source)


def patch_task2_radbert_loader(path: Path) -> None:
    """Permit only the deterministic RoBERTa position-id buffer mismatch.

    Older Transformers checkpoints persisted ``position_ids`` while newer
    releases reconstruct it.  Every learned tensor must still match exactly.
    """
    source = path.read_text()
    marker = "VLM3D_TASK2_RADBERT_POSITION_IDS_COMPAT_V1"
    if marker in source:
        return
    old = "rb.load_state_dict(state,strict=True)"
    new = (
        "_rb_incompat=rb.load_state_dict(state,strict=False); "
        "assert not _rb_incompat.missing_keys and "
        "set(_rb_incompat.unexpected_keys) <= {'model.embeddings.position_ids'}, "
        "f'RadBERT checkpoint incompatibility: {_rb_incompat}'; "
        f"print('[task2-weights] {marker}', _rb_incompat)"
    )
    if source.count(old) != 1:
        raise SystemExit(f"Expected one strict RadBERT load in {path}; found {source.count(old)}")
    path.write_text(source.replace(old, new))


def zip_tree(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".incomplete")
    temporary.unlink(missing_ok=True)
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for path in sorted(source.rglob("*")):
            if not path.is_file() or path.name in (".DS_Store",) or "__MACOSX" in path.parts:
                continue
            archive.write(path, path.relative_to(source).as_posix())
    with zipfile.ZipFile(temporary) as archive:
        bad = archive.testzip()
        if bad:
            raise RuntimeError(f"Corrupt ZIP member: {bad}")
    os.replace(temporary, target)


def extract_zip(path: Path, target: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        archive.extractall(target)
    children = [item for item in target.iterdir() if item.name != "__MACOSX"]
    if len(children) == 1 and children[0].is_dir():
        nested = children[0]
        for child in list(nested.iterdir()):
            shutil.move(str(child), target / child.name)
        nested.rmdir()


def load_gate(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deliverables", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    deliverables = locate(args.deliverables)
    args.output.mkdir(parents=True, exist_ok=True)
    kit = Path(__file__).resolve().parent

    task1_config_path = args.results / "task1" / "task1_metric_calibration.json"
    task2_config_path = args.results / "task2" / "reranker_metric_alignment.json"
    task1_config = load_gate(task1_config_path)
    task2_config = load_gate(task2_config_path)

    task1_overlay = args.output / "task1_source"
    task2_overlay = args.output / "task2_source"
    shutil.copytree(deliverables / "task1_source", task1_overlay, dirs_exist_ok=True)
    shutil.copytree(deliverables / "task2_source", task2_overlay, dirs_exist_ok=True)
    patch_task2_radbert_loader(task2_overlay / "process.py")
    if task1_config.get("accepted") is True:
        shutil.copy2(kit / "task1_calibrate_output.py", task1_overlay)
        insert_docker_copy(
            task1_overlay / "Dockerfile",
            "COPY task1_calibrate_output.py /opt/app/task1_calibrate_output.py",
            "VLM3D_TASK1_METRIC_CALIBRATION_COPY_V1",
        )
        patch_task1_entrypoint(task1_overlay / "entrypoint.sh")
    else:
        print("[package] Task 1 gate rejected; preserving original source/weights")
    if task2_config.get("accepted") is True:
        shutil.copy2(kit / "task2_diversity.py", task2_overlay)
        shutil.copy2(kit / "task2_diverse_entry.py", task2_overlay)
        shutil.copy2(kit / "vlm3d_reward_model.py", task2_overlay)
        insert_docker_copy(
            task2_overlay / "Dockerfile",
            "COPY task2_diversity.py task2_diverse_entry.py vlm3d_reward_model.py /opt/app/",
            "VLM3D_TASK2_DIVERSITY_COPY_V1",
        )
        patch_task2_entrypoint(task2_overlay / "entrypoint.sh")
    else:
        print("[package] Task 2 gate rejected; preserving original source/weights")

    with tempfile.TemporaryDirectory(prefix="task1_metric_weights_") as temporary:
        root = Path(temporary)
        extract_zip(deliverables / "task1_weights.zip", root)
        if task1_config.get("accepted") is True:
            shutil.copy2(task1_config_path, root / "task1_metric_calibration.json")
        zip_tree(root, args.output / "task1_weights_metric.zip")
    with tempfile.TemporaryDirectory(prefix="task2_metric_weights_") as temporary:
        root = Path(temporary)
        extract_zip(deliverables / "task2_weights.zip", root)
        if task2_config.get("accepted") is True:
            shutil.copy2(args.results / "task2" / "reranker.joblib", root / "reranker.joblib")
            config_path = root / "reranker.json"
            config = json.loads(config_path.read_text()) if config_path.is_file() else {}
            config["metric_alignment"] = task2_config
            config_path.write_text(json.dumps(config, indent=2) + "\n")
        zip_tree(root, args.output / "task2_weights_metric.zip")

    manifest = {
        "status": "PASS",
        "deliverables_source": str(deliverables),
        "task1": {"calibration": task1_config, "source_overlay": str(task1_overlay)},
        "task2": {"reranker": task2_config, "source_overlay": str(task2_overlay)},
        "task3": "Packaged separately after official FVD selection.",
    }
    (args.output / "PACKAGING_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "status": "PASS",
        "task1_weights": str(args.output / "task1_weights_metric.zip"),
        "task2_weights": str(args.output / "task2_weights_metric.zip"),
        "next": "Download this directory to the Mac and run mac_apply_and_harden.sh",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

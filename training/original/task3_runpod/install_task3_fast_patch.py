#!/usr/bin/env python3
"""Install the batched Task-3 inference function into one or more Python files.

Usage:
  python install_task3_fast_patch.py model_code.py model_code_from_original.py \
      forithmus_task3_docker/model_code_from_original.py

The installer makes one sibling backup before the first change and refuses to
touch files that do not define predict_task3_report_to_ct.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
import shutil
import sys
import tempfile
import textwrap


MARKER = "TASK3_BATCHED_RESUMABLE_INFERENCE_V3"

NEW_FUNCTION = textwrap.dedent(
    r'''
    def predict_task3_report_to_ct(
        input_path,
        autoencoder_checkpoint_path,
        mapper_checkpoint_dir,
        output_dir,
        output_format="npy",
        max_cases=None,
    ):
        """Batched, resumable Task-3 inference. TASK3_BATCHED_RESUMABLE_INFERENCE_V3."""
        import collections
        import gc
        import json
        import os
        import time
        import uuid
        from concurrent.futures import ThreadPoolExecutor
        from contextlib import nullcontext

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        allow_cpu = os.getenv("TASK3_ALLOW_CPU", "0") == "1"
        if device.type != "cuda" and not allow_cpu:
            raise RuntimeError(
                "Task 3 cannot meet the challenge deadline on CPU. CUDA was not "
                "detected inside the runtime. Fix the CUDA image/--gpus setting, or "
                "set TASK3_ALLOW_CPU=1 only for a tiny local smoke test."
            )

        batch_size = max(1, int(os.getenv("TASK3_BATCH_SIZE", "16")))
        save_workers = max(1, int(os.getenv("TASK3_SAVE_WORKERS", "4")))
        use_amp = device.type == "cuda" and os.getenv("TASK3_AMP", "1") == "1"
        resume = os.getenv("TASK3_RESUME", "1") == "1"
        progress_path_raw = os.getenv("TASK3_PROGRESS_PATH", "").strip()
        progress_path = Path(progress_path_raw) if progress_path_raw else None
        if progress_path is not None:
            progress_path.parent.mkdir(parents=True, exist_ok=True)
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

        print(
            f"[task3-fast] device={device} batch={batch_size} "
            f"save_workers={save_workers} amp={use_amp} resume={resume}",
            flush=True,
        )

        autoencoder, _ = load_task3_autoencoder_checkpoint(
            autoencoder_checkpoint_path,
            device=device,
        )
        mapper, tokenizer, mapper_ckpt = load_task3_mapper_checkpoint(
            mapper_checkpoint_dir,
            device=device,
        )
        autoencoder.eval()
        mapper.eval()

        prompts = read_task3_prompts(input_path)
        if max_cases is not None:
            prompts = prompts[:max_cases]
        max_text_length = mapper_ckpt["max_text_length"]

        def output_path_for(case_id):
            if output_format == "npy":
                return output_dir / f"{case_id}.npy"
            if output_format == "nii.gz":
                return output_dir / f"{case_id}.nii.gz"
            raise ValueError(f"Unsupported output_format: {output_format}")

        def looks_complete(path):
            if not path.is_file() or path.stat().st_size < 1024:
                return False
            if output_format == "npy":
                return True
            try:
                import nibabel as nib
                img = nib.load(str(path))
                return len(img.shape) == 3 and all(int(x) > 0 for x in img.shape)
            except Exception:
                return False

        written_paths = []
        pending = []
        for item in prompts:
            path = output_path_for(item["case_id"])
            if resume and looks_complete(path):
                written_paths.append(path)
            else:
                pending.append(item)

        resumed = len(prompts) - len(pending)
        completed_names = {path.name for path in written_paths}
        if resumed:
            print(f"[task3-fast] resuming after {resumed} completed volumes", flush=True)

        def write_progress(state, error=None):
            if progress_path is None:
                return
            payload = {
                "state": state,
                "total_cases": len(prompts),
                "completed_cases": len(completed_names),
                "remaining_cases": len(prompts) - len(completed_names),
                "completed_filenames": sorted(completed_names),
                "updated_unix": time.time(),
            }
            if error is not None:
                payload["error"] = str(error)
            tmp = progress_path.with_name(
                f".{progress_path.name}.{uuid.uuid4().hex}.tmp"
            )
            tmp.write_text(json.dumps(payload, indent=2))
            os.replace(tmp, progress_path)

        def atomic_save(path, volume_hu):
            if output_format == "npy":
                tmp = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp.npy")
                np.save(str(tmp), volume_hu.astype(np.float32, copy=False))
            else:
                import nibabel as nib
                tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp.nii.gz")
                # CT HU are naturally integral. int16 halves raw I/O versus float32;
                # rounding changes intensity by at most 0.5 HU.
                data = np.rint(volume_hu).clip(-32768, 32767).astype(np.int16)
                img = nib.Nifti1Image(data, affine=np.eye(4))
                nib.save(img, str(tmp))
            os.replace(tmp, path)
            return path

        started = time.perf_counter()
        executor = ThreadPoolExecutor(max_workers=save_workers)
        save_queue = collections.deque()
        cursor = 0

        def harvest_saves(block_one=False):
            """Record completed atomic writes; optionally wait for the oldest."""
            waited = False
            while save_queue:
                future = save_queue[0]
                if not future.done() and not (block_one and not waited):
                    break
                path = save_queue.popleft().result()
                completed_names.add(path.name)
                waited = True

        try:
            write_progress("running")
            while cursor < len(pending):
                current_bs = min(batch_size, len(pending) - cursor)

                while True:
                    batch = pending[cursor : cursor + current_bs]
                    reports = [str(item["report"]) for item in batch]
                    try:
                        tokenized = tokenizer(
                            reports,
                            padding=True,
                            truncation=True,
                            max_length=max_text_length,
                            return_tensors="pt",
                        )
                        input_ids = tokenized["input_ids"].to(device, non_blocking=True)
                        attention_mask = tokenized["attention_mask"].to(
                            device, non_blocking=True
                        )

                        amp_context = (
                            torch.autocast(device_type="cuda", dtype=torch.float16)
                            if use_amp
                            else nullcontext()
                        )
                        with torch.inference_mode(), amp_context:
                            z_pred = mapper(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                            )
                            x_pred = autoencoder.decode(z_pred)
                        # One transfer per batch instead of one transfer per case.
                        volume_norm = (
                            x_pred[:, 0]
                            .detach()
                            .to(device="cpu", dtype=torch.float32)
                            .numpy()
                        )
                        volume_hu = normalized_to_hu(volume_norm)
                        del tokenized, input_ids, attention_mask, z_pred, x_pred
                        break
                    except RuntimeError as exc:
                        is_oom = "out of memory" in str(exc).lower()
                        if device.type != "cuda" or not is_oom or current_bs == 1:
                            raise
                        current_bs = max(1, current_bs // 2)
                        batch_size = current_bs
                        print(
                            f"[task3-fast] CUDA OOM; retrying with batch={current_bs}",
                            flush=True,
                        )
                        gc.collect()
                        torch.cuda.empty_cache()

                for item, volume in zip(batch, volume_hu):
                    out_path = output_path_for(item["case_id"])
                    save_queue.append(executor.submit(atomic_save, out_path, volume))
                    written_paths.append(out_path)

                # Bound host RAM while still overlapping compression with GPU work.
                while len(save_queue) > save_workers * 4:
                    harvest_saves(block_one=True)

                harvest_saves(block_one=False)
                write_progress("running")

                cursor += current_bs
                complete = resumed + cursor
                elapsed = max(time.perf_counter() - started, 1e-6)
                rate = cursor / elapsed
                eta = (len(pending) - cursor) / max(rate, 1e-9)
                print(
                    f"[task3-fast] {complete}/{len(prompts)} complete; "
                    f"{rate:.2f} new volumes/s; ETA {eta / 60:.1f} min",
                    flush=True,
                )

            while save_queue:
                harvest_saves(block_one=True)
            write_progress("complete")
        except BaseException as exc:
            harvest_saves(block_one=False)
            write_progress("interrupted", error=exc)
            raise
        finally:
            executor.shutdown(wait=True, cancel_futures=False)

        elapsed = time.perf_counter() - started
        print(
            f"[task3-fast] finished {len(prompts)} volumes in "
            f"{elapsed / 60:.1f} min",
            flush=True,
        )
        return written_paths
    '''
).lstrip()


def install_one(path: Path) -> str:
    if not path.is_file():
        return f"SKIP missing: {path}"
    source = path.read_text()
    if MARKER in source:
        return f"OK already patched: {path}"

    tree = ast.parse(source, filename=str(path))
    matches = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "predict_task3_report_to_ct"
    ]
    if len(matches) != 1:
        return f"SKIP expected one target function, found {len(matches)}: {path}"

    node = matches[0]
    lines = source.splitlines(keepends=True)
    replacement = NEW_FUNCTION
    if not replacement.endswith("\n"):
        replacement += "\n"
    new_source = "".join(lines[: node.lineno - 1]) + replacement + "".join(lines[node.end_lineno :])
    compile(new_source, str(path), "exec")

    backup = path.with_name(path.name + ".before_task3_batching.bak")
    if not backup.exists():
        shutil.copy2(path, backup)

    mode = path.stat().st_mode
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(new_source)
        tmp = Path(handle.name)
    os.chmod(tmp, mode)
    os.replace(tmp, path)
    return f"PATCHED: {path} (backup: {backup.name})"


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    changed = 0
    for raw in sys.argv[1:]:
        result = install_one(Path(raw).resolve())
        print(result)
        changed += result.startswith("PATCHED")
    return 0 if changed or all(Path(p).is_file() for p in sys.argv[1:]) else 1


if __name__ == "__main__":
    raise SystemExit(main())

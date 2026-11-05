#!/usr/bin/env bash
# Run Task-1 calibration and Task-2 diverse-candidate reward distillation.
set -Eeuo pipefail

KIT_DIR="${KIT_DIR:-/workspace/vlm3d_metric_alignment_runpod_kit}"
DELIVERABLES_ROOT="${DELIVERABLES_ROOT:-/workspace/vlm3d_metric_workspace/vlm3d_weeklong_deliverables}"
STATE_ROOT="${STATE_ROOT:-/workspace/vlm3d_persistent/metric_alignment}"
TENSOR_CACHE="${TENSOR_CACHE:-/workspace/vlm3d_tensor_cache}"
VALID_REPORTS="${VALID_REPORTS:-/workspace/vlm3d_original_task_split/CT-RATE/dataset/radiology_text_reports/validation_reports.csv}"
VALID_LABELS="${VALID_LABELS:-}"
VALID_CASES="${VALID_CASES:-0}"
SHARD_SIZE="${TASK12_SHARD_SIZE:-64}"
BASELINE_PYTHON="${BASELINE_PYTHON:-/usr/local/bin/python}"
TASK1_PYTHON="${TASK1_PYTHON:-$BASELINE_PYTHON}"
TASK2_PYTHON="${TASK2_PYTHON:-$BASELINE_PYTHON}"
METRIC_PYTHON="${METRIC_PYTHON:-/workspace/vlm3d_envs/metric-tools/bin/python}"
WORK="$STATE_ROOT/task12_work"
RESULTS="$STATE_ROOT/results"

step() { printf '\n========== [%s/8] %s ==========\n' "$1" "$2"; }

step 1 "Resolve sources, labels, weights, and cached CTs"
for path in "$DELIVERABLES_ROOT/task1_source/process.py" "$DELIVERABLES_ROOT/task2_source/process.py" \
  "$DELIVERABLES_ROOT/task1_weights.zip" "$DELIVERABLES_ROOT/task2_weights.zip" \
  "$TENSOR_CACHE" "$VALID_REPORTS"; do
  [[ -e "$path" ]] || { echo "MISSING: $path" >&2; exit 2; }
done
if [[ -z "$VALID_LABELS" ]]; then
  VALID_LABELS="$(find /workspace -type f \( -iname '*validation*label*.csv' -o -iname 'validation_predicted_labels.csv' \) 2>/dev/null | head -n 1 || true)"
fi
[[ -n "$VALID_LABELS" && -f "$VALID_LABELS" ]] || {
  echo "ERROR: set VALID_LABELS to the CT-RATE validation abnormality-label CSV." >&2
  echo "The script will not invent Task-1 labels." >&2
  exit 2
}
echo "Deliverables: $DELIVERABLES_ROOT"
echo "Validation reports: $VALID_REPORTS"
echo "Validation labels: $VALID_LABELS"
echo "Tensor cache: $TENSOR_CACHE"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

step 2 "Create durable working copies and extract weights"
mkdir -p "$WORK" "$RESULTS/task1" "$RESULTS/task2"
if [[ ! -f "$WORK/.sources_ready" ]]; then
  rm -rf "$WORK/task1_source" "$WORK/task2_source" "$WORK/task1_weights" "$WORK/task2_weights"
  cp -a "$DELIVERABLES_ROOT/task1_source" "$WORK/task1_source"
  cp -a "$DELIVERABLES_ROOT/task2_source" "$WORK/task2_source"
  mkdir -p "$WORK/task1_weights" "$WORK/task2_weights"
  unzip -q "$DELIVERABLES_ROOT/task1_weights.zip" -d "$WORK/task1_weights"
  unzip -q "$DELIVERABLES_ROOT/task2_weights.zip" -d "$WORK/task2_weights"
  "$BASELINE_PYTHON" "$KIT_DIR/make_process_env_aware.py" "$WORK/task1_source/process.py"
  "$BASELINE_PYTHON" "$KIT_DIR/make_process_env_aware.py" "$WORK/task2_source/process.py"
  cp "$KIT_DIR/task2_diversity.py" "$KIT_DIR/task2_capture_entry.py" "$WORK/task2_source/"
  touch "$WORK/.sources_ready"
fi
echo "Working source checkpoint: $WORK/.sources_ready"

step 3 "Stage all selected validation CTs from the durable tensor cache"
"$METRIC_PYTHON" "$KIT_DIR/stage_validation_volumes.py" \
  --cache "$TENSOR_CACHE" \
  --cases-csv "$VALID_REPORTS" \
  --output "$WORK/validation_ct" \
  --max-cases "$VALID_CASES"

step 4 "Create durable validation shards (only an active shard can be lost)"
"$METRIC_PYTHON" "$KIT_DIR/make_validation_shards.py" \
  --input "$WORK/validation_ct" \
  --output "$WORK/shards" \
  --size "$SHARD_SIZE"

step 5 "Run current Task-1 and diverse Task-2 inference shard by shard"
while IFS= read -r shard; do
  name="$(basename "$shard")"
  mkdir -p "$shard/task1_output" "$shard/task2_output"
  echo "----- $name -----"
  if [[ ! -f "$shard/TASK1_PASS" ]]; then
    env PYTHONPATH="$WORK/task1_source" \
      VLM3D_INPUT="$shard/input" \
      VLM3D_OUTPUT="$shard/task1_output" \
      VLM3D_WEIGHTS="$WORK/task1_weights" \
      "$TASK1_PYTHON" "$WORK/task1_source/process.py"
    "$METRIC_PYTHON" "$KIT_DIR/extract_task1_validation_table.py" \
      --predictions "$shard/task1_output/results.json" \
      --labels "$VALID_LABELS" \
      --output "$shard/task1_validation.csv" \
      --minimum-rows 1
    touch "$shard/TASK1_PASS"
  else
    echo "RESUME: $name Task 1 PASS"
  fi
  if [[ ! -f "$shard/TASK2_PASS" ]]; then
    env PYTHONPATH="$WORK/task2_source:$KIT_DIR" \
      VLM3D_INPUT="$shard/input" \
      VLM3D_OUTPUT="$shard/task2_output" \
      VLM3D_WEIGHTS="$WORK/task2_weights" \
      TASK2_CAPTURE_JSONL="$shard/task2_capture.jsonl" \
      TASK2_REFERENCE_CSV="$VALID_REPORTS" \
      TASK2_DIVERSITY_SEED="$((29001 + 10#${name##*_}))" \
      "$TASK2_PYTHON" "$WORK/task2_source/task2_capture_entry.py"
    touch "$shard/TASK2_PASS"
  else
    echo "RESUME: $name Task 2 PASS"
  fi
done < <(find "$WORK/shards" -mindepth 1 -maxdepth 1 -type d -name 'shard_*' | sort)

"$METRIC_PYTHON" - "$WORK/shards" "$WORK/task1_validation.csv" "$WORK/task2_capture.jsonl" <<'PY'
from pathlib import Path
import sys
import pandas as pd
root, task1_out, task2_out = map(Path, sys.argv[1:])
tables = [pd.read_csv(path) for path in sorted(root.glob("shard_*/task1_validation.csv"))]
if not tables:
    raise SystemExit("No completed Task-1 shard tables")
pd.concat(tables, ignore_index=True).to_csv(task1_out, index=False)
with task2_out.open("w") as output:
    captures = sorted(root.glob("shard_*/task2_capture.jsonl"))
    if not captures:
        raise SystemExit("No completed Task-2 shard captures")
    for path in captures:
        output.write(path.read_text())
print(f"[shards] merged Task1 rows={sum(len(table) for table in tables)} Task2 shards={len(captures)}")
PY

step 6 "Tune monotonic Task-1 calibration on one split; lock-test on another"
"$METRIC_PYTHON" "$KIT_DIR/metric_alignment.py" task1-fit \
  --input "$WORK/task1_validation.csv" \
  --output "$RESULTS/task1"

step 7 "Distill held-out token + clinical rewards into an inference-time reranker"
"$METRIC_PYTHON" "$KIT_DIR/metric_alignment.py" task2-fit \
  --input "$WORK/task2_capture.jsonl" \
  --output "$RESULTS/task2" \
  --minimum-unique-ratio 0.75

step 8 "Package accepted Task-1/2 changes without modifying originals"
"$METRIC_PYTHON" "$KIT_DIR/install_submission_changes.py" \
  --deliverables "$DELIVERABLES_ROOT" \
  --results "$RESULTS" \
  --output "$STATE_ROOT/submission_overlays"
echo "TASKS 1-2 METRIC ALIGNMENT PASS"
echo "Results: $RESULTS"
echo "Mac overlays: $STATE_ROOT/submission_overlays"

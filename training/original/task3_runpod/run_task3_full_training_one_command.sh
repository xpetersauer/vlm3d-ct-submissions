#!/usr/bin/env bash
# Smoke-train then resume-safe full-data Task-3 training. No wall-clock timer.
set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-/workspace/vlm3d_task3/task3_source_baseline}"
WEIGHTS_ZIP="${WEIGHTS_ZIP:-/workspace/vlm3d_task3/task3_weights_baseline.zip}"
KIT_DIR="${KIT_DIR:-/workspace/vlm3d_task3_runpod_kit}"
STATE_ROOT="${STATE_ROOT:-/workspace/vlm3d_persistent}"
TRAIN_ROOT="${TASK3_FULL_TRAIN_ROOT:-$STATE_ROOT/task3_full_training_47149}"
SMOKE_ROOT="$STATE_ROOT/task3_training_smoke"
DATA_ROOT="${DATA_ROOT:-$STATE_ROOT/CT-RATE}"
CACHE_DIR="${CT_CACHE_DIR:-$STATE_ROOT/ct_rate_volume_cache}"
AE_EPOCHS="${AE_EPOCHS:-5}"
MAPPER_EPOCHS="${MAPPER_EPOCHS:-5}"
CHECKPOINT_EVERY="${TASK3_CHECKPOINT_EVERY_STEPS:-250}"
TRAIN_REPORTS_CSV="${TRAIN_REPORTS_CSV:-$DATA_ROOT/dataset/radiology_text_reports/train_reports.csv}"
BASELINE_PYTHON="${TASK3_BASELINE_PYTHON:-$(command -v python)}"

mkdir -p "$STATE_ROOT" "$TRAIN_ROOT" "$SMOKE_ROOT" "$DATA_ROOT" "$CACHE_DIR"
exec 9>"$TRAIN_ROOT/run.lock"
if command -v flock >/dev/null 2>&1 && ! flock -n 9; then
  echo "ERROR: the full Task-3 training pipeline is already running" >&2
  exit 3
fi

session_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$TRAIN_ROOT/logs"
exec > >(tee -a "$TRAIN_ROOT/logs/session_${session_stamp}.log") 2>&1

status_file="$TRAIN_ROOT/status.env"
write_status() {
  local state="$1"
  local tmp="$TRAIN_ROOT/.status.tmp.$$"
  {
    printf 'state=%q\n' "$state"
    printf 'updated_utc=%q\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'project_dir=%q\n' "$PROJECT_DIR"
    printf 'train_reports_csv=%q\n' "$TRAIN_REPORTS_CSV"
  } > "$tmp"
  mv "$tmp" "$status_file"
}
trap 'rc=$?; if [[ $rc -ne 0 ]]; then write_status failed_or_interrupted; fi' EXIT
write_status starting

for required in \
  "$PROJECT_DIR/task3_report_to_ct.py" \
  "$PROJECT_DIR/model_code_from_original.py" \
  "$PROJECT_DIR/requirements.txt" \
  "$WEIGHTS_ZIP" \
  "$KIT_DIR/install_task3_training_resume_patch.py" \
  "$KIT_DIR/install_task3_tensor_cache_patch.py"; do
  if [[ ! -f "$required" ]]; then
    echo "ERROR: missing required file: $required" >&2
    exit 2
  fi
done

echo "Task-3 full-training pipeline"
echo "Project: $PROJECT_DIR"
echo "Persistent state: $STATE_ROOT"
echo "AE epochs: $AE_EPOCHS"
echo "Mapper epochs: $MAPPER_EPOCHS"
echo "Checkpoint every $CHECKPOINT_EVERY optimizer steps"
df -hT /workspace
nvidia-smi

if ! command -v rg >/dev/null 2>&1 || ! command -v unzip >/dev/null 2>&1; then
  apt-get update
  apt-get install -y ripgrep unzip
fi

requirements_hash="$(sha256sum "$PROJECT_DIR/requirements.txt" | awk '{print $1}')"
requirements_marker="$STATE_ROOT/.task3_requirements_${requirements_hash}.ok"
if [[ ! -f "$requirements_marker" ]]; then
  "$BASELINE_PYTHON" -m pip install --no-cache-dir -r "$PROJECT_DIR/requirements.txt"
  touch "$requirements_marker"
fi

"$BASELINE_PYTHON" - <<'PY'
import torch
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
print("GPU:", torch.cuda.get_device_name(0))
PY

"$BASELINE_PYTHON" "$KIT_DIR/install_task3_training_resume_patch.py" \
  "$PROJECT_DIR/model_code_from_original.py"
"$BASELINE_PYTHON" "$KIT_DIR/install_task3_fast_patch.py" \
  "$PROJECT_DIR/model_code_from_original.py"
"$BASELINE_PYTHON" "$KIT_DIR/install_task3_tensor_cache_patch.py" \
  "$PROJECT_DIR/model_code_from_original.py"
"$BASELINE_PYTHON" -m py_compile "$PROJECT_DIR/model_code_from_original.py"

BASELINE_ROOT="$STATE_ROOT/shared_task3_weights"
mkdir -p "$BASELINE_ROOT"
unzip -oq "$WEIGHTS_ZIP" -d "$BASELINE_ROOT"
BASELINE_AE="$(find "$BASELINE_ROOT" -type f -name task3_autoencoder.pt -print -quit)"
BASELINE_MAPPER_PT="$(find "$BASELINE_ROOT" -type f -path '*/task3_report_to_latent/mapper.pt' -print -quit)"
if [[ -z "$BASELINE_AE" || -z "$BASELINE_MAPPER_PT" ]]; then
  echo "ERROR: baseline checkpoint layout was not found after extraction" >&2
  find "$BASELINE_ROOT" -maxdepth 4 -type f | sort | head -80
  exit 2
fi
BASELINE_MAPPER_DIR="$(dirname "$BASELINE_MAPPER_PT")"

if [[ ! -f "$TRAIN_REPORTS_CSV" ]]; then
  if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN must be configured to download gated CT-RATE data" >&2
    exit 2
  fi
  python - "$DATA_ROOT" <<'PY'
import os
from pathlib import Path
import sys
from huggingface_hub import hf_hub_download

root = Path(sys.argv[1])
path = hf_hub_download(
    repo_id="ibrahimhamamci/CT-RATE",
    repo_type="dataset",
    token=os.environ["HF_TOKEN"],
    filename="train_reports.csv",
    subfolder="dataset/radiology_text_reports",
    local_dir=str(root),
)
print("Training reports:", path)
PY
fi
test -f "$TRAIN_REPORTS_CSV"
FULL_TRAIN_ROWS="$($BASELINE_PYTHON - "$TRAIN_REPORTS_CSV" <<'PY'
import pandas as pd
import sys
print(len(pd.read_csv(sys.argv[1])))
PY
)"
if [[ "$FULL_TRAIN_ROWS" -lt 1000 ]]; then
  echo "ERROR: expected the full training CSV, found only $FULL_TRAIN_ROWS rows" >&2
  exit 2
fi
echo "Full training rows: $FULL_TRAIN_ROWS"

# Keep the large lazy CT cache on persistent storage even though the original
# entrypoint uses a relative default path.
if [[ -L "$PROJECT_DIR/ct_rate_volume_cache" ]]; then
  current_target="$(readlink "$PROJECT_DIR/ct_rate_volume_cache")"
  if [[ "$current_target" != "$CACHE_DIR" ]]; then
    echo "ERROR: existing cache symlink points to $current_target, expected $CACHE_DIR" >&2
    exit 2
  fi
elif [[ -e "$PROJECT_DIR/ct_rate_volume_cache" ]]; then
  echo "ERROR: $PROJECT_DIR/ct_rate_volume_cache exists and is not a symlink" >&2
  echo "Move that cache into $CACHE_DIR before retrying." >&2
  exit 2
else
  ln -s "$CACHE_DIR" "$PROJECT_DIR/ct_rate_volume_cache"
fi

initialize_checkpoints() {
  local target_root="$1"
  local target_ae="$target_root/checkpoints/task3_autoencoder.pt"
  local target_mapper="$target_root/checkpoints/task3_report_to_latent"
  mkdir -p "$target_mapper"
  if [[ ! -f "$target_ae" ]]; then
    cp "$BASELINE_AE" "$target_ae"
  fi
  if [[ ! -f "$target_mapper/mapper.pt" ]]; then
    cp -R "$BASELINE_MAPPER_DIR/." "$target_mapper/"
  fi
}

initialize_checkpoints "$SMOKE_ROOT"
initialize_checkpoints "$TRAIN_ROOT"

run_training() {
  local root="$1"
  local rows_arg="$2"
  local ae_epochs="$3"
  local mapper_epochs="$4"
  local progress="$root/progress.json"
  local -a extra_args=()
  if [[ -n "$rows_arg" ]]; then
    extra_args+=(--max-rows "$rows_arg")
  fi
  (
    cd "$PROJECT_DIR"
    env \
      TASK3_CHECKPOINT_EVERY_STEPS="$CHECKPOINT_EVERY" \
      TASK3_TRAIN_PROGRESS_PATH="$progress" \
      "$BASELINE_PYTHON" task3_report_to_ct.py \
        --mode train \
        --train-reports-csv "$TRAIN_REPORTS_CSV" \
        --autoencoder-checkpoint "$root/checkpoints/task3_autoencoder.pt" \
        --mapper-checkpoint-dir "$root/checkpoints/task3_report_to_latent" \
        --ae-epochs "$ae_epochs" \
        --mapper-epochs "$mapper_epochs" \
        "${extra_args[@]}"
  )
}

if [[ ! -f "$SMOKE_ROOT/SMOKE_PASS" ]]; then
  write_status smoke_training
  echo "Running 10-row, one-epoch-per-stage training smoke test"
  run_training "$SMOKE_ROOT" 10 1 1
  touch "$SMOKE_ROOT/SMOKE_PASS"
fi

write_status full_training
echo "Starting full-data training: no --max-rows cap"
echo "Passing --max-rows=$FULL_TRAIN_ROWS explicitly because the upstream CLI defaults to 50."
run_training "$TRAIN_ROOT" "$FULL_TRAIN_ROWS" "$AE_EPOCHS" "$MAPPER_EPOCHS"

write_status packaging
(
  cd "$PROJECT_DIR"
  "$BASELINE_PYTHON" task3_report_to_ct.py \
    --mode make_weights \
    --autoencoder-checkpoint "$TRAIN_ROOT/checkpoints/task3_autoencoder.pt" \
    --mapper-checkpoint-dir "$TRAIN_ROOT/checkpoints/task3_report_to_latent" \
    --weights-zip "$TRAIN_ROOT/task3_weights_full.zip"
)
unzip -tq "$TRAIN_ROOT/task3_weights_full.zip"

tar -czf "$TRAIN_ROOT/task3_full_trained_source.tar.gz" \
  -C "$PROJECT_DIR" \
  model_code_from_original.py task3_report_to_ct.py requirements.txt \
  Dockerfile process.py entrypoint.sh vlm3d_common.py

python - "$TRAIN_ROOT" "$AE_EPOCHS" "$MAPPER_EPOCHS" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
result = {
    "status": "PASS",
    "dataset": "complete train_reports.csv (no max_rows cap)",
    "autoencoder_epochs": int(sys.argv[2]),
    "mapper_epochs": int(sys.argv[3]),
    "weights_zip": str(root / "task3_weights_full.zip"),
}
(root / "RESULTS.json").write_text(json.dumps(result, indent=2))
(root / "RESULTS.md").write_text(
    "# Task 3 full-data training\n\n"
    "- Status: **PASS**\n"
    "- Dataset: complete `train_reports.csv` (no `--max-rows`)\n"
    f"- Autoencoder epochs: **{result['autoencoder_epochs']}**\n"
    f"- Mapper epochs: **{result['mapper_epochs']}**\n"
    "- Weights: `task3_weights_full.zip`\n"
)
PY

write_status complete
trap - EXIT
echo "Task-3 full-data training PASS"
echo "Artifacts: $TRAIN_ROOT"

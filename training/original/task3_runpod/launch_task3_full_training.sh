#!/usr/bin/env bash
# One-command RunPod launcher. Re-running resumes the same full-data job.
set -Eeuo pipefail

PAYLOAD="${PAYLOAD:-/workspace/vlm3d_task3_runpod_payload.tar.gz}"
KIT_DIR="${KIT_DIR:-/workspace/vlm3d_task3_runpod_kit}"
STATE_ROOT="${STATE_ROOT:-/workspace/vlm3d_persistent}"
PROJECT_ROOT="${PROJECT_ROOT:-/workspace/vlm3d_task3}"
PROJECT_DIR="$PROJECT_ROOT/task3_source_baseline"
WEIGHTS_ZIP="$PROJECT_ROOT/task3_weights_baseline.zip"
SUPERVISOR_LOG="$STATE_ROOT/task3_full_training_supervisor.log"
PID_FILE="$STATE_ROOT/task3_full_training_supervisor.pid"

if [[ -f "$PID_FILE" ]]; then
  previous_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "$previous_pid" ]] && kill -0 "$previous_pid" 2>/dev/null; then
    echo "Task-3 full training is already running as PID $previous_pid"
    echo "Log: $SUPERVISOR_LOG"
    exit 0
  fi
fi

if [[ ! -f "$PROJECT_DIR/task3_report_to_ct.py" || ! -f "$WEIGHTS_ZIP" ]]; then
  if [[ ! -f "$PAYLOAD" ]]; then
    echo "ERROR: missing payload: $PAYLOAD" >&2
    exit 2
  fi
  mkdir -p "$PROJECT_ROOT"
  tar -xzf "$PAYLOAD" -C "$PROJECT_ROOT"
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  if [[ ! -t 0 ]]; then
    echo "ERROR: configure HF_TOKEN in the RunPod environment before launching" >&2
    exit 2
  fi
  read -r -s -p "Hugging Face token for gated CT-RATE access: " HF_TOKEN
  echo
  export HF_TOKEN
fi

mkdir -p "$STATE_ROOT"
export PROJECT_DIR WEIGHTS_ZIP KIT_DIR STATE_ROOT
export AE_EPOCHS="${AE_EPOCHS:-5}"
export MAPPER_EPOCHS="${MAPPER_EPOCHS:-5}"
export TASK3_CHECKPOINT_EVERY_STEPS="${TASK3_CHECKPOINT_EVERY_STEPS:-250}"

nohup setsid bash "$KIT_DIR/run_task3_full_training_one_command.sh" \
  > "$SUPERVISOR_LOG" 2>&1 < /dev/null &
pid="$!"
echo "$pid" > "$PID_FILE"

echo "Started Task-3 smoke → complete-dataset training as PID $pid"
echo "No wall-clock timeout is active."
echo "Log: $SUPERVISOR_LOG"
echo "Progress: $STATE_ROOT/task3_full_training/progress.json"
echo "Re-run this launcher after a pod stop to resume from the latest step checkpoint."

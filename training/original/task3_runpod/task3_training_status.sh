#!/usr/bin/env bash
set -Eeuo pipefail
STATE_ROOT="${STATE_ROOT:-/workspace/vlm3d_persistent}"
TRAIN_ROOT="${TASK3_FULL_TRAIN_ROOT:-$STATE_ROOT/task3_full_training_47149}"
echo "Supervisor"
tail -n 80 "$STATE_ROOT/task3_full_training_supervisor.log" 2>/dev/null || true
echo
echo "Status"
cat "$TRAIN_ROOT/status.env" 2>/dev/null || true
echo
echo "Progress"
python -m json.tool "$TRAIN_ROOT/progress.json" 2>/dev/null \
  | sed -n '1,40p' || true
echo
echo "Artifacts"
find "$TRAIN_ROOT" -maxdepth 2 -type f \
  \( -name 'RESULTS.*' -o -name '*.zip' -o -name '*.pt' \) \
  -exec ls -lh {} \; 2>/dev/null || true

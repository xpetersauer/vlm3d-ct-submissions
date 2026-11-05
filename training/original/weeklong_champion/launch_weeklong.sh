#!/usr/bin/env bash
set -euo pipefail
ROOT=/workspace/vlm3d_original_task_split
cd "$ROOT"

# Cheap preflight before detaching. This does not import/download models or touch the active trainer.
python -m py_compile weeklong_champion/champion/*.py weeklong_champion/templates/task1/*.py weeklong_champion/templates/task2/*.py
bash -n weeklong_champion/*.sh weeklong_champion/templates/task1/*.sh weeklong_champion/templates/task2/*.sh
echo "Weeklong source preflight: OK"
if tmux has-session -t weeklong_champion 2>/dev/null; then
  echo "weeklong_champion is already running."
  tmux ls
  exit 0
fi
mkdir -p /workspace/vlm3d_weeklong/logs
chmod +x "$ROOT/weeklong_champion/queue_weeklong.sh" "$ROOT/weeklong_champion/status_weeklong.sh"
tmux new-session -d -s weeklong_champion \
  "cd '$ROOT' && ./weeklong_champion/queue_weeklong.sh 2>&1 | tee /workspace/vlm3d_weeklong/logs/MASTER.log"
sleep 2
echo "Launched detached tmux session: weeklong_champion"
echo "This DOES NOT stop or alter sota_all; the queue waits for it."
tmux ls
tail -30 /workspace/vlm3d_weeklong/logs/MASTER.log || true

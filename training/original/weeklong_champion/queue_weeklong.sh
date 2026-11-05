#!/usr/bin/env bash
set -uo pipefail

ROOT=/workspace/vlm3d_original_task_split
BUNDLE="$ROOT/weeklong_champion"
WORK=/workspace/vlm3d_weeklong
LOG="$WORK/logs"
STAGE="$WORK/stages"
mkdir -p "$LOG" "$STAGE" "$WORK/preserved"
cd "$ROOT"
export PYTHONPATH="$BUNDLE/champion:$ROOT:$ROOT/sota_deadline:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

stamp(){ date '+%Y-%m-%d %H:%M:%S %z'; }
status(){ printf '%s\t%s\t%s\t%s\n' "$(stamp)" "$1" "$2" "${3:-}" | tee -a "$WORK/stage_status.tsv"; }

run_stage(){
  local mode="$1"; local name="$2"; shift 2
  local done="$STAGE/$name.done"; local failed="$STAGE/$name.failed"; local running="$STAGE/$name.running"; local log="$LOG/$name.log"
  if [[ -f "$done" ]]; then echo "[$(stamp)] SKIP already complete: $name"; return 0; fi
  rm -f "$failed"; touch "$running"; status "$name" STARTED "$*"
  set +e
  "$@" 2>&1 | tee "$log"
  local rc=${PIPESTATUS[0]}
  set -e
  rm -f "$running"
  if [[ $rc -eq 0 ]]; then touch "$done"; status "$name" COMPLETE; return 0; fi
  echo "$rc" > "$failed"; status "$name" FAILED "exit=$rc"
  if [[ "$mode" == required ]]; then
    echo "REQUIRED STAGE FAILED: $name" >&2
    exit "$rc"
  fi
  echo "Optional stage failed; continuing: $name" >&2
  return 0
}

set -e
status queue STARTED "weeklong champion queue"

echo "======================================================"
echo "WAITING FOR CURRENT sota_all RUN — IT IS NOT KILLED"
echo "======================================================"
while tmux has-session -t sota_all 2>/dev/null || pgrep -f 'train_sota_task[12]\.py.*sota_task[12]_ALL_5ep' >/dev/null 2>&1; do
  echo "[$(stamp)] sota_all still active; sleeping 5 minutes..."
  if [[ -f logs/SOTA_ALL_5EP_MASTER.log ]]; then
    grep -E 'DATA AUDIT|epoch=|TASK [12]|COMPLETE|Traceback|ERROR' logs/SOTA_ALL_5EP_MASTER.log | tail -8 || true
  fi
  sleep 300
done
status wait_for_sota_all COMPLETE "current training no longer active"

# Preserve all important prior artifacts without deleting anything.
run_stage required preserve bash -lc '
set -e
mkdir -p /workspace/vlm3d_weeklong/preserved
cd /workspace/vlm3d_original_task_split
for p in \
  checkpoints/sota_task1_ALL_5ep/best.pt \
  checkpoints/sota_task2_ALL_5ep/best.pt \
  checkpoints/sota_task1_5k_25ep/best.pt \
  checkpoints/sota_task1_5k/best.pt \
  checkpoints/sota_task2_5k/best.pt \
  /workspace/vlm3d_DEADLINE_5k_for_mac.tar.gz; do
  if [ -e "$p" ]; then cp -a "$p" /workspace/vlm3d_weeklong/preserved/$(basename "$(dirname "$p")")_$(basename "$p") 2>/dev/null || cp -a "$p" /workspace/vlm3d_weeklong/preserved/; fi
done
'

# Report-derived soft targets. Optional: a gated/missing HF text-classifier asset must not sink the run.
run_stage optional radbert_soft python -u "$BUNDLE/champion/radbert_soft_targets.py"

# Budget-friendly foundation-model branch: reuse compact CTs, freeze DINOv2, cache slice features once.
run_stage optional dino_features python -u "$BUNDLE/champion/extract_dino_features.py" --n-slices 24 --image-size 224 --image-batch 96

if [[ -f "$WORK/dino_features/train_features.f16.mmap" && -f "$WORK/dino_features/valid_features.f16.mmap" ]]; then
  run_stage optional dino_bce python -u "$BUNDLE/champion/train_dino_head.py" --variant bce --seed 17 --epochs 15 --patience 4 --out "$WORK/dino_bce"
  run_stage optional dino_asl python -u "$BUNDLE/champion/train_dino_head.py" --variant asl --seed 17 --epochs 15 --patience 4 --out "$WORK/dino_asl"
  run_stage optional dino_full_s17 python -u "$BUNDLE/champion/train_dino_head.py" --variant full --seed 17 --epochs 15 --patience 4 --out "$WORK/dino_full_s17"
  run_stage optional dino_full_s29 python -u "$BUNDLE/champion/train_dino_head.py" --variant full --seed 29 --epochs 15 --patience 4 --out "$WORK/dino_full_s29"
  run_stage optional dino_full_s43 python -u "$BUNDLE/champion/train_dino_head.py" --variant full --seed 43 --epochs 15 --patience 4 --out "$WORK/dino_full_s43"
else
  status dino_heads SKIPPED "DINO feature stage unavailable"
fi

# One full-image structured fine-tune with the objectives discussed: ASL + EMA +
# reconstruction consistency + report soft targets (when RadBERT succeeded).
run_stage optional structured_swin python -u "$BUNDLE/champion/train_structured_swin.py" --epochs 3 --batch-size 2 --out "$WORK/structured_swin"

# Always obtain a baseline prediction candidate. Prefer current all-data checkpoint; fall back safely.
BASE_CKPT=""
for p in \
  "$ROOT/checkpoints/sota_task1_ALL_5ep/best.pt" \
  "$ROOT/checkpoints/sota_task1_5k_25ep/best.pt" \
  "$ROOT/checkpoints/sota_task1_5k/best.pt"; do
  if [[ -f "$p" ]]; then BASE_CKPT="$p"; break; fi
done
if [[ -z "$BASE_CKPT" ]]; then echo "No Task1 checkpoint exists" >&2; exit 20; fi
mkdir -p "$WORK/baseline_swin"; printf '%s\n' "$BASE_CKPT" > "$WORK/baseline_swin/source_checkpoint.txt"
run_stage required baseline_swin python -u "$BUNDLE/champion/predict_swin.py" --checkpoint "$BASE_CKPT" --out "$WORK/baseline_swin"

# Tournament + conditional graph + separate calibration + untouched audit partition.
run_stage required posthoc python -u "$BUNDLE/champion/posthoc_select.py"

# Report generate-and-rerank is optional; if it does not beat beam-0 on its check set,
# packaging automatically keeps ordinary generation.
run_stage optional task2_rerank python -u "$BUNDLE/champion/task2_rerank.py" --max-cases 180 --candidates 4 --max-new-tokens 220

# Produce thin Docker source folders, weights ZIPs, metrics/writeup, and one download archive.
run_stage required package python -u "$BUNDLE/champion/package_deliverables.py"

status queue COMPLETE "/workspace/vlm3d_WEEKLONG_CHAMPION_DELIVERABLES.tar.gz"
echo "======================================================"
echo "WEEKLONG CHAMPION PIPELINE COMPLETE"
echo "Download: /workspace/vlm3d_WEEKLONG_CHAMPION_DELIVERABLES.tar.gz"
echo "======================================================"

#!/usr/bin/env bash
set +e
WORK=/workspace/vlm3d_weeklong
ROOT=/workspace/vlm3d_original_task_split

echo "===== TMUX ====="
tmux ls 2>/dev/null || echo "no tmux sessions"
echo
echo "===== ACTIVE GPU/TRAINING PROCESSES ====="
ps -eo pid,etime,%cpu,%mem,args | grep -E 'queue_weeklong|extract_dino|train_dino_head|train_structured_swin|task2_rerank|train_sota_task[12]\.py' | grep -v grep || true
echo
echo "===== STAGES ====="
tail -30 "$WORK/stage_status.tsv" 2>/dev/null || echo "No stage status yet"
echo
echo "===== DINO FEATURE PROGRESS ====="
python - <<'PY' 2>/dev/null || true
from pathlib import Path
import numpy as np
w=Path('/workspace/vlm3d_weeklong/dino_features')
for split in ['train','valid']:
 p=w/f'{split}_done.npy'
 if p.exists():
  x=np.load(p); print(f'{split}: {int(x.sum()):,}/{len(x):,} ({100*x.mean():.1f}%)')
PY
echo
echo "===== RECENT MODEL CURVES ====="
for f in "$WORK"/dino_*/history.csv "$WORK"/structured_swin/history.csv; do
  [ -f "$f" ] || continue
  echo "--- $f"; tail -5 "$f"
done
echo
echo "===== CACHE ====="
echo -n "compact tensors: "; find /workspace/vlm3d_tensor_cache/64x96x96 -type f -name '*.pt' 2>/dev/null | wc -l
du -sh /workspace/vlm3d_tensor_cache "$WORK" 2>/dev/null || true
echo
echo "===== FINAL RECIPE / DELIVERABLE ====="
[ -f "$WORK/posthoc/final_recipe.json" ] && python - <<'PY'
import json
p='/workspace/vlm3d_weeklong/posthoc/final_recipe.json'; r=json.load(open(p)); print('selected:',r['selected_name']); print('graph:',r['graph']['accepted']); print('calibration:',r['calibration']['accepted']); print('audit:',r['audit_metrics'])
PY
ls -lh /workspace/vlm3d_WEEKLONG_CHAMPION_DELIVERABLES.tar.gz 2>/dev/null || echo "final archive not ready"
echo
echo "===== GPU ====="
nvidia-smi 2>/dev/null || true
echo
echo "===== MASTER LOG TAIL ====="
tail -35 "$WORK/logs/MASTER.log" 2>/dev/null || true

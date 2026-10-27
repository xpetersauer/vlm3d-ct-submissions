#!/usr/bin/env bash
set -euo pipefail
mkdir -p /output
# VLM3D_FORITHMUS_STANDALONE_MAC_V4: task2 output gate
python -u /opt/app/task2_diverse_entry.py
exec python /opt/app/forithmus_contract_guard.py --task task2
# VLM3D_TASK2_DIVERSE_ENTRY_V1

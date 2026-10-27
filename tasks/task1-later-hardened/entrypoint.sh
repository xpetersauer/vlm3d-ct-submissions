#!/usr/bin/env bash
set -euo pipefail
mkdir -p /output
# VLM3D_FORITHMUS_STANDALONE_MAC_V4: task1 output gate
python -u /opt/app/process.py
exec python /opt/app/forithmus_contract_guard.py --task task1
# VLM3D_TASK1_METRIC_CALIBRATION_V1
python /opt/app/task1_calibrate_output.py --predictions /output/results.json --calibration /weights/task1_metric_calibration.json

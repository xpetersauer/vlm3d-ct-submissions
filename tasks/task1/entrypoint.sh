#!/usr/bin/env bash
set -e
set -euo pipefail
mkdir -p /output
python -u /opt/app/process.py
# VLM3D_TASK1_METRIC_CALIBRATION_ORDER_V2
python /opt/app/task1_calibrate_output.py --predictions /output/results.json --calibration /weights/task1_metric_calibration.json
exec python /opt/app/forithmus_contract_guard.py --task task1

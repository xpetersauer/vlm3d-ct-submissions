#!/bin/sh
set -eu
mkdir -p /output
# VLM3D_FORITHMUS_STANDALONE_MAC_V4: task3 output gate
python /opt/app/process.py
exec python /opt/app/forithmus_contract_guard.py --task task3

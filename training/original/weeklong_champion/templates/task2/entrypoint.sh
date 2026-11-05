#!/usr/bin/env bash
set -euo pipefail
mkdir -p /output
exec python -u /opt/app/process.py

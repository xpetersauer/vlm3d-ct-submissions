#!/usr/bin/env bash
set -euo pipefail
TAG=${1:-vlm3d-weeklong-task1:latest}
docker build --platform linux/amd64 -t "$TAG" .
docker save "$TAG" | gzip > submission_task1.tar.gz
echo "Built submission_task1.tar.gz"

#!/usr/bin/env bash
set -euo pipefail
TAG=${1:-vlm3d-weeklong-task2:latest}
docker build --platform linux/amd64 -t "$TAG" .
docker save "$TAG" | gzip > submission_task2.tar.gz
echo "Built submission_task2.tar.gz"

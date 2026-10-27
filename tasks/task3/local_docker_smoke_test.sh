#!/bin/sh
set -eu
# Run after `forithmus generate` creates .forithmus/test_data/input.
# This test mounts ./weights/ directly. Unzip your weights.zip to ./weights first.
rm -rf ./_local_output
mkdir -p ./_local_output ./weights
if [ -f ../weights_task3.zip ] && [ ! -e ./weights/.unzipped ]; then
  unzip -o ../weights_task3.zip -d ./weights
  touch ./weights/.unzipped
fi
docker run --rm --gpus all   -v "$PWD/.forithmus/test_data/input:/input:ro"   -v "$PWD/_local_output:/output"   -v "$PWD/weights:/weights:ro"   vlm3d-task3-ctgen:latest
python forithmus_validate_outputs.py task3 ./_local_output

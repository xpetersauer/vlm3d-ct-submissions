#!/bin/sh
set -eu
# Official format smoke test. Requires: pip install --upgrade forithmus; forithmus login; forithmus init ct-volume-generation
forithmus generate
./build.sh
forithmus test vlm3d-task3-ctgen:latest --timeout 600
# You can also validate a previously-created output directory:
# forithmus validate ./_local_output --input ./.forithmus/test_data/input

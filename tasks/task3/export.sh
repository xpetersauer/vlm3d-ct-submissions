#!/bin/sh
set -eu
docker save vlm3d-task3-ctgen:latest | gzip > vlm3d-task3-ctgen.tar.gz
ls -lh vlm3d-task3-ctgen.tar.gz

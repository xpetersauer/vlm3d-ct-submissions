#!/bin/sh
set -eu
# Edit -d to name the method shown on leaderboard. Edit --time-budget if needed.
forithmus init ct-volume-generation
forithmus submit vlm3d-task3-ctgen.tar.gz   --phase main-2026   --tier gpu-l4-xl   --time-budget 1200   --weights ../weights_task3.zip   -d "AE+DistilBERT report-to-CT from original notebook"

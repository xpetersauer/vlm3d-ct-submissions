# VLM3D weeklong set-and-forget pipeline

This folder is designed to be extracted **inside** `/workspace/vlm3d_original_task_split` while the existing `sota_all` tmux job is still running.

## Safety behavior

- It does **not** kill, edit, resume, or compete with the current `sota_all` run.
- `launch_weeklong.sh` starts a separate detached tmux session named `weeklong_champion`.
- The queue sleeps until both the `sota_all` tmux session and the all-data Task-1/Task-2 trainer processes are gone.
- Existing 5k and all-data checkpoints are never deleted or overwritten.
- Stages are restartable using `.done` files under `/workspace/vlm3d_weeklong/stages`.
- Optional experimental stages fail open; the known Task-1 checkpoint remains a valid fallback.
- The script prepares deliverables but does **not** automatically submit to Forithmus. Authentication can expire, and final Docker smoke testing should be done when you return.

The RunPod itself must stay running. Closing SSH is fine; stopping/terminating the Pod stops compute.

## Core experiment

After the existing all-data job finishes, the queue runs:

1. Official CT-RATE RadBERT on real training reports to cache 18-dimensional soft report targets. If the gated checkpoint cannot be downloaded, this stage is skipped rather than killing the pipeline.
2. A cost-controlled foundation-model branch: frozen DINOv2 slice features + learned 18-label query-attention pooling. It reuses the already compact-cached CTs. This is **AnyMC3D-inspired**, not an exact AnyMC3D reproduction; full-resolution LoRA backbone adaptation is intentionally omitted for cost/storage reasons.
3. Cheap DINO head ablations: BCE; ASL+EMA; and ASL+EMA + report soft supervision + same-scan reconstruction consistency, with three seeds for the full head.
4. One 3-epoch all-data structured fine-tune of the existing interacting-label-token Swin using ASL + EMA + reconstruction consistency + report soft supervision when available.
5. A patient-grouped validation tournament. The first 500 official validation rows are marked `baseline_seen` because the queued all-data Task-1 job used them for checkpoint selection; they are excluded from posthoc development/audit partitions.
6. Your directed conditional-label idea: train-only P(A), P(A|B), and joint support create a directed graph; the graph message is gated by per-label predictive entropy. Lambda=0 is always an option.
7. Separate calibration partition and a final audit partition.
8. Optional Task-2 generate-and-rerank. The report generator stays frozen. Four beam candidates are reranked from features available at inference time: pathology agreement with the generator's own auxiliary image head, text confidence, length, repetition, and beam rank. Reference reports/labels are used only as development targets, never as hidden-test inputs.
9. Thin Docker source folders, separate weights ZIPs, metrics, an auto-generated writeup, checksums, and one final archive.

External real datasets and synthetic data are intentionally **OFF** in this unattended pipeline. They can be added later as explicit ablations without muddying the core methodological writeup or risking the ~$100 compute target.

## Launch

From the normal RunPod shell, not from inside tmux:

```bash
cd /workspace/vlm3d_original_task_split
chmod +x weeklong_champion/*.sh
./weeklong_champion/launch_weeklong.sh
```

You may disconnect from SSH after the launcher prints the tmux sessions.

## Status

```bash
cd /workspace/vlm3d_original_task_split
./weeklong_champion/status_weeklong.sh
```

## Final output

When complete:

```text
/workspace/vlm3d_WEEKLONG_CHAMPION_DELIVERABLES.tar.gz
```

The extracted archive contains approximately:

```text
vlm3d_weeklong_deliverables/
  WEEKLONG_RESULTS.md
  SHA256_MANIFEST.json
  task1_source/
  task1_weights.zip
  task2_source/
  task2_weights.zip
  task3_source_baseline/
  task3_weights_baseline.zip
```

For Task 1 and Task 2, copy the source folder and corresponding `weights.zip` to the Mac. Run `build_on_mac.sh` in the source folder, then use the current Forithmus CLI to `generate`, `test`, `validate`, and submit. Do not skip the local Forithmus smoke test even if the automated validation metrics look excellent.

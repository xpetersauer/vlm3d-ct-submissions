# Final submission workflow

## Models corresponding to the released entries

| Task | Submission | Final source | Final weights |
|---|---|---|---|
| Abnormality classification | cd1082ec | tasks/task1 | task1_weights.zip |
| Report generation | a8799c78 | tasks/task2 | task2_weights.zip, reconstructed from the two release parts |
| CT volume generation | 888a2309 | tasks/task3 | task3_weights_full_47149.zip |

The source snapshots and release bundle identities define these entries.
Later shared-encoder, posterior-fusion, diffusion and Task-3 metric-training
experiments are separate from this release.

## Task 1

1. Extract frozen DINOv2 slice features and train the three full attention heads.
2. Select the DINO ensemble and fit the stored recipe calibration.
3. Fit the additional per-label temperature, bias and decision-threshold transform.
4. At inference, execute process.py, then task1_calibrate_output.py, then the
   contract guard. This order is already present in tasks/task1/entrypoint.sh.

An intermediate startup script placed calibration after exec, making it
unreachable. The released Task-1 source uses the repaired order. The final
conditional-label graph coefficients are zero; the graph does not change the
released predictions. This entry uses the existing trained weights plus the
calibration files; a timestamp change alone does not indicate retraining.

## Task 2

1. Train the volumetric Swin–Qwen report generator with LoRA and pathology supervision.
2. Develop the first report-consistency reranker.
3. Capture diverse validation candidates and fit the metric-aligned reward model.
4. At inference, the diversity entrypoint generates candidates and process.py
   selects one report using image–text pathology agreement and the learned reward.

The final source includes the RadBERT position_ids compatibility check and its
required model dependencies. It permits that known extra key while rejecting
missing or other unexpected RadBERT keys. Generator loading itself is less
strict, as documented in the methods. Keep the complete Task-2 bundle together;
the language model alone is insufficient.

## Task 3

1. Initialize from the earlier baseline and train using the full report table.
2. Package the autoencoder and report-to-latent mapper into the full-47,149 ZIP.
3. Use batched, mixed-precision inference with atomic NIfTI saves.
4. Strip repeated MHA/NIfTI suffixes from identifiers before saving case_id.nii.gz.

The training launcher explicitly passes the full CSV row count: omitting
--max-rows in the older wrapper used its 50-row default. The released entry uses
the original full-47,149 weights, not the later metric-training candidate and
not a diffusion model. The filename repair prevents case IDs such as
case.nii.gz.mha from becoming invalid generated-output names.

## Build and run the final models

The Dockerfiles already incorporate the dependency/ABI hardening, Ninja package
repair and output guards used in the final source snapshots. Use these files
directly rather than reapplying every historical patch.

```bash
python3 scripts/download_weights.py all
docker build --platform linux/amd64 -t vlm3d-ct-task1:final tasks/task1
docker build --platform linux/amd64 -t vlm3d-ct-task2:final tasks/task2
docker build --platform linux/amd64 -t vlm3d-ct-task3:final tasks/task3
```

On a Linux NVIDIA host, put CT inputs in input/task1 or input/task2 and Task-3
prompts in input/task3/prompts.json. For a fresh Task-1 output directory:

```bash
mkdir -p input/task1 output/task1
docker run --rm --gpus all --platform linux/amd64 \
  --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$PWD/input/task1,dst=/input,readonly" \
  --mount "type=bind,src=$PWD/weights/task1,dst=/weights,readonly" \
  --mount "type=bind,src=$PWD/output/task1,dst=/output" \
  vlm3d-ct-task1:final
```

Substitute task2 or task3 throughout for the other tasks. Start with a new output
directory when changing checkpoints or prompts, particularly for resumable
Task-3 inference. USAGE.md describes the expected input and output structures.

## Official results

Task 1: CRG 0.4803, macro-AUROC 0.7814, macro-F1 0.5145.
Task 2: CRG 0.3497, macro-F1 0.1264.
Task 3: platform-reported FVD (CTNet) 0.8323, average FID 2.5D 3435.7003,
average CT-CLIPScore 10.3737. Full results are in CHALLENGE_FORM.md.

## Reproduction checks

Source syntax, workspace preparation and compatibility helpers were checked
using CPU fixtures. These checks do not reproduce the official challenge
evaluation; the official scores above come from the submitted model runs.

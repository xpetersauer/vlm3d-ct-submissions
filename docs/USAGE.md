# Installation, usage and file formats

## Requirements

The historical Dockerfiles use pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime,
Python dependencies pinned in each task's requirements.txt, and an unprivileged
appuser. Linux/amd64, Docker and an NVIDIA CUDA runtime are needed for practical
inference. A Mac can inspect/build amd64 images but is not equivalent to the
CUDA evaluation environment. Network access may be required at BUILD time;
inference is intended to use locally packaged model assets offline.

These are example local reproduction commands based on the collected mounts
and entrypoints, not newly GPU-tested commands. Original files are preserved,
including bugs; do not silently repair a historical submission before comparison.

## Prepare weights

Keep downloaded weight ZIPs outside normal Git. Check their SHA-256 with:

```bash
python3 scripts/verify_weights.py --task task1 --file /absolute/path/task1_weights.zip
python3 scripts/verify_weights.py --task task2 --file /absolute/path/task2_weights.zip
python3 scripts/verify_weights.py --task task3 --file /absolute/path/task3_weights_full_47149.zip
```

Extract each verified, trusted bundle into a NEW dedicated weights directory.
The archive contents must be at that directory's root, not nested inside an
additional wrapper directory. Only load checkpoints/joblib files from trusted
sources. Binary checkpoint loading was not performed in this audit.

## Build one image (Linux NVIDIA host)

From the repository root, choose task1, task2 or task3:

```bash
docker build --platform linux/amd64 -t vlm3d-ct-task1:historical tasks/task1
docker build --platform linux/amd64 -t vlm3d-ct-task2:historical tasks/task2
docker build --platform linux/amd64 -t vlm3d-ct-task3:historical tasks/task3
```

## Example inference

Replace the three /absolute/path values with real existing directories. Output
must be writable by the selected container user. Below task1 is shown; change
the image tag and mount corresponding inputs/weights for task2 or task3.

```bash
docker run --rm --gpus all --platform linux/amd64 \
  --user "$(id -u):$(id -g)" \
  --mount type=bind,src=/absolute/path/input,dst=/input,readonly \
  --mount type=bind,src=/absolute/path/extracted_weights,dst=/weights,readonly \
  --mount type=bind,src=/absolute/path/output,dst=/output \
  vlm3d-ct-task1:historical
```

Do not put reference reports, labels or hidden-test files in inference mounts
unless explicitly required by the official task input schema. The task1/2
classifier/report paths discover .nii and .nii.gz recursively under /input.
Task3 expects report prompts, not CT images. Validate against the organizer's
actual schema, not merely the conceptual examples below.

## Task 1

Input: NIfTI CT volumes under /input; suffix-stripped filename becomes case ID.
Output: /output/results.json with metadata and predictions list. Each item has
input_image_name and probabilities mapping all 18 labels to finite [0,1] scores.
The exact label order is in configs/task1/final_recipe.json.

Synthetic schema illustration (not a full valid 18-label output):

```json
{"predictions": [{"input_image_name": "synthetic_case", "probabilities": {"Cardiomegaly": 0.25}}]}
```

## Task 2

Input: NIfTI CT volumes under /input.
Output: /output/predictions.json containing a list, with outputs/value nesting
and generated_reports entries with input_image_name and report. The source
process.py constructs the precise metadata wrapper. The contract guard may
further normalize/check outputs; preserve it when reproducing the container.

```json
[{"outputs": [{"value": {"name": "Generated reports", "type": "Report generation", "version": "1.0", "generated_reports": [{"input_image_name": "synthetic_case", "report": "Synthetic example only."}]}}]}]
```

## Task 3

Input: /input/prompts.json. One supported synthetic prompt format is:

```json
{"prompts": [{"input_image_name": "synthetic_case.mha", "report": "FINDINGS: Synthetic example only. IMPRESSION: Synthetic example only."}]}
```

Output: /output/synthetic_case.nii.gz, with recurrent suffixes removed from IDs.
Data are rounded HU-valued int16, identity affine; shape follows checkpoint
metadata (source default 64x96x96). Inference defaults: batch16, save_workers4,
AMP enabled and resume enabled. Override with TASK3_BATCH_SIZE,
TASK3_SAVE_WORKERS, TASK3_AMP, TASK3_RESUME. CUDA OOM halves batch size.
Do not treat synthetic prompts as performance-validation data.

## Training/evaluation settings

Task-1/2 submitted inference source does not include the exact historical
head/generator trainer invocation and optimizer schedules. Task-3 exposes
training functions, but defaults do not certify actual full-run settings.
Do not launch new training to reconstruct history without a separate plan.
Validation counts, known internal/official scores and unresolved provenance are
reported in CHALLENGE_FORM.md; no missing number has been invented.

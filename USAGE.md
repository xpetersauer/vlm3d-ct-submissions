# Installation and inference

## Requirements

The Dockerfiles use pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime,
Python dependencies pinned in each task's requirements.txt, and an unprivileged
appuser. Linux/amd64, Docker and an NVIDIA CUDA runtime are needed for practical
inference. A Mac can inspect/build amd64 images but is not equivalent to the
CUDA evaluation environment. Network access may be required at BUILD time;
inference is intended to use locally packaged model assets offline.

## Download and extract weights

```bash
python3 scripts/download_weights.py task1
python3 scripts/download_weights.py task2
python3 scripts/download_weights.py task3
```

Each command verifies the original ZIP checksum and extracts the complete bundle
under weights/task1, weights/task2 or weights/task3. Task 2's release parts are
joined automatically. The source/configuration paths are tasks/taskN and
configs/taskN for all three tasks. Only load trusted checkpoints.

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

## Training and evaluation

Preprocessing, objectives, available validation results and limitations are
described in [CHALLENGE_FORM.md](CHALLENGE_FORM.md). Task-specific configuration
files are under configs/. Saved checkpoint metadata controls model dimensions.

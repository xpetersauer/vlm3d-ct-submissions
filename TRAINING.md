# Training the submitted model architectures

The original DINO head trainer, feature extractor, selection code, report
generator trainer and Task-3 training/resume patches are included under
training/original/. Their original contents are preserved. The stage runner
creates isolated working copies and writes new training outputs separately from
the released submission weights.

Task 1 uses the recovered weeklong trainer. Task 2 includes the recovered early
trainer and a clearly identified compatibility adapter for its later all-data
command. Task 3 uses the recovered two-stage training and resume patches. Fresh
training does not imply byte-identical reconstruction of the released checkpoints.

## Environment

Run training on a Linux NVIDIA CUDA host, not on the Mac. From this repository:

```bash
python3 -m venv /workspace/vlm3d_repro_env
source /workspace/vlm3d_repro_env/bin/activate
python -m pip install -r tasks/task2/requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The requirements specify PyTorch 2.4.1 and the model dependencies. Dataset access
requires authorization for ibrahimhamamci/CT-RATE and an HF_TOKEN environment
variable. Set the token privately in the pod environment; do not commit it.

## Download the four metadata tables

```bash
python - <<'PY'
import os
from huggingface_hub import hf_hub_download
for name in [
    "dataset/radiology_text_reports/train_reports.csv",
    "dataset/radiology_text_reports/validation_reports.csv",
    "dataset/multi_abnormality_labels/train_predicted_labels.csv",
    "dataset/multi_abnormality_labels/valid_predicted_labels.csv",
]:
    hf_hub_download(repo_id="ibrahimhamamci/CT-RATE", repo_type="dataset",
                    filename=name, local_dir="/workspace/CT-RATE",
                    token=os.environ["HF_TOKEN"])
PY
```

## Prepare a new workspace

```bash
python training/prepare_workspace.py \
  --output /workspace/vlm3d_reproduction \
  --data-root /workspace/CT-RATE
```

To reuse the old compact tensor cache, add:

```text
--tensor-cache /workspace/vlm3d_tensor_cache
```

The preparation command refuses to overwrite an existing experiment. It copies
the final model definitions into a working project and applies compatibility
fixes only to those working copies. Submitted inference source is unchanged.
Use --dry-run on the stage runner to print commands without starting computation.

## Task 1: DINOv2 features, attention heads and selection

```bash
python training/run_stage.py --workspace /workspace/vlm3d_reproduction radbert
python training/run_stage.py --workspace /workspace/vlm3d_reproduction task1-features
python training/run_stage.py --workspace /workspace/vlm3d_reproduction task1-heads
python training/run_stage.py --workspace /workspace/vlm3d_reproduction task1-select
```

The feature extractor freezes DINOv2-base and computes 72 tokens per volume:
24 slices under each of three intensity windows. Feature batches contain up to
96 images. Features are stored in fp16 memory maps; row order and completion
masks are checked when extraction resumes.

The final full-head recipe launches seeds 17, 29 and 43, each with a maximum of
15 epochs and early-stopping patience 4. The trainer uses batch size 128, AdamW
with learning rate 1e-3 and weight decay 0.01, cosine decay to 5% of the starting
rate, gradient clipping at 1.0 and EMA decay 0.999. ASL uses negative/positive
focusing parameters 4/1 and clipping 0.05. If aligned report soft targets exist,
their BCE term receives weight 0.20. Same-scan reconstruction consistency has
weight 0.10 and is evaluated every fourth optimization step when a paired
reconstruction exists. The stage runner prints whether report targets are present.

Each head is selected by model-development AUROC. posthoc_select.py compares
available candidates, averages full-head logits for the ensemble, and fits the
conditional graph and initial calibration. The released final recipe selected
the DINO ensemble and has zero graph-mixing coefficients. Outputs are under
/workspace/vlm3d_reproduction/work/dino_full_s17, dino_full_s29, dino_full_s43
and posthoc. The recovered original queue also contains the BCE/ASL and Swin
comparison stages.

## Task 2: Swin–Qwen generator

The August terminal log records a five-epoch all-data generator run with batch
size 1, learning rate 1e-4, vision learning rate 1e-5, warmup ratio 0.05, 250
validation cases, 256 target tokens, 32 visual tokens and pathology-loss weight
0.25. The original initialization path was checkpoints/task1_DEADLINE_5k.pt.
It must be a compatible Swin checkpoint with encoder.* keys, not the released
DINO head bundle.

The archived early trainer does not accept the later --val-reports-csv and
--val-labels-csv flags visible in that log. train_task2_full_data.py is a new
compatibility adapter: it adds those flags, removes the early 10,000-row cap,
uses the compact tensor cache and keeps the original optimization loop. It is
not presented as the missing historical patch.

With the original initialization checkpoint available:

```bash
python training/run_stage.py \
  --workspace /workspace/vlm3d_reproduction \
  --task1-checkpoint /workspace/original_checkpoints/task1_DEADLINE_5k.pt \
  task2-train
```

If that checkpoint is unavailable, this command trains a NEW baseline Swin
initialization using 5,000 training cases. This enables a fresh training run but
does not reproduce the historical initialization:

```bash
python training/run_stage.py --workspace /workspace/vlm3d_reproduction baseline-swin
python training/run_stage.py \
  --workspace /workspace/vlm3d_reproduction \
  --task1-checkpoint /workspace/vlm3d_reproduction/work/new_baseline_swin.pt \
  task2-train
```

The original trainer uses AdamW, weight decay 0.01, cosine scheduling with
warmup, clipping at 1.0 and minimum validation loss for checkpoint selection.
The visual encoder is frozen unless --unfreeze-vision is explicitly enabled.
The stage runner follows the recorded frozen-vision command.

The original first-stage report selector can then be fitted with:

```bash
python training/run_stage.py --workspace /workspace/vlm3d_reproduction task2-rerank
```

The final metric-aligned selector and Task-1 calibration use the next section.

## Final calibration and reward fitting

The fitting code and validation-capture pipeline are preserved under
training/original/metric_alignment/. If the original validation CSV/JSONL files
are available, run:

```bash
python training/original/metric_alignment/metric_alignment.py task1-fit \
  --input /workspace/validation/task1_validation.csv \
  --output /workspace/reproduced_calibration/task1

python training/original/metric_alignment/metric_alignment.py task2-fit \
  --input /workspace/validation/task2_capture.jsonl \
  --output /workspace/reproduced_calibration/task2 \
  --minimum-unique-ratio 0.75
```

Task-1 tables require paired true_<label>/prob_<label> columns and case/group
identifiers, or an NPZ with y_true, probabilities, labels and groups.
Task-2 captures contain reference reports, candidate reports and inference-time
features. They are local development artifacts and must not be published with
patient information. The recovered run_task12_metric_alignment.sh documents
how validation volumes were staged, sharded, predicted and captured before
fitting. For inference with the existing release, use its stored calibrator and
reranker; no fitting is required.

## Task 3: autoencoder and report mapper

Copy the retained baseline ZIP to the GPU host. The Mac path was
~/Downloads/vlm3d_final/vlm3d_weeklong_deliverables/task3_weights_baseline.zip.
Then run:

```bash
python training/run_stage.py \
  --workspace /workspace/vlm3d_reproduction \
  --initial-task3-zip /workspace/task3_weights_baseline.zip \
  task3-train
```

This initializes from that baseline, explicitly passes the complete training
row count, and trains five autoencoder epochs followed by five mapper epochs.
The recipe uses AdamW, learning rate 1e-4 and batch size 1. The autoencoder uses
image L1; the mapper uses latent MSE plus image L1 weighted 0.25. The autoencoder
is frozen during mapper training, and DistilBERT is frozen by default.
Step/epoch progress and optimizer state are saved by the recovered resume patch.
The new output ZIP is work/task3_full_training/task3_weights_retrained.zip.
Repeating the command resumes from saved state; it does not overwrite released
weights. The original recorded launcher and status script are retained under
training/original/task3_runpod/.

## Inspect metadata in the final weights

These commands read trusted checkpoint metadata on CPU without training:

```bash
python training/inspect_final_metadata.py /path/to/task1_weights.zip --task task1
python training/inspect_final_metadata.py /path/to/task2_weights.zip --task task2
python training/inspect_final_metadata.py /path/to/task3_weights_full_47149.zip --task task3
```

Task 1 exposes seed, selected epoch, loss variant and whether report supervision
was active. Task 2 may expose epoch, learning-state fields and model parameters.
Task 3's resumable checkpoints may expose stage, completed epochs and dataset
size. Task 2 loading requires enough CPU RAM for its large checkpoint.

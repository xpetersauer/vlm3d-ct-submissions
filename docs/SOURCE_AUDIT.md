# Source audit and corrections

## Evidence scope

The collector output VLM3D_CT_Evidence_20261027_143743_021c01.zip supplies source,
small JSON configs, ZIP member inventories, SHA-256 hashes and packaging records.
Its EVIDENCE_INDEX reports Missing 1, Skipped 0, Errors 0. The sole missing item
is task3_baseline_fixed_archive, corresponding to
task3_submission_initial_hardened_v2.tar.gz. The retained full47149 archive and
both Task-3 weight variants are present in its hash inventory.

Binary weights and Docker archives were not uploaded. This audit does not
deserialize PyTorch/joblib checkpoints, extract a filesystem from a Docker image,
run CUDA inference, or independently reproduce official scores.

## Task 1: code path versus weight changes

All three collected weight variants have SHA-256
5f252b1b2da96516303762fde1ec28af428849edb83afb7150435562b71d2d27.
Earlier concern that different modification times implied different contents is
resolved: the copies are byte-identical. An overwrite can still copy identical bytes.

tasks/task1/entrypoint.sh runs process.py, then task1_calibrate_output.py, then
execs the contract guard. tasks/task1-later-hardened/entrypoint.sh execs the
guard before the calibration command, so that command is unreachable in a
successful shell execution. Both source variants otherwise produce the base
recipe predictions. This supports the reported execution-order explanation;
it does not prove a particular official score without the matching platform run.

The recipe selects dino_full_ensemble, contains three head filenames and no
resolved Swin checkpoint. All graph.lambda_by_label entries are zero. Graph
correction is therefore a no-op despite graph.accepted=true. Do not describe
the submitted classifier as a learned effective graph-fusion system.

Base recipe calibration is sigmoid(a * logit(p) + b). The additional output
transform is sigmoid(logit(p) / temperature + bias - logit(threshold)). The
composition is monotone for each label. This does not imply independently
validated Bayesian posterior probabilities or improved AUROC ranking.

## Task 2: actual LoRA and decoding behavior

process.py constructs CTCausalReportGenerator with freeze_vision=true and
use_lora=true. sota_models.py specifies Qwen attention LoRA on q/k/v/o projections,
default rank16/alpha32/dropout0.05, and language loss + 0.25 pathology BCE.
Qwen model files, RadBERT files, classifier weights and reranker files occur in
the weight inventory. The exact historical optimizer/epoch settings are absent.

The wrapper patches Transformers GenerationMixin.generate and advances a
thread-local global call index. process.py makes one generation call per CT and
requests four return sequences. Therefore the code cycles policies across CT
generation calls, not one fixed policy per candidate within every CT. When the
nominal greedy policy sees four requested outputs, it raises num_beams to four;
the first two policies may thus both use four-beam decoding. Do not describe
every candidate set as containing greedy + beam + multiple nucleus arms.

RadBERT-derived probabilities are compared with the image pathology head to
produce six candidate features. The source uses the highest ridge-predicted
reward. This is an inference-available image/report consistency mechanism;
it is not the later hierarchical Bayesian fusion proposal.

## Task 3: original full-training checkpoint, not later reweighting

Participant terminal history links task3_weights_full_47149.zip to
task3_full_training_47149/task3_weights_full.zip. Do not replace it with the
task3_metric_training_47149 checkpoint. The recovered model has a four-stage
convolutional autoencoder and DistilBERT-to-latent mapper, not diffusion.
Docker explicitly predownloads distilbert-base-uncased. The autoencoder and
mapper binary metadata control actual dimensions; source defaults are not an
independent read of those metadata. Source training functions use image L1 and
latent MSE + weighted image L1; whether the historical full trainer used every
default cannot be established from inference source alone.

Output is int16 HU NIfTI with identity affine, produced in batches with atomic
saving and resume logic. Suffix cleaning is recursive. No CFG, spacing
conditioning, diffusion sampling, candidate tournament or exact-FVD model
selection is established for the final reported artifact.

## Validation limitations

Task-1 recipe partitions and the later calibration partition are both drawn
from the 3,039-case validation pool. The latter's held-out rows are independent
of its parameter fitting only; they are not proven untouched by all earlier
model/recipe selection. No membership-level leakage audit can be performed from
aggregate counts. Task-2 excludes 250 generator-selection cases according to its
config, but full split membership is absent. These caveats belong in the paper.

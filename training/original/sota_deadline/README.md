# VLM3D deadline-oriented modernized models

These are **experimental training replacements**, not drop-in challenge submission wrappers yet. Keep your working baseline/Docker path untouched until these beat it.

## Task 1

`train_sota_task1.py` replaces global-average-pool -> Linear(18) with:

```text
CT
 -> Swin visual tokens
 -> 18 learned label tokens cross-attend to image tokens
 -> label-token self-attention
 -> 18 logits
```

The labels are *not* assumed independent: label tokens explicitly interact. `sota_models.py` also supports an optional directed label-graph mixer, which is the natural deep-learning extension of the post-hoc hierarchy experiment.

Training adds:
- patient-group-disjoint internal validation
- `best.pt`, `last.pt`, and periodic step checkpoints
- AdamW
- 5% warmup + cosine decay
- gradient clipping
- separate LR for visual backbone vs label head

Example deadline run:

```bash
python train_sota_task1.py \
  --labels-csv ../CT-RATE/dataset/multi_abnormality_labels/train_predicted_labels.csv \
  --out ./checkpoints/sota_task1_10k \
  --max-rows 10000 \
  --epochs 1
```

## Task 2

`train_sota_task2.py` uses:

```text
CT
 -> Swin visual tokens
 -> Perceiver-style resampler (32 visual tokens)
 -> small decoder-only instruction LM
 -> LoRA
 -> full FINDINGS + IMPRESSION report

plus:
resampled visual tokens -> 18-label auxiliary pathology head
```

Default causal LM is `Qwen/Qwen2.5-0.5B-Instruct`, chosen because a 0.5B decoder is more deadline-realistic than 7B while still testing the decoder-only + LoRA design.

The auxiliary pathology head is *not required* for the CPU post-hoc fusion test. It is useful for a future end-to-end version because it forces visual tokens to preserve pathology information and provides native `p_text/visual-clinical` evidence.

Safe PEFT install on the existing RunPod environment:

```bash
python -m pip install --no-deps peft==0.13.2
```

Do not reinstall Torch.

Example:

```bash
python train_sota_task2.py \
  --reports-csv ../CT-RATE/dataset/radiology_text_reports/train_reports_complete.csv \
  --labels-csv ../CT-RATE/dataset/multi_abnormality_labels/train_predicted_labels.csv \
  --out ./checkpoints/sota_task2_10k \
  --max-rows 10000 \
  --epochs 1
```

Optionally transfer the Swin encoder from a compatible Task-1 checkpoint:

```bash
... --task1-checkpoint /path/to/task1_checkpoint.pt
```

## Task 3

For the 3-day deadline, keep the current Task-3 architecture and add best/last/step checkpointing rather than replacing it with latent diffusion. The research-grade direction is a report-conditioned 3D latent diffusion/flow model with stronger medical text conditioning, but that is a much higher-risk rewrite and train.

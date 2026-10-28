# VLM3D CT-RATE submissions

Code, configurations and trained weights for Peter Sauer's three CT challenge submissions.

## Download the model weights

**[Download weights from GitHub Releases](https://github.com/xpetersauer/vlm3d-ct-submissions/releases/tag/weights-v1)**

Open **Assets** and download the files for your task:

| Task | Submission | Download |
|---|---|---|
| 1 — Abnormality classification | cd1082ec | [task1_weights.zip](https://github.com/xpetersauer/vlm3d-ct-submissions/releases/download/weights-v1/task1_weights.zip) |
| 2 — Report generation | a8799c78 | [Part 000](https://github.com/xpetersauer/vlm3d-ct-submissions/releases/download/weights-v1/task2_weights.zip.part000) + [Part 001](https://github.com/xpetersauer/vlm3d-ct-submissions/releases/download/weights-v1/task2_weights.zip.part001) |
| 3 — CT volume generation | 888a2309 | [task3_weights_full_47149.zip](https://github.com/xpetersauer/vlm3d-ct-submissions/releases/download/weights-v1/task3_weights_full_47149.zip) |

Task 1 and Task 3: download and unzip the corresponding file.

Task 2: download **both** parts into the same folder, then run:

```bash
cat task2_weights.zip.part000 task2_weights.zip.part001 > task2_weights.zip
unzip task2_weights.zip
```

### Automatic download

From the repository root, run one command for the task you want:

```bash
python3 scripts/download_weights.py task1
python3 scripts/download_weights.py task2
python3 scripts/download_weights.py task3
```

Or download all three:

```bash
python3 scripts/download_weights.py all
```

The script downloads, joins Task 2's parts, checks the ZIP's SHA-256 and extracts
to weights/task1, weights/task2 or weights/task3. Python 3 is the only download
dependency. Keep the complete extracted bundle together. Cloning Git alone does
not download the weights.

## Models

| Task | Method | Main checkpoint files |
|---|---|---|
| 1 | DINOv2-base slice encoder, three pathology-query heads, calibrated ensemble | dino_head_0.pt, dino_head_1.pt, dino_head_2.pt |
| 2 | Volumetric Swin encoder, Qwen2.5-0.5B LoRA, pathology-consistency reranking | task2_best.pt, RadBertClassifier.pth, reranker.joblib |
| 3 | DistilBERT report mapper and a single-latent 3D autoencoder | task3_autoencoder.pt, task3_report_to_latent/mapper.pt |

## Documentation

- [Methodology, validation and limitations for each task](CHALLENGE_FORM.md)
- [Installation, input/output formats and inference](USAGE.md)
- [References](REFERENCES.md)
- [Usage terms](TERMS.md)

Source code is in tasks/task1, tasks/task2 and tasks/task3. Configurations and
weight filenames for **all three tasks** are in configs/task1, configs/task2
and configs/task3. Task 3's runtime settings are in configs/task3/inference.json
and configs/task3/runtime.env. The large weight files are release assets,
separate from the configuration files.

## Team and data

Peter Sauer — UCLA–Caltech Medical Scientist Training Program,
David Geffen School of Medicine at UCLA.

Training and development used challenge-provided CT-RATE data and pretrained
models. Obtain the data through the [CT-RATE dataset page](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE).
Official task schemas and evaluation code are available in
[VLM3D-Dockers](https://github.com/forithmus/VLM3D-Dockers).

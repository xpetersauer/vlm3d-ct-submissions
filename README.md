# VLM3D CT-RATE — documented submission snapshots

For copy/paste GitHub source and weight-upload commands, open
[UPLOAD_TO_GITHUB.md](UPLOAD_TO_GITHUB.md). Normal Git contains source/configs;
the exact weight bundles are uploaded separately as release assets.

**Participant:** Peter Sauer. **Draft affiliation:** UCLA–Caltech Medical Scientist
Training Program, David Geffen School of Medicine at UCLA. Confirm final team
membership, official affiliation wording and challenge team name before publication.

This repository assembles the source and small configuration files recovered
from the participant's Mac. It documents three reported submissions and preserves
one later classification variant. It contains **no binary neural weights or
Docker archives**. Authorized public weight hosting is still required for a fully
reproducible public release. Inference was not rerun during this documentation pass.

## Copy/paste challenge responses

[CHALLENGE_FORM.md](CHALLENGE_FORM.md) contains a 200–300-word methodology for
each task, checklist selections, data sources, validation, limitations and weight
availability text. It supersedes the earlier speculative drafts. Exact original
training schedules and two official score mappings remain explicitly unresolved.

| Task / displayed ID | Method | Source | Weight bundle |
|---|---|---|---|
| 1 / cd1082ec | DINOv2-base + three query-pooling heads + repaired output calibration | tasks/task1 | task1_weights.zip |
| 2 / a8799c78 | SwinUNETR-derived visual tokens + Qwen2.5-0.5B LoRA + RadBERT/ridge reranking | tasks/task2 | task2_weights.zip |
| 3 / 888a2309 | DistilBERT report mapper + single-latent 3D autoencoder | tasks/task3 | task3_weights_full_47149.zip |
| Later Task-1 hardened / ID not supplied | Same weights, unreachable additional output calibration after exec | tasks/task1-later-hardened | Same Task-1 bundle |

IDs are displayed prefixes, not independently verified full platform UUIDs.
The correspondence is based on participant descriptions and terminal history.
Source snapshots were not independently extracted from the uploaded Docker images.

## Required documentation map

- Methods, pretrained components, data sources: [challenge responses](CHALLENGE_FORM.md).
- Team/affiliation: above; other members and preferred wording require confirmation.
- Access, preprocessing, training/evaluation: [challenge responses](CHALLENGE_FORM.md).
- Installation, I/O and inference commands: [USAGE.md](docs/USAGE.md).
- Per-submission configs: configs/task1 and configs/task2; Task-3 shape/latent
  metadata reside in its binary checkpoints, absent from this small-file collection.
- Final filenames, SHA-256 and bundle inventories: provenance/ARTIFACTS.json.
- Results and limitations: [challenge responses](CHALLENGE_FORM.md).
- Source audit and evidence limits: [SOURCE_AUDIT.md](docs/SOURCE_AUDIT.md).
- References and usage terms: [REFERENCES.md](docs/REFERENCES.md), [USAGE_TERMS.md](docs/USAGE_TERMS.md).

## Data access

Obtain CT-RATE under its authorized access and license procedures at
https://huggingface.co/datasets/ibrahimhamamci/CT-RATE . The challenge's reference
contracts/evaluators are at https://github.com/forithmus/VLM3D-Dockers .
Do not commit scans, patient reports, validation membership with identifying
information, hidden evaluation data or credentials. Pretraining weights are not
evidence that an additional dataset was directly used by this team.

## What the audit establishes

The uploaded collection reports one missing file: the old Task-3 baseline-fixed
Docker archive. Retained final archives, source snapshots, weight hashes and
configs were collected without reported errors. All three collected Task-1 ZIPs
share the same SHA-256, despite different modification times. The repaired
Task-1 entrypoint runs output calibration before the contract guard; the later
entrypoint places it after exec, making it unreachable. The final label-graph
coefficients are all zero. Task 2 actually uses LoRA. These facts replace earlier
provisional descriptions. They do not prove source-to-image binary equivalence.

## Remaining release actions

1. Confirm score and full ID for cd1082ec and 888a2309; Task-2 official screenshot
   identifies CRG 0.3497 and macro-F1 0.1264 for a8799c78.
2. Publish authorized exact weight bundles, add versioned download links, and
   document access restrictions if redistribution is not permitted.
3. Confirm team and review component licenses. No blanket license is imposed here.
4. Recover missing historical training settings/logs if required; do not replace
   them with CLI defaults. Add further valid leaderboard runs if the organizers
   require documentation of every run rather than the selected methods.
5. Run the exact model's local contract test before claiming a reproduced result.

AI assisted development and retrospective documentation. Human review and
appropriate venue disclosure remain necessary. No posterior-fusion, diffusion,
synthetic-augmentation or new shared-CV results are attributed to these submissions.

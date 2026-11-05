# Public methods/assets used or adapted

The unattended queue intentionally uses CT-RATE plus public pretrained models only. It does not ingest another patient dataset or synthetic CT dataset.

- **CT-RATE / CT-CLIP** — dataset, compact-cache source volumes/reports/labels, and the public RadBERT text-classifier checkpoint. Official repository: https://github.com/ibrahimethemhamamci/CT-CLIP ; dataset: https://huggingface.co/datasets/ibrahimhamamci/CT-RATE . CT-RATE naming encodes split, patient, scan, and reconstruction, enabling same-scan reconstruction pairing.
- **DINOv2** — frozen public 2D visual foundation model used for the cost-controlled slice-feature branch: https://github.com/facebookresearch/dinov2 and https://huggingface.co/facebook/dinov2-base .
- **AnyMC3D** — methodological inspiration for treating a 3D study as slice-wise 2D foundation-model features followed by lightweight task-specific aggregation: https://arxiv.org/abs/2512.12887 . The queued implementation is explicitly not an exact AnyMC3D reproduction because it freezes DINOv2 and works from the existing 64x96x96 compact cache rather than performing full-resolution LoRA adaptation.
- **Asymmetric Loss (ASL)** — Ridnik et al., ICCV 2021: https://arxiv.org/abs/2009.14119 .
- **Forithmus VLM3D canonical Docker examples/evaluators** — https://github.com/forithmus/VLM3D-Dockers . Final Docker/weights artifacts should be tested against the current CLI and phase schema before submission.

The uncertainty-gated directed conditional graph, same-scan reconstruction-consistency combination, report-soft-target combination, validation-partition safeguards, and conditional model/reranker acceptance logic in this bundle are project-specific experimental components.

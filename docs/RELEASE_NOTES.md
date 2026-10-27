# Exact retained CT submission weights

This release supplies the retained weight archives documented in this
repository. Source snapshots/configurations and limitations are in the repository.
Do not treat displayed submission ID prefixes as independently verified full IDs.

- Task1 cd1082ec: task1_weights.zip; DINOv2 query-pooling heads and calibration.
- Task2 a8799c78: task2_weights.zip, split into numbered parts to fit release limits;
  Swin/Qwen LoRA generator, RadBERT classifier and learned reranker.
- Task3 888a2309: task3_weights_full_47149.zip; full-training latent-regression model.

Download both Task2 parts and concatenate part000 followed by part001 to restore
task2_weights.zip. Verify SHA-256 against the repository's provenance manifest
before extraction. SHA256SUMS.txt lists the uploaded asset checksums, including
the individual Task2 parts. No Docker archives or patient data are required here.

Component licenses and data access terms apply independently; no new blanket
license over upstream model weights is granted by this release text. The uploader
must confirm redistribution permission. Intended use is research/benchmarking;
no clinical safety, independent GPU rerun or prospective validation is claimed.

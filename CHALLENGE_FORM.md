# Challenge form responses

Prepared from the uploaded Mac source/configuration evidence on 2026-10-27.
The IDs below are displayed prefixes supplied by the participant, not verified
full UUIDs. Source inspection establishes behavior of the collected snapshots;
it is not an independent extraction from the platform Docker artifacts.
The existing weights remain on the participant's Mac; public download URLs have
not been supplied. Keep those availability statements accurate when submitting.

## Task 1

**Method name:** Multi-window DINOv2 Query-Pooling Ensemble with Decision Calibration

**Leaderboard name / submission ID:** cd1082ec — Task1 DINO ensemble calibration-order repaired

**Challenge task:** CT Abnormality Classification

### Methodology

Our classifier represents a CT examination through multi-window two-dimensional slice features and predicts 18 abnormalities using a three-head attention ensemble. CT intensities are sanitized, clipped to −1000 to 1000 HU, normalized, and resized to a 64 × 96 × 96 tensor. Uniformly sampled planes along the first tensor axis are rendered using three intensity windows: −1000 to 1000 HU, −150 to 250 HU, and −1000 to 400 HU. The resulting images are resized to 224 × 224, replicated across three channels, and normalized with the packaged image processor.

A pretrained DINOv2-base backbone produces a 768-dimensional classification-token feature for each image. Each task-specific head projects the feature sequence to 384 dimensions, adds learned positional embeddings and a residual adapter, and uses 18 learned pathology queries with eight-head cross-attention. Three saved heads are evaluated, their logits averaged, and a sigmoid applied. Head checkpoints determine the slice count; inference defaults to 24. The selected recipe specifies the DINO ensemble rather than the available Swin alternative. Although the recipe contains a label-cooccurrence graph, its per-label mixing coefficients are all zero, so graph correction has no effect.

Inference first applies the recipe's per-label affine logit calibration. The repaired entrypoint then applies an additional held-out temperature, bias, and threshold shift before output validation. This corrects an execution-order bug: in the later hardened snapshot, the additional calibration command follows an exec statement and is unreachable. All collected Task-1 weight copies have identical SHA-256 hashes, supporting a code-path difference rather than changed weights. Original head-training optimizer and epoch settings were not recovered. This system does not implement Bayesian image–report posterior fusion.

### Methodological Approaches

- Select: **2D / slice-based model**; **Ensemble**; **Other approach**.
- Other approach text: Multi-window slice encoding, pathology-query attention pooling, and post-hoc decision calibration.
- Do not select a fully 3D vision model, retrieval, or vision–language fusion for this final DINO path. The backbone processes 2D images; volume-level aggregation is learned.

### Data Sources

- Select: **Challenge-provided data**.
- Pretrained DINOv2 weights were used. Do not select “Pretrained model only; no additional data,” because task-specific fitting used CT-RATE.
- No separately collected public, private/institutional, or synthetic CT training dataset is established by these artifacts.

### Validation and Evaluation

The selected final recipe records 3,039 validation cases partitioned as baseline_seen 500, modeldev 1,037, selection 497, graph 366, calibration 348, and audit 291. Recipe-level metrics are macro-AUROC 0.811029 and macro-F1 0.276196 on the audit subset, and macro-AUROC 0.810752 and macro-F1 0.285727 on all 3,039 validation cases; these are internal recipe metrics before the additional output transform. The separate metric-calibration record contains 1,828 fitting rows and 1,211 held-out rows. It reports held-out macro-F1 0.289035 before and 0.523334 after calibration, with AUROC 0.814864. The development code uses grouped splitting; membership files were not included, so patient isolation and independence from earlier pipeline selection cannot be independently verified. The platform list shows a classification score of 0.4803, but its association with cd1082ec was not explicitly supplied. Confirm that page before assigning the number to this ID. No internal metric is represented as an official hidden-test result.

### Limits and Limitations

Slice sampling and coarse volume resizing may miss small lesions and lose spatial or physical-spacing information. The preprocessing does not establish anatomical reorientation or spacing harmonization; a first-axis slice is not necessarily anatomically axial. Malformed or unreadable inputs may be replaced with an air-valued placeholder by the hardened loader, yielding a plausible-looking score without valid image evidence. Report-derived labels are noisy. Threshold shifting optimizes decisions and is not proof of clinical posterior calibration. Validation was reused in several stages; a holdout for the additional calibrator is not automatically untouched by the entire development pipeline. Demographic, scanner, external-site and prospective performance are untested. The label graph is inactive in the final recipe.

### Model Weight Availability

**Current selection:** Not available / access restricted (until an authorized public URL exists).

Submission cd1082ec references **task1_weights.zip**, SHA-256 **5f252b1b2da96516303762fde1ec28af428849edb83afb7150435562b71d2d27**. Required files include dino_head_0.pt, dino_head_1.pt, dino_head_2.pt, dinov2-base model/processor assets, final_recipe.json and task1_metric_calibration.json. Current, pre-overwrite and pre-repair copies are byte-identical. Public URL: not yet supplied. If hosted later, select Hugging Face or GitHub and insert the actual versioned URL; do not claim availability before upload.

## Task 2

**Method name:** Swin–Qwen LoRA Report Generation with Image–Text Consistency Reranking

**Leaderboard name / submission ID:** a8799c78 — Task2 diverse candidates held-out reward reranker hardened

**Challenge task:** CT Report Generation

### Methodology

Our report-generation system combines a volumetric SwinUNETR-derived encoder, a Perceiver-style visual resampler, and Qwen2.5-0.5B-Instruct. CT inputs are sanitized, clipped to −1000 to 1000 HU, normalized, and resized to 64 × 96 × 96. The deepest visual features are adaptively pooled to a checkpoint-defined grid, projected into the language-model embedding space, and compressed into a fixed visual prefix. The source defaults are a 4 × 6 × 6 grid and 32 resampled tokens. An auxiliary head predicts 18 abnormality probabilities from these visual tokens.

The generator uses LoRA adapters on the language model's query, key, value, and output attention projections, with source defaults of rank 16, scaling 32, and dropout 0.05. Its training objective combines autoregressive report loss with auxiliary pathology binary cross-entropy weighted by 0.25. The visual encoder is frozen by the submitted model constructor. The exact historical optimizer schedule and epoch count are not established by the collected inference files.

At inference, four candidate reports are requested, with a 220-token generation limit. A wrapper cycles greedy, beam, and nucleus policies across consecutive generation calls; it does not explicitly produce one candidate from each policy for every scan. A packaged RadBERT-RoBERTa-4m classifier estimates abnormality probabilities from each candidate. Six features capture agreement with the visual pathology head, text confidence, length, repetition, and candidate rank. A regularized ridge reward predictor selects the highest-scoring report. Reference-derived clinical and lexical rewards supervise development only; reference reports are not inference inputs. This is image–text consistency reranking, not the later hierarchical Bayesian posterior-fusion proposal.

### Methodological Approaches

- Select: **3D vision model**; **Vision-language model**; **Language model**; **Generative model**; **Parameter-efficient fine-tuning**; **Other approach**.
- Other approach text: Auxiliary pathology supervision and RadBERT-based image–report consistency reranking with reference-reward distillation.
- Do not select retrieval. Multiple candidate reports do not by themselves constitute a model ensemble.

### Data Sources

- Select: **Challenge-provided data**.
- The method also uses pretrained Qwen and RadBERT assets. This is not “Pretrained model only; no additional data.”
- Generated candidate reports and computed rewards are development intermediates, not evidence of synthetic-image augmentation. Select “Synthetic data / generated labels” only if the organizer's definition includes these computed supervision targets; describe that narrow meaning explicitly.

### Validation and Evaluation

The saved reranker configuration states that the first 250 eligible validation cases were excluded from reranker development because those cases had been used to select the generator checkpoint. It records an earlier beam-zero utility of 0.140371 and reranked utility of 0.176949. The later metric-alignment record reports candidate uniqueness 0.990951 and held-out reward proxy 0.304476 before versus 0.314752 after learned selection; the candidate oracle reward was 0.371740. The reference reward combines rule-based clinical F1 (0.45), BLEU (0.15), ROUGE-L (0.15), METEOR-lite (0.10), and CIDEr-like (0.15). The recovered fitting implementation uses an 80/20 grouped split with default seed 29, but actual split counts and patient membership were not saved in this evidence package. Official submission a8799c78 is directly identified in the platform screenshot: CRG 0.3497, macro-F1 0.1264, BLEU mean 0.1657, ROUGE-L 0.2391, METEOR 0.1753, accuracy 0.7576, precision 0.2817, and recall 0.0964. Internal rewards are not official CRG.

### Limits and Limitations

Reranking cannot recover findings absent from all candidates. The auxiliary visual head and report classifier can share report-derived label biases; their agreement is not independent confirmation of pathology. Clinical omission, hallucination, laterality and negation errors remain possible. Decoding policy and random seed depend on generation-call order, which can vary with input ordering or sharding. The first two policies may both yield four-beam decoding because four return sequences are requested. Coarse volumes may obscure small pathology, and unreadable inputs can fall back to an air placeholder. A strict=False generator load is not a comprehensive checkpoint-compatibility audit. External, subgroup, prospective and radiologist-assessed performance are untested.

### Model Weight Availability

**Current selection:** Not available / access restricted (until an authorized public URL exists).

Submission a8799c78 references **task2_weights.zip**, SHA-256 **ab8417b0e0b7395988ecdd748ad1fc543fc6f1d9866a420b95150d28bf793ebc**. Required files include task2_best.pt, Qwen assets under qwen2.5-0.5b-instruct/, RadBERT assets under RadBERT-RoBERTa-4m/, RadBertClassifier.pth, reranker.joblib and reranker.json. The release must distribute the complete required bundle, not just a nominal adapter. Public URL: not yet supplied.

## Task 3

**Method name:** DistilBERT-Conditioned Single-Latent 3D CT Autoencoder

**Leaderboard name / submission ID:** 888a2309 — Task3 full-47149 checkpoint, batched and MHA-ID hardened

**Challenge task:** CT Volume Generation

### Methodology

Our text-conditioned CT generator maps a radiology report to a compact volumetric latent and decodes that latent into a three-dimensional CT image. The submitted code uses a pretrained DistilBERT text encoder, identified in its Dockerfile as distilbert-base-uncased. Attention-masked mean pooling summarizes the token sequence. A multilayer perceptron with GELU activation and layer normalization maps the pooled text representation to a CT latent; the default latent dimension is 512. The report mapper is paired with a convolutional 3D autoencoder whose decoder reconstructs a volume from this single vector.

The supplied autoencoder has four strided convolutional encoder stages and four transposed-convolution decoder stages, with instance normalization and GELU activations; the final decoder uses tanh. Source defaults specify a 64 × 96 × 96 output and 16 initial channels, while saved checkpoint metadata controls the actual instantiated dimensions. The supplied baseline training functions use L1 image reconstruction for the autoencoder and latent mean-squared error plus weighted image L1 reconstruction for the mapper. The default mapper freezes DistilBERT. Actual full-run epoch counts, learning rates, and validation selection were not recovered from the binary checkpoint metadata or training logs.

Terminal history links this submission to the original full47149 training run, rather than the later metric-surrogate reweighting experiment. Inference uses mixed precision, batched prediction, automatic batch-size reduction after CUDA memory errors, asynchronous atomic volume writing, and resumability checks. Output is HU-valued int16 NIfTI with an identity affine. Recursive suffix removal repairs MHA-related case identifiers. This is deterministic text-to-latent regression, not latent diffusion, and it has no classifier-free guidance or voxel-spacing conditioning.

### Methodological Approaches

- Select: **3D vision model**; **Language model**; **Generative model**; **Other approach**.
- Other approach text: Attention-pooled report embeddings, latent regression and a convolutional 3D autoencoder decoder.
- No retrieval, diffusion, ensemble or parameter-efficient language-model fine-tuning is established for this submitted path.

### Data Sources

- Select: **Challenge-provided data**.
- Pretrained DistilBERT is used, but additional CT-RATE training means “Pretrained model only; no additional data” does not apply.
- Synthetic CT is the output of the method; that does not imply synthetic training data were used.

### Validation and Evaluation

The retained full47149 weight bundle is linked by terminal history to task3_full_training_47149/task3_weights_full.zip. It is not the later task3_metric_training_47149 checkpoint. Earlier development included a 100-case completion/format check, which initially exposed an identifier failure; this is not a clinical-quality validation cohort. The uploaded evidence establishes the inference architecture and exact archive hashes but does not contain the full-training split manifest, resolved hyperparameters or an official FVD-based selection record. Displayed ID 888a2309 and its full47149 description were supplied by the participant. The platform list independently shows successful volume-generation scores 0.8323 and 0.8918; which value belongs to this ID, and the FVD_CTNet/semantic metric breakdown, remain unresolved. Do not infer a raw FVD value from the displayed composite score or assign the higher number without its detail page.

### Limits and Limitations

A single latent bottleneck and low output resolution can smooth anatomy and omit small abnormalities. Deterministic regression does not express report-conditioned anatomical diversity. Identity-affine output does not preserve physical geometry from a source scan, and voxel spacing is not a conditioning variable. Report truncation and the absence of explicit structured pathology constraints limit text grounding. Completion checks do not establish realism or clinical fidelity, and the resume check verifies basic image readability rather than matching the output to a particular prompt/checkpoint. Full training provenance, external generalization, subgroup analyses and radiologist validation remain incomplete. No result from later diffusion or surrogate-reweighting studies is claimed for this submission.

### Model Weight Availability

**Current selection:** Not available / access restricted (until an authorized public URL exists).

Submission 888a2309 references **task3_weights_full_47149.zip**, SHA-256 **b06dea5916646cda857f815d595e609523384566fcd602cbc34ad3decf13722a**. Required contents are task3_autoencoder.pt, task3_report_to_latent/mapper.pt and the tokenizer assets in task3_report_to_latent/. DistilBERT is additionally cached into the image during Docker build. Public URL: not yet supplied. The older baseline weights are separate and must not be substituted.

## Required final human additions

- Confirm official score for cd1082ec and 888a2309, including full metric breakdowns and full IDs if available.
- Confirm challenge team name, all members and preferred affiliations.
- Choose authorized weight hosting and add real versioned URLs; inspect component licenses before redistribution.
- Resolve any required training-log/checkpoint metadata gaps rather than substituting code defaults.

The text describes the source snapshots provided. No GPU inference, binary checkpoint loading, or independent platform artifact extraction was performed in this documentation pass.

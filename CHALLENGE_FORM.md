# Methods, validation and limitations

## Task 1 — Multi-window DINOv2 Query-Pooling Ensemble with Decision Calibration

### Methodology

The abnormality classifier represents each CT examination using multi-window slice features and predicts 18 abnormalities with an ensemble of three attention-pooling heads. Input intensities are sanitized, clipped to −1000 to 1000 HU, normalized and resized to 64 × 96 × 96. Uniformly sampled planes along the first tensor axis are rendered using three intensity windows: −1000 to 1000 HU, −150 to 250 HU and −1000 to 400 HU. Images are resized to 224 × 224, replicated across three channels and normalized using the pretrained image processor.

A pretrained DINOv2-base encoder provides a 768-dimensional classification-token feature for each slice image. Each task-specific head projects the feature sequence into 384 dimensions, adds positional embeddings and a residual adapter, and aggregates information using 18 learned pathology queries with eight-head cross-attention. Three trained heads are evaluated, their logits are averaged and sigmoid probabilities are computed. The number of sampled slices is specified by the head checkpoint, with an inference default of 24.

The selected inference recipe applies per-label affine transformations to the ensemble logits. A further output transformation uses fitted temperature, bias and threshold parameters before the challenge-format validation step. This adjusts the operating point for multilabel classification while retaining continuous scores. The submission uses this repaired calibration execution order and the DINO ensemble. The model combines pretrained visual representations with task-specific learned aggregation and post-hoc decision calibration; it does not require a reference report during inference.

The fitted recipe coefficients are distributed in configs/task1/final_recipe.json, and the additional per-label temperatures, biases and thresholds are in configs/task1/task1_metric_calibration.json. The final transform is sigmoid(logit(p)/temperature + bias − logit(threshold)). These are stored fitted parameters, rather than generic calibration defaults.

### Validation and Evaluation

Submission **cd1082ec**: “Task1 DINO ensemble calibration-order repaired.”

Recipe development used 3,039 validation cases: 500 for prior baseline development, 1,037 for model development, 497 for selection, 366 for graph fitting, 348 for calibration and 291 for final internal evaluation. Before the additional output transformation, the final internal subset achieved macro-AUROC 0.8110 and macro-F1 0.2762. A separate calibration evaluation used 1,828 fitting and 1,211 held-out cases, reporting macro-F1 increasing from 0.2890 to 0.5233, with raw macro-AUROC 0.8149. These internal evaluations use distinct protocols and are reported separately from official challenge results.

Official platform evaluation reported **CRG score 0.4803**, **macro-AUROC 0.7814**, **macro-F1 0.5145**, accuracy 0.7445, precision 0.4418 and recall 0.6582. The platform reported a runtime of 76.7 minutes on gpu-l4-xl.

Model selection and calibration are distinct steps: the saved recipe selects the DINO ensemble, while the additional transformation sets per-label operating points using development predictions. Its configuration records acceptance and the fitted parameters for all 18 labels. Positive-temperature transforms preserve within-label ranking mathematically, apart from numerical clipping or rounding; improvement in thresholded F1 therefore does not itself demonstrate improved discrimination. The ensemble consists of three saved heads, without establishing that they are independent cross-validation folds. The distributed configuration also preserves label ordering, allowing the same fitted transformation to be applied consistently to predictions from the released model.

### Limits and Limitations

Slice sampling and coarse resizing can miss small lesions and discard physical-spacing information. The first tensor axis is not guaranteed to be anatomically axial. Report-derived labels can contain errors. Threshold adjustment does not establish calibrated clinical posterior probabilities. Validation data participated in multiple development stages, so the additional calibration holdout is not necessarily independent of all earlier model selection. Unreadable inputs may trigger an air-valued placeholder. Exact historical optimizer schedules and epoch counts are unavailable. External-site, subgroup, prospective and clinical-use performance have not been established.

The pretrained image encoder operates on two-dimensional slices, so learned aggregation does not guarantee preservation of three-dimensional relationships. The selected slice count also limits the evidence available for abnormalities that occupy only a small portion of the examination. Multi-window rendering changes intensity contrast but cannot restore resolution lost during volume resizing. Error rates may differ by abnormality prevalence, acquisition protocol and reconstruction kernel.

The stored decision thresholds were optimized for the development distribution. Their clinical consequences depend on prevalence, the costs of missed and false-positive findings, and the intended decision workflow. No uncertainty intervals, abstention mechanism or prospective calibration assessment are provided for individual predictions. Official aggregate metrics do not establish equal sensitivity across patient groups or disease severities. The released model should therefore be interpreted as a challenge classifier whose operating points require independent evaluation before transfer to a new setting.

## Task 2 — Swin–Qwen LoRA Report Generation with Pathology-Consistency Reranking

### Methodology

The report-generation model combines a volumetric SwinUNETR-derived encoder, a Perceiver-style visual resampler and Qwen2.5-0.5B-Instruct. CT intensities are sanitized, clipped to −1000 to 1000 HU, normalized and resized to 64 × 96 × 96. Deep visual features are adaptively pooled to a spatial grid, projected into the language-model embedding space and compressed into a fixed visual prefix. The implementation defaults to a 4 × 6 × 6 grid and 32 visual tokens, with checkpoint metadata controlling the instantiated configuration. An auxiliary head predicts 18 abnormalities from the volumetric representation.

The language model is adapted using LoRA on query, key, value and output attention projections. The implementation specifies rank 16, scaling 32 and dropout 0.05. Its objective combines autoregressive report prediction with auxiliary pathology binary cross-entropy weighted by 0.25. The visual encoder is frozen by the inference model constructor. Challenge-provided CT images, reports and abnormality labels provide the task-specific supervision.

At inference, the system requests four candidate reports with a 220-token generation limit. A decoding wrapper cycles generation policies across successive calls. A packaged RadBERT classifier estimates abnormality probabilities from each candidate. Six reranking features summarize agreement with the image-based pathology predictions, text confidence, report length, repetition and candidate rank. A regularized ridge reward predictor selects the highest-scoring candidate. Reference-derived clinical and lexical rewards supervise reranker development, whereas inference requires only the input CT and the packaged model assets.

The released configuration fixes the candidate count and reranking feature order. Decoding uses a default seed of 29001 plus the generation-call index, so its stochastic policies are reproducible only with the same call ordering and runtime conditions.

### Validation and Evaluation

Submission **a8799c78**: “Task2 diverse candidates held-out reward reranker hardened.” Reranker development excluded the first 250 eligible validation cases previously used for generator checkpoint selection. The fitting implementation uses grouped splitting with an 80/20 default ratio; complete historical split membership and counts are unavailable. A later held-out reward comparison improved from 0.3045 to 0.3148; the candidate oracle reached 0.3717. This internal reward combines clinical F1, BLEU, ROUGE-L, METEOR-lite and CIDEr-like components and is distinct from official CRG. Official results were **CRG 0.3497**, macro-F1 0.1264, BLEU mean 0.1657, ROUGE-L 0.2391, METEOR 0.1753, accuracy 0.7576, precision 0.2817 and recall 0.0964.

The saved metric-alignment configuration records a candidate uniqueness ratio of 0.9910 and six inference-available features. Development reward weights are 0.45 for clinical F1, 0.15 each for BLEU and ROUGE-L, 0.10 for METEOR-lite, and 0.15 for CIDEr-like. The oracle measures the best candidate obtainable with reference information and is not a deployable selector. References supervise the reward targets during development but are absent from the submitted inference inputs. The configuration identifies the generator checkpoint used for candidate production; a checkpoint filename alone does not establish the complete generator training schedule. After the held-out comparison, the fitting implementation refits a final reward model using all development rows; its training fit is not an additional independent evaluation.

### Limits and Limitations

Reranking cannot recover findings absent from every candidate. Shared report-derived label biases can make image–text agreement misleading. Omissions, hallucinations, laterality mistakes and negation errors remain possible. Decoding policy and random seed depend on generation-call order, so ordering or sharding can affect outputs. Coarse volumes may obscure small pathology. Generator checkpoint loading permits unmatched keys rather than enforcing full strict compatibility. Exact historical optimizer schedules and epoch counts are unavailable. External-site, demographic subgroup, prospective and radiologist-assessed performance have not been established.

The image pathology head and text classifier summarize reports using a finite abnormality vocabulary. Agreement on those labels cannot validate the full report, including severity, location, incidental findings or clinically relevant changes over time. A high-scoring report can therefore remain incomplete or factually incorrect. The learned reward combines approximate clinical and lexical measures, which may reward familiar wording without guaranteeing faithful volumetric grounding.

The language model's pretrained linguistic knowledge is not patient-specific evidence. Visual-prefix compression and fixed report-length limits can omit information from complex examinations. Report-derived training labels can encode documentation practices and unequal disease representation. Overall accuracy is especially difficult to interpret when negative labels dominate. No study here establishes reader benefit, safety of downstream clinical decisions, or performance across languages and reporting conventions. Comparing internal reward improvement with official CRG requires separate evaluation rather than assuming that the two scales measure identical quality.

## Task 3 — DistilBERT-Conditioned Single-Latent 3D CT Autoencoder

### Methodology

The CT-generation model maps a radiology report to a compact volumetric latent representation and decodes it into a three-dimensional CT image. A pretrained DistilBERT-base-uncased encoder produces contextual token embeddings. Attention-masked mean pooling summarizes the token sequence, and a multilayer perceptron with GELU activation and layer normalization maps this report representation to the CT latent space. The implementation defaults to a 512-dimensional latent vector, with saved checkpoint metadata controlling the actual instantiated dimensions.

The corresponding 3D autoencoder contains four strided convolutional encoder stages and four transposed-convolution decoder stages. Instance normalization and GELU activations are used between stages, with a final tanh output. Architecture defaults specify a 64 × 96 × 96 output volume and 16 initial channels. The training implementation uses L1 image reconstruction for the autoencoder and latent mean-squared error plus weighted image L1 reconstruction for the report mapper. The mapper's default image-reconstruction weight is 0.25, and its text encoder is frozen by default.

The submitted full-47,149 checkpoint is used for inference. Reports are processed in batches with mixed precision on CUDA. Memory errors trigger a reduction in batch size; asynchronous atomic writing and resumability support larger inference jobs. Generated intensities are converted to HU-valued int16 NIfTI files with an identity affine. Recursive suffix removal normalizes case identifiers before output validation. This architecture performs text-to-latent regression followed by volumetric decoding; it does not use latent diffusion, classifier-free guidance or voxel-spacing conditioning.

The recovered training implementation uses AdamW with a default learning rate of 1e-4 and batch size 1. The full-data launcher requests five autoencoder epochs followed by five mapper epochs by default. These settings describe the recoverable recipe; run-specific overrides are not established.

### Validation and Evaluation

Submission **888a2309**: “Task3 full-47149 checkpoint, batched and MHA-ID hardened.”

Evaluation used the original full-47,149 checkpoint, distributed as task3_weights_full_47149.zip. Development included completion and output-format checks. The original training/validation membership and checkpoint-selection record are unavailable; the official result evaluates this submitted checkpoint and does not establish a comparison against later training candidates.

Official platform evaluation reported **FVD (CTNet) 0.8323**, FID 2.5D average 3435.7003 and CT-CLIPScore average 10.3737. Plane-specific FID 2.5D values were XY 4240.9594, XZ 3134.4108 and YZ 2931.7306. CT-CLIPScore was 14.6963 for image-to-image (I2I) and 6.0512 for text-to-image (T2I). The platform reported a runtime of 1.5 minutes on gpu-l4-xl. Metric names and values are reported as displayed by the platform.

The recovered launcher initializes from the packaged baseline checkpoints and passes the complete training-CSV row count explicitly, avoiding the wrapper's default 50-row limit. Its default recipe uses five epochs per stage, AdamW, learning rate 1e-4 and batch size 1. Autoencoder fitting minimizes image L1 loss; mapper fitting combines latent mean-squared error with image L1 weighted by 0.25. The resume patch checkpoints every 250 steps by default. These implementation settings support reconstruction of the training procedure, but the launcher alone cannot verify the resolved settings or completion state of a particular historical run.

### Limits and Limitations

A single latent bottleneck and low output resolution can smooth anatomy and omit small abnormalities. Deterministic regression does not represent the diversity of anatomies consistent with a report. Identity-affine output does not preserve patient-space geometry, and voxel spacing is not a conditioning variable. Report truncation and the lack of explicit structured-pathology constraints can weaken clinical grounding. Resume checks establish basic image readability rather than binding an output to a particular prompt and checkpoint. Format validity does not establish realism or clinical fidelity. External-site, subgroup, prospective and radiologist-assessed performance have not been established.

A report does not uniquely specify a CT examination: normal structures, patient habitus and unreported variation can differ even when the text is identical. Consequently, disagreement with a reference volume is not always a clinical error, while visually plausible anatomy is not proof that every described finding was generated. The model has no explicit mechanism to represent that ambiguity through multiple conditional samples.

Distribution-level FVD and FID scores do not verify lesion-level accuracy, anatomy at a particular location, or consistency of each generated image with its report. CT-CLIP similarity is likewise a learned embedding measure rather than a radiologist's assessment. The reported platform metrics lack accompanying bootstrap confidence intervals or repeated-seed analyses here. Without controlled checkpoint comparisons, the official score cannot isolate the effect of full-data training from other changes. Further conclusions about clinical utility or faithful patient-specific synthesis would require independent evaluation.

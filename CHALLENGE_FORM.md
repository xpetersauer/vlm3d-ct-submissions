# Methods, validation and limitations

## Task 1 — Multi-window DINOv2 ensemble with decision calibration

### Methodology

We developed a multilabel classifier that combines pretrained slice representations with learned attention pooling to identify 18 CT abnormalities. CT intensities were clipped to −1000 to 1000 HU, normalized and resized to 64 × 96 × 96. Uniformly sampled slices were displayed using three windows: −1000 to 1000 HU, −150 to 250 HU and −1000 to 400 HU. Each view was resized to 224 × 224 and normalized using the pretrained image processor.

A pretrained DINOv2-base encoder generated 768-dimensional slice features. Three task-specific heads independently projected these features to 384 dimensions and aggregated them using positional embeddings, residual adapters and 18 learned pathology queries. Each query attended to the slice sequence through eight-head cross-attention. The heads produced one logit per abnormality. Challenge-provided abnormality labels supplied task-specific supervision for the attention-pooling heads.

At inference, the three heads' logits were averaged and converted to probabilities. The slice count was read from the checkpoint, with a default of 24. The selected recipe then applied per-label affine logit calibration. A second transformation applied fitted temperatures, biases and threshold shifts before output validation. Its fitting procedure estimated temperature and bias using regularized binary log-loss and selected thresholds to maximize development-set F1. The fitted values are included in the released configurations. The pipeline used CT images alone at inference and returned continuous abnormality scores for each examination.

### Validation and Evaluation

Our submission was assessed through internal validation and official challenge evaluation. Internal development used 3,039 examinations: 500 for baseline development, 1,037 for model development, 497 for model selection, 366 for graph fitting, 348 for calibration and 291 for final internal evaluation. The selected configuration used the three-head DINO ensemble. Its final internal evaluation achieved macro-AUROC 0.8110 and macro-F1 0.2762 before the additional output calibration.

The additional calibration analysis used 1,828 fitting examinations and 1,211 held-out examinations. Per-label temperature and bias parameters were fitted using regularized binary log-loss. Thresholds were selected by maximizing F1 on the fitting partition and then applied to the held-out predictions. Held-out macro-F1 increased from 0.2890 to 0.5233, with macro-AUROC 0.8149. The final transformation shifted the fitted decision thresholds to a common output threshold of 0.5. The released configuration contains the temperature, bias and threshold for each abnormality.

On official challenge evaluation, our submission achieved CRG score 0.4803, macro-AUROC 0.7814 and macro-F1 0.5145. Accuracy was 0.7445, precision was 0.4418 and recall was 0.6582. The platform reported a runtime of 76.7 minutes on gpu-l4-xl. The official evaluation assessed the complete inference pipeline, including the ensemble and both calibration transformations. The stored label order was preserved throughout model prediction, calibration and output formatting.

### Limits and Limitations

Uniform slice sampling can miss abnormalities confined to a small region, including small nodules or limited pleural findings. Resizing the entire examination to 64 × 96 × 96 further reduces spatial detail. The three intensity windows improve contrast representation but cannot recover information removed by resizing. The preprocessing does not guarantee anatomical reorientation, so the first tensor axis need not correspond to the same anatomical plane in every input. Physical voxel spacing is not explicitly modeled.

The classifier aggregates two-dimensional features rather than learning a fully volumetric representation. Attention pooling may therefore capture abnormality-associated appearance without preserving the spatial relationships needed to distinguish similar findings. The evaluation does not isolate the contributions of window selection, slice count and head ensembling. Three heads also do not establish the benefits of a patient-level cross-validation ensemble.

Report-derived labels can omit visible findings or inherit reporting conventions. Thresholds fitted to these labels may favor common findings and the development cohort's reporting patterns. Decision calibration is not evidence that the returned scores represent reliable clinical posterior probabilities. Reuse of validation data across development stages limits interpretation of the additional calibration holdout. Finally, the loader can replace unreadable scans with an air-valued tensor, allowing predictions without usable image evidence. The reported aggregate metrics do not quantify how often that fallback occurred.

## Task 2 — Volumetric report generation with pathology-consistency reranking

### Methodology

We developed a vision–language model that generates candidate CT reports and selects among them using image–text pathology agreement. CT intensities were clipped to −1000 to 1000 HU, normalized and resized to 64 × 96 × 96. A SwinUNETR-derived volumetric encoder produced spatial features, which were projected into the language-model embedding space and compressed by a Perceiver-style resampler. The architecture used a default 4 × 6 × 6 feature grid and 32 visual tokens, with checkpoint metadata specifying the instantiated configuration.

The visual tokens conditioned pretrained Qwen2.5-0.5B-Instruct. LoRA adapters were configured on query, key, value and output attention projections with rank 16, scaling 32 and dropout 0.05. An auxiliary head predicted 18 abnormalities from the visual representation. The training objective combined autoregressive report loss with pathology binary cross-entropy weighted by 0.25. Challenge-provided reports and labels supervised these objectives. The visual encoder was frozen by the model constructor.

At inference, the generator requested four reports with a 220-token output limit. A wrapper cycled beam-based and nucleus-sampling policies across generation calls. A packaged RadBERT-RoBERTa-4m classifier estimated abnormality probabilities for each report. Six features summarized image–text agreement, text confidence, length, repetition and candidate rank. A ridge regression reward model selected the highest-scoring report. Reference-based clinical and lexical rewards supervised reranker development, but reference reports were not required at inference.

### Validation and Evaluation

Our submission used generator checkpoint selection followed by development of the report reranker. The first 250 eligible validation examinations were excluded from reranker development because they had been used for generator selection. The reranker fitting procedure uses an 80/20 grouped split with default random seed 29.

Candidate reports were evaluated against reference reports using clinical F1, BLEU, ROUGE-L, METEOR-lite and CIDEr-like scores. These components were combined with weights of 0.45, 0.15, 0.15, 0.10 and 0.15, respectively. The learned selector improved held-out reward from 0.3045 to 0.3148. Selecting the highest-reward candidate using the reference report yielded an oracle reward of 0.3717. Mean within-examination candidate uniqueness was 0.9910.

The reranker used six features available during inference: image–text pathology cross-entropy, absolute pathology-probability difference, text confidence, report length, four-word repetition and candidate rank. Each examination contributed four candidates, and the report with the highest predicted reward was returned. Reference reports were used to calculate development rewards.

Official challenge evaluation yielded CRG 0.3497 and macro-F1 0.1264. BLEU mean was 0.1657, ROUGE-L was 0.2391 and METEOR was 0.1753. Accuracy, precision and recall were 0.7576, 0.2817 and 0.0964, respectively. These official results evaluate report generation and selection together. The final selector ranked candidates using the learned reward rather than candidate generation order.

### Limits and Limitations

Reranking can select only among the generated candidates. If all candidates omit a finding, a stronger selector cannot recover it. Four reports may also provide limited factual diversity even when their wording differs. The decoding wrapper changes policy between successive generation calls rather than explicitly assigning a different policy to every candidate within each examination. Because four return sequences are requested, its nominal greedy and four-beam policies can both become four-beam decoding. Candidate generation also depends on input order through the call-index seed.

The consistency score compares two predictions derived from imperfect report-based supervision. Agreement between the image pathology head and RadBERT is therefore not independent confirmation that a finding is present. Their 18-label vocabulary cannot fully represent laterality, severity, precise location or incidental observations. Fluent reports can receive favorable agreement scores while remaining incomplete or incorrect.

The visual prefix compresses coarse volumetric features into a fixed token sequence, and the 220-token output limit may truncate complex examinations. The reward combines rule-based clinical labels and lexical similarity, potentially favoring familiar phrasing over complete factual coverage. The available evaluation does not separately quantify hallucination, omission, laterality and negation errors. Differences between the development reward and official CRG also limit conclusions about whether the reranker improves the challenge's intended clinical reporting criteria.

## Task 3 — DistilBERT-conditioned latent regression for CT generation

### Methodology

We developed a report-conditioned generator that predicts a single CT latent vector and decodes it into a three-dimensional volume. Pretrained DistilBERT-base-uncased encoded the report, and attention-masked mean pooling summarized its token embeddings. A multilayer perceptron with GELU activation and layer normalization mapped this representation to the autoencoder's latent space. The architecture defaults were a 512-dimensional latent, 16 initial convolutional channels and a 64 × 96 × 96 output; checkpoint metadata controlled the instantiated dimensions.

The autoencoder comprised four strided convolutional encoder stages and four transposed-convolution decoder stages, with instance normalization, GELU activations and a final tanh output. Training proceeded in two stages: image reconstruction with L1 loss, followed by report-to-latent regression with latent mean-squared error plus image L1 loss weighted by 0.25. The recovered training recipe used AdamW, learning rate 1e-4, batch size 1 and five epochs per stage by default. During mapper training, the autoencoder was frozen; the text encoder was frozen by default. These values are the defaults specified by the training code.

Inference used the submitted full-47,149 checkpoint. Reports were processed in CUDA mixed-precision batches, with automatic batch reduction after memory errors. The predicted latent was decoded directly, without diffusion sampling. Volumes were written atomically as HU-valued int16 NIfTI images with identity affines. Case identifiers were normalized by removing repeated filename suffixes, and readable existing outputs could be reused when resuming inference.

### Validation and Evaluation

Our submission used the original full-47,149 training checkpoint for report-conditioned CT generation. The full-data training procedure initialized the autoencoder and report mapper from baseline checkpoints and used the complete training report table. Its default schedule comprised five epochs per stage, with checkpoints saved every 250 steps. The resulting autoencoder and mapper were packaged together for inference.

Development evaluation checked that every input prompt produced a readable volume with the expected case identifier and NIfTI output format. Identifier handling removed repeated filename suffixes, including MHA suffixes, before writing the final output. Batched inference and resumability supported evaluation of larger prompt collections.

Official challenge evaluation assessed the generated volumes using FVD (CTNet), FID 2.5D and CT-CLIPScore. Our submission achieved FVD (CTNet) 0.8323, average FID 2.5D 3435.7003 and average CT-CLIPScore 10.3737. Plane-specific FID values were 4240.9594 for XY, 3134.4108 for XZ and 2931.7306 for YZ. CT-CLIPScore was 14.6963 for image-to-image similarity and 6.0512 for text-to-image similarity.

FVD summarizes volumetric feature distributions, while the three FID measurements assess image distributions in orthogonal planes. CT-CLIPScore measures similarity in the learned image–text representation space. The platform reported a runtime of 1.5 minutes on gpu-l4-xl for our submission. The final output consisted of one HU-valued three-dimensional image for each case identifier.

### Limits and Limitations

Mapping a report to a single latent vector creates a strong information bottleneck. The low-resolution decoder can smooth fine structures and omit small abnormalities even when they are described in the report. Regression losses encourage average predictions when multiple anatomies are compatible with the same text. Because inference does not sample alternative latent representations, the model cannot express that conditional ambiguity through diverse generated volumes.

Mean pooling compresses the report into one representation and may weaken distinctions involving negation, laterality or relationships between findings. Input truncation can remove clinically relevant text, while the absence of explicit pathology constraints leaves report fidelity dependent on the learned mapping. Voxel spacing is not a conditioning variable. Identity-affine outputs therefore do not support a claim of accurate physical geometry or patient-specific spatial correspondence.

FVD and FID summarize distributions of learned image features rather than verifying each generated lesion. CT-CLIP similarity likewise does not establish that all report statements are visually supported. A report also underdetermines normal anatomy, making reference-image similarity an imperfect measure of conditional correctness. The available results do not separate reconstruction quality from report-conditioning quality or isolate the benefit of full-data training. Finally, the resume mechanism checks image readability rather than matching each output to its prompt and checkpoint, so reused files can invalidate a comparison if those inputs change.

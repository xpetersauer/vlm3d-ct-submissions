# References and component provenance

- CT-RATE/CT-CLIP: https://arxiv.org/abs/2403.17834
- Official VLM3D code/contracts: https://github.com/forithmus/VLM3D-Dockers
- DINOv2: https://arxiv.org/abs/2304.07193 ; https://github.com/facebookresearch/dinov2
- Qwen2.5-0.5B-Instruct: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
- Swin medical pretraining/architecture: https://arxiv.org/abs/2111.14791
  (architecture citation does not establish which visual pretraining weights were used).
- MONAI: https://github.com/Project-MONAI/MONAI
- LoRA: https://arxiv.org/abs/2106.09685
- DistilBERT: https://arxiv.org/abs/1910.01108
- Calibration: https://proceedings.mlr.press/v70/guo17a.html
- Radiology-report evaluation: https://doi.org/10.1016/j.patter.2023.100802

The submitted RadBERT assets are named RadBERT-RoBERTa-4m and
RadBertClassifier.pth; the config identifies a RoBERTa-base architecture. Exact
upstream RadBERT repository revision and task classifier provenance require
the original asset metadata/model card, not a guess from the folder name.

docs/references.bib is the recovered broader draft bibliography. It contains
additional proposed-research references and unreviewed metadata; their presence
is not proof that those methods were used. The primary documentation evidence
here is the included source/config snapshots and provenance/ARTIFACTS.json.

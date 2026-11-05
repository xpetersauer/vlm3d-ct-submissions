"""Differentiable surrogates for VLM3D metric-aligned training.

These are deliberately named *surrogates*.  Exact CRG, BLEU and FVD remain
validation/model-selection metrics because their ranking, thresholding,
token-sampling, and corpus-statistic operations are not ordinary per-example
back-propagation losses.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def soft_macro_f1_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6):
    probabilities = logits.sigmoid()
    targets = targets.float()
    tp = (probabilities * targets).sum(dim=0)
    fp = (probabilities * (1.0 - targets)).sum(dim=0)
    fn = ((1.0 - probabilities) * targets).sum(dim=0)
    f1 = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    return 1.0 - f1.mean()


def asymmetric_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    positive_gamma: float = 0.0,
    negative_gamma: float = 2.0,
    clip: float = 0.05,
):
    targets = targets.float()
    probability = logits.sigmoid()
    positive = probability
    negative = 1.0 - probability
    if clip > 0:
        negative = (negative + clip).clamp(max=1.0)
    log_loss = targets * torch.log(positive.clamp_min(1e-8))
    log_loss += (1.0 - targets) * torch.log(negative.clamp_min(1e-8))
    weight = torch.pow(1.0 - positive - negative, 0)  # shape/device anchor
    weight = torch.where(
        targets > 0.5,
        torch.pow(1.0 - positive, positive_gamma),
        torch.pow(1.0 - negative, negative_gamma),
    )
    return -(weight * log_loss).mean()


def pairwise_auc_surrogate(logits: torch.Tensor, targets: torch.Tensor):
    """Logistic positive-vs-negative ranking loss, averaged over valid labels."""
    terms = []
    for label in range(logits.shape[1]):
        score = logits[:, label]
        truth = targets[:, label] > 0.5
        positive = score[truth]
        negative = score[~truth]
        if positive.numel() and negative.numel():
            differences = positive[:, None] - negative[None, :]
            terms.append(F.softplus(-differences).mean())
    return torch.stack(terms).mean() if terms else logits.sum() * 0.0


def crg_surrogate_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    bce_weight: float = 0.55,
    soft_f1_weight: float = 0.35,
    rank_weight: float = 0.10,
):
    """Task-1 loss aligned with macro-F1 and AUROC components of CRG."""
    bce = asymmetric_bce_loss(logits, targets)
    soft_f1 = soft_macro_f1_loss(logits, targets)
    rank = pairwise_auc_surrogate(logits.float(), targets.float())
    total = bce_weight * bce + soft_f1_weight * soft_f1 + rank_weight * rank
    return total, {
        "asymmetric_bce": bce.detach(),
        "soft_macro_f1": soft_f1.detach(),
        "pairwise_rank": rank.detach(),
    }


def _gradient_l1(prediction: torch.Tensor, target: torch.Tensor):
    terms = []
    for dimension in (-3, -2, -1):
        pred_diff = torch.diff(prediction, dim=dimension)
        target_diff = torch.diff(target, dim=dimension)
        terms.append(F.l1_loss(pred_diff, target_diff))
    return torch.stack(terms).mean()


def _multiscale_l1(prediction: torch.Tensor, target: torch.Tensor):
    terms = []
    for kernel in (2, 4, 8):
        if min(prediction.shape[-3:]) < kernel:
            continue
        pred_pool = F.avg_pool3d(prediction, kernel_size=kernel, stride=kernel)
        target_pool = F.avg_pool3d(target, kernel_size=kernel, stride=kernel)
        terms.append(F.l1_loss(pred_pool, target_pool))
    return torch.stack(terms).mean() if terms else prediction.sum() * 0.0


def _feature_moment_loss(prediction: torch.Tensor, target: torch.Tensor):
    """Stable mean/variance/cross-channel surrogate for FVD moments.

    This is not official CT-Net FVD.  It gives gradients toward matching the
    first two statistics that FVD later measures in a frozen feature space.
    """
    pred = F.adaptive_avg_pool3d(prediction.float(), (8, 12, 12)).flatten(2)
    real = F.adaptive_avg_pool3d(target.float(), (8, 12, 12)).flatten(2)
    pred_mean = pred.mean(dim=-1)
    real_mean = real.mean(dim=-1)
    pred_centered = pred - pred.mean(dim=-1, keepdim=True)
    real_centered = real - real.mean(dim=-1, keepdim=True)
    pred_var = pred_centered.square().mean(dim=-1)
    real_var = real_centered.square().mean(dim=-1)
    return F.mse_loss(pred_mean, real_mean) + F.mse_loss(pred_var, real_var)


def task3_autoencoder_metric_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    l1_weight: float = 0.60,
    gradient_weight: float = 0.15,
    multiscale_weight: float = 0.15,
    moment_weight: float = 0.10,
):
    l1 = F.l1_loss(prediction, target)
    gradient = _gradient_l1(prediction, target)
    multiscale = _multiscale_l1(prediction, target)
    moments = _feature_moment_loss(prediction, target)
    total = (
        l1_weight * l1
        + gradient_weight * gradient
        + multiscale_weight * multiscale
        + moment_weight * moments
    )
    return total, {
        "voxel_l1": l1.detach(),
        "gradient_l1": gradient.detach(),
        "multiscale_l1": multiscale.detach(),
        "fvd_moment_surrogate": moments.detach(),
    }


def task3_mapper_metric_loss(
    predicted_latent: torch.Tensor,
    target_latent: torch.Tensor,
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    latent_weight: float = 1.0,
    image_weight: float = 0.35,
):
    latent = F.smooth_l1_loss(predicted_latent, target_latent)
    image, components = task3_autoencoder_metric_loss(prediction, target)
    total = latent_weight * latent + image_weight * image
    return total, {"latent_smooth_l1": latent.detach(), **components}


def self_critical_sequence_loss(
    sampled_token_log_probabilities: torch.Tensor,
    sampled_reward: torch.Tensor,
    greedy_reward: torch.Tensor,
    valid_token_mask: torch.Tensor | None = None,
):
    """Optional Task-2 SCST loss using reference metrics during training only."""
    advantage = (sampled_reward - greedy_reward).detach()
    log_probabilities = sampled_token_log_probabilities
    if valid_token_mask is not None:
        mask = valid_token_mask.float()
        sequence_log_probability = (log_probabilities * mask).sum(-1) / mask.sum(-1).clamp_min(1.0)
    else:
        sequence_log_probability = log_probabilities.mean(-1)
    return -(advantage * sequence_log_probability).mean()

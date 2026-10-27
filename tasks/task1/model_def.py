from __future__ import annotations

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricLoss(nn.Module):
    """Multi-label ASL: downweights easy negatives while retaining positive gradients."""
    def __init__(self, gamma_neg=4.0, gamma_pos=1.0, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = float(gamma_neg)
        self.gamma_pos = float(gamma_pos)
        self.clip = float(clip)
        self.eps = float(eps)

    def forward(self, logits, targets):
        x = torch.sigmoid(logits)
        xs_pos = x
        xs_neg = 1 - x
        if self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
        loss = targets * torch.log(xs_pos.clamp_min(self.eps))
        loss += (1 - targets) * torch.log(xs_neg.clamp_min(self.eps))
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            with torch.no_grad():
                pt = xs_pos * targets + xs_neg * (1 - targets)
                gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
                weight = torch.pow(1 - pt, gamma)
            loss *= weight
        return -loss.mean()


class QueryPoolClassifier(nn.Module):
    """Cheap AnyMC3D-inspired head: frozen slice features -> learned query attention -> 18 logits."""
    def __init__(self, feature_dim=768, d_model=384, nhead=8, num_labels=18, dropout=.1):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.d_model = int(d_model)
        self.num_labels = int(num_labels)
        self.proj = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.residual_adapter = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, d_model),
        )
        self.pos = nn.Parameter(torch.randn(1, 96, d_model) * .01)
        self.query = nn.Parameter(torch.randn(1, num_labels, d_model) * .02)
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.out = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, feats):
        # feats: [B,T,F]
        z = self.proj(feats)
        z = z + .1 * self.residual_adapter(z)
        z = z + self.pos[:, :z.shape[1]]
        q = self.query.expand(z.shape[0], -1, -1)
        a, _ = self.attn(self.norm_q(q), self.norm_kv(z), self.norm_kv(z), need_weights=False)
        h = q + a
        h = h + self.ff(h)
        return self.out(h).squeeze(-1)


class CPUEMA:
    """EMA for a small head; shadow weights live on CPU to avoid meaningful GPU-memory cost."""
    def __init__(self, model, decay=.999):
        self.decay = float(decay)
        self.shadow = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            x = v.detach().cpu()
            if not torch.is_floating_point(x):
                self.shadow[k] = x.clone()
            else:
                self.shadow[k].mul_(self.decay).add_(x, alpha=1 - self.decay)

    def copy_to(self, model):
        model.load_state_dict(self.shadow, strict=True)


class DeviceEMA:
    """EMA shadow copy on the same device; useful for the larger Swin classifier."""
    def __init__(self, model, decay=.999):
        self.decay = float(decay)
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if k not in self.shadow:
                self.shadow[k] = v.detach().clone()
            elif torch.is_floating_point(v):
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v.detach())

    def state_dict(self):
        return {k: v.detach().cpu() for k, v in self.shadow.items()}

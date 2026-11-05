#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import torch


def patient_group_from_volume_name(x: str) -> str:
    name = str(x).split("/")[-1]
    for suf in [".nii.gz", ".nii", ".npy"]:
        if name.endswith(suf):
            name = name[:-len(suf)]
    parts = name.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else name


def get_cosine_with_warmup(optimizer, total_steps, warmup_ratio=0.05):
    from transformers import get_cosine_schedule_with_warmup
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    return get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max(total_steps, warmup_steps + 1),
    )


def save_training_state(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    global_step,
    best_val_loss,
    extra=None,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_val_loss": float(best_val_loss),
        "extra": extra or {},
    }
    torch.save(payload, path)
    return path


class CheckpointManager:
    """
    Always maintains:
      last.pt         most recently completed epoch
      best.pt         lowest validation loss
      step_latest.pt  periodically overwritten crash-safety snapshot
    """
    def __init__(self, root, every_steps=1000):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.every_steps = int(every_steps)

    def maybe_step(self, model, optimizer, scheduler, epoch, global_step, best_val_loss, extra=None):
        if self.every_steps > 0 and global_step % self.every_steps == 0:
            return save_training_state(
                self.root / "step_latest.pt",
                model, optimizer, scheduler, epoch, global_step, best_val_loss, extra,
            )

    def end_epoch(self, model, optimizer, scheduler, epoch, global_step, val_loss, best_val_loss, extra=None):
        save_training_state(
            self.root / "last.pt",
            model, optimizer, scheduler, epoch, global_step, best_val_loss, extra,
        )
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = float(val_loss)
            save_training_state(
                self.root / "best.pt",
                model, optimizer, scheduler, epoch, global_step, best_val_loss, extra,
            )
        return best_val_loss, improved

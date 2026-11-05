#!/usr/bin/env python3
"""Install the Task-3 FVD-aligned surrogate into the resumable baseline trainer."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil


MARKER = "VLM3D_TASK3_FVD_SURROGATE_V1"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_code", type=Path)
    args = parser.parse_args()
    path = args.model_code.resolve()
    source = path.read_text()
    if MARKER in source:
        print(f"[task3-loss] already installed: {path}")
        return 0
    backup = path.with_name(path.name + ".before_fvd_surrogate.bak")
    if not backup.exists():
        shutil.copy2(path, backup)
    module_source = Path(__file__).with_name("metric_losses.py")
    shutil.copy2(module_source, path.parent / "metric_losses.py")

    import_anchor = "import torch.nn.functional as F"
    if import_anchor not in source:
        raise SystemExit(f"Cannot find {import_anchor!r} in {path}")
    source = source.replace(
        import_anchor,
        import_anchor + "\nfrom metric_losses import task3_autoencoder_metric_loss, task3_mapper_metric_loss  # " + MARKER,
        1,
    )
    ae_old = "loss = F.l1_loss(x_hat, x)"
    if source.count(ae_old) != 1:
        raise SystemExit(f"Expected exactly one AE L1 loss, found {source.count(ae_old)}")
    source = source.replace(
        ae_old,
        "loss, metric_components = task3_autoencoder_metric_loss(x_hat, x)",
        1,
    )
    mapper_old = """latent_loss = F.mse_loss(z_pred, z_true)
                x_pred = autoencoder.decode(z_pred)
                recon_loss = F.l1_loss(x_pred, x)
                loss = latent_loss + recon_loss_weight * recon_loss"""
    mapper_new = """x_pred = autoencoder.decode(z_pred)
                loss, metric_components = task3_mapper_metric_loss(
                    z_pred, z_true, x_pred, x,
                    image_weight=max(float(recon_loss_weight), 0.35),
                )"""
    if source.count(mapper_old) != 1:
        raise SystemExit(f"Expected exactly one mapper loss block, found {source.count(mapper_old)}")
    source = source.replace(mapper_old, mapper_new, 1)
    loss_value = "loss_value = float(loss.detach().cpu())"
    if source.count(loss_value) < 2:
        raise SystemExit("Could not locate both Task-3 loss logging sites")
    logging = """loss_value = float(loss.detach().cpu())
            if step == 1 or step % checkpoint_every == 0:
                component_text = " ".join(
                    f"{name}={float(value.cpu()):.4f}"
                    for name, value in metric_components.items()
                )
                print(f"[task3-metric-loss] {component_text}", flush=True)"""
    source = source.replace(loss_value, logging)
    path.write_text(source)
    compile(source, str(path), "exec")
    print(f"[task3-loss] PATCHED: {path}")
    print(f"[task3-loss] backup: {backup}")
    print("[task3-loss] exact FVD is still required for checkpoint selection")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Install step-checkpointed, resumable Task-3 training functions.

The patch preserves the original checkpoint keys used by inference while adding
optimizer/scaler/training state. A pre-existing baseline checkpoint without
training state is treated as warm-start weights at epoch zero.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
import shutil
import sys
import tempfile
import textwrap


MARKER = "TASK3_FULLDATA_RESUMABLE_TRAINING_V1"

NEW_AE = textwrap.dedent(
    r'''
    def train_task3_autoencoder(
        train_df,
        checkpoint_path,
        target_shape=(64, 96, 96),
        latent_dim=512,
        base_channels=16,
        batch_size=1,
        epochs=5,
        lr=1e-4,
        num_workers=0,
        cache_dir="./ct_rate_volume_cache",
        hf_token=None,
        max_rows=None,
    ):
        """Stage-A training with atomic step resume. TASK3_FULLDATA_RESUMABLE_TRAINING_V1."""
        import json
        import os
        import time
        import uuid
        from torch.utils.data import Subset

        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_every = max(1, int(os.getenv("TASK3_CHECKPOINT_EVERY_STEPS", "250")))
        seed = int(os.getenv("TASK3_TRAIN_SEED", "29"))
        progress_raw = os.getenv("TASK3_TRAIN_PROGRESS_PATH", "").strip()
        progress_path = Path(progress_raw) if progress_raw else None
        if progress_path is not None:
            progress_path.parent.mkdir(parents=True, exist_ok=True)

        ds = CTReportToVolumeDataset(
            df=train_df,
            target_shape=target_shape,
            cache_dir=cache_dir,
            hf_token=hf_token,
            max_rows=max_rows,
            report_col="Report_EN",
        )
        if len(ds) == 0:
            raise ValueError("Task-3 autoencoder training dataset is empty")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            raise RuntimeError("Full Task-3 training requires CUDA")

        existing = None
        if checkpoint_path.is_file():
            existing = torch.load(checkpoint_path, map_location="cpu")
            target_shape = tuple(existing.get("target_shape", target_shape))
            latent_dim = int(existing.get("latent_dim", latent_dim))
            base_channels = int(existing.get("base_channels", base_channels))

        model = CTVolumeAutoencoder(
            target_shape=target_shape,
            latent_dim=latent_dim,
            base_channels=base_channels,
        ).to(device)
        if existing is not None and "model" in existing:
            model.load_state_dict(existing["model"])
            print(f"[task3-train] warm-started AE from {checkpoint_path}", flush=True)

        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        state = (existing or {}).get("training_state") or {}
        if state.get("stage") != "autoencoder":
            state = {}
        if state.get("optimizer"):
            opt.load_state_dict(state["optimizer"])
        if state.get("scaler"):
            scaler.load_state_dict(state["scaler"])

        start_epoch = int(state.get("epoch", 0))
        start_sample = int(state.get("next_sample", 0))
        if state.get("complete") and start_epoch >= epochs:
            print(f"[task3-train] AE already complete at epoch {start_epoch}", flush=True)
            return model, checkpoint_path

        def write_progress(epoch, next_sample, loss, complete=False):
            if progress_path is None:
                return
            payload = {
                "stage": "autoencoder",
                "epoch_zero_based": epoch,
                "epochs_target": epochs,
                "next_sample": next_sample,
                "dataset_rows": len(ds),
                "last_loss": None if loss is None else float(loss),
                "complete": bool(complete),
                "updated_unix": time.time(),
            }
            tmp = progress_path.with_name(f".{progress_path.name}.{uuid.uuid4().hex}.tmp")
            tmp.write_text(json.dumps(payload, indent=2))
            os.replace(tmp, progress_path)

        def save_checkpoint(epoch, next_sample, loss, complete=False):
            training_state = {
                "stage": "autoencoder",
                "epoch": int(epoch),
                "next_sample": int(next_sample),
                "epochs_target": int(epochs),
                "dataset_rows": int(len(ds)),
                "optimizer": opt.state_dict(),
                "scaler": scaler.state_dict(),
                "complete": bool(complete),
                "last_loss": None if loss is None else float(loss),
            }
            payload = {
                "model": model.state_dict(),
                "target_shape": tuple(target_shape),
                "latent_dim": latent_dim,
                "base_channels": base_channels,
                "model_class": "CTVolumeAutoencoder",
                "training_state": training_state,
            }
            tmp = checkpoint_path.with_name(
                f".{checkpoint_path.name}.{uuid.uuid4().hex}.tmp"
            )
            torch.save(payload, tmp)
            os.replace(tmp, checkpoint_path)
            write_progress(epoch, next_sample, loss, complete=complete)

        for epoch in range(start_epoch, epochs):
            generator = torch.Generator().manual_seed(seed + epoch)
            order = torch.randperm(len(ds), generator=generator).tolist()
            first_sample = start_sample if epoch == start_epoch else 0
            remaining = Subset(ds, order[first_sample:])
            dl = DataLoader(
                remaining,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )
            model.train()
            losses = []
            processed = first_sample

            for step, (x, report_text, volume_names) in enumerate(dl, start=1):
                x = x.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=True):
                    x_hat, z = model(x)
                    loss = F.l1_loss(x_hat, x)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                loss_value = float(loss.detach().cpu())
                losses.append(loss_value)
                processed = min(len(ds), processed + int(x.shape[0]))

                if step % checkpoint_every == 0:
                    save_checkpoint(epoch, processed, loss_value, complete=False)
                    print(
                        f"[task3-train] AE epoch={epoch + 1}/{epochs} "
                        f"samples={processed}/{len(ds)} loss={loss_value:.4f} checkpointed",
                        flush=True,
                    )

            mean_loss = float(np.mean(losses)) if losses else None
            save_checkpoint(epoch + 1, 0, mean_loss, complete=(epoch + 1 >= epochs))
            print(
                f"[task3-train] AE epoch={epoch + 1}/{epochs} complete "
                f"loss={mean_loss}",
                flush=True,
            )
            start_sample = 0

        return model, checkpoint_path
    '''
).lstrip()


NEW_MAPPER = textwrap.dedent(
    r'''
    def train_task3_report_to_latent(
        train_df,
        autoencoder_checkpoint_path,
        mapper_checkpoint_dir,
        text_encoder_name="distilbert-base-uncased",
        target_shape=(64, 96, 96),
        latent_dim=512,
        batch_size=1,
        epochs=5,
        lr=1e-4,
        num_workers=0,
        cache_dir="./ct_rate_volume_cache",
        hf_token=None,
        max_rows=None,
        max_text_length=256,
        freeze_text_encoder=True,
        recon_loss_weight=0.25,
    ):
        """Stage-B training with atomic step resume. TASK3_FULLDATA_RESUMABLE_TRAINING_V1."""
        import json
        import os
        import time
        import uuid
        from torch.utils.data import Subset

        mapper_checkpoint_dir = Path(mapper_checkpoint_dir)
        mapper_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = mapper_checkpoint_dir / "mapper.pt"
        checkpoint_every = max(1, int(os.getenv("TASK3_CHECKPOINT_EVERY_STEPS", "250")))
        seed = int(os.getenv("TASK3_TRAIN_SEED", "29"))
        progress_raw = os.getenv("TASK3_TRAIN_PROGRESS_PATH", "").strip()
        progress_path = Path(progress_raw) if progress_raw else None
        if progress_path is not None:
            progress_path.parent.mkdir(parents=True, exist_ok=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            raise RuntimeError("Full Task-3 training requires CUDA")
        autoencoder, ae_ckpt = load_task3_autoencoder_checkpoint(
            autoencoder_checkpoint_path,
            device=device,
        )
        for parameter in autoencoder.parameters():
            parameter.requires_grad = False

        existing = torch.load(checkpoint_path, map_location="cpu") if checkpoint_path.is_file() else None
        if existing is not None:
            text_encoder_name = existing.get("text_encoder_name", text_encoder_name)
            latent_dim = int(existing.get("latent_dim", latent_dim))
            target_shape = tuple(existing.get("target_shape", target_shape))
            max_text_length = int(existing.get("max_text_length", max_text_length))
            freeze_text_encoder = bool(existing.get("freeze_text_encoder", freeze_text_encoder))

        if (mapper_checkpoint_dir / "tokenizer_config.json").is_file():
            tokenizer = AutoTokenizer.from_pretrained(mapper_checkpoint_dir)
        else:
            tokenizer = AutoTokenizer.from_pretrained(text_encoder_name)
        mapper = ReportToCTLatentMapper(
            text_encoder_name=text_encoder_name,
            latent_dim=latent_dim,
            freeze_text_encoder=freeze_text_encoder,
        ).to(device)
        if existing is not None and "model" in existing:
            mapper.load_state_dict(existing["model"])
            print(f"[task3-train] warm-started mapper from {checkpoint_path}", flush=True)

        ds = CTReportToVolumeDataset(
            df=train_df,
            target_shape=target_shape,
            cache_dir=cache_dir,
            hf_token=hf_token,
            max_rows=max_rows,
            report_col="Report_EN",
        )
        if len(ds) == 0:
            raise ValueError("Task-3 mapper training dataset is empty")

        opt = torch.optim.AdamW(
            [parameter for parameter in mapper.parameters() if parameter.requires_grad],
            lr=lr,
        )
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        state = (existing or {}).get("training_state") or {}
        if state.get("stage") != "mapper":
            state = {}
        if state.get("optimizer"):
            opt.load_state_dict(state["optimizer"])
        if state.get("scaler"):
            scaler.load_state_dict(state["scaler"])

        start_epoch = int(state.get("epoch", 0))
        start_sample = int(state.get("next_sample", 0))
        tokenizer.save_pretrained(mapper_checkpoint_dir)
        if state.get("complete") and start_epoch >= epochs:
            print(f"[task3-train] mapper already complete at epoch {start_epoch}", flush=True)
            return mapper, tokenizer, mapper_checkpoint_dir

        def write_progress(epoch, next_sample, loss, complete=False):
            if progress_path is None:
                return
            payload = {
                "stage": "mapper",
                "epoch_zero_based": epoch,
                "epochs_target": epochs,
                "next_sample": next_sample,
                "dataset_rows": len(ds),
                "last_loss": None if loss is None else float(loss),
                "complete": bool(complete),
                "updated_unix": time.time(),
            }
            tmp = progress_path.with_name(f".{progress_path.name}.{uuid.uuid4().hex}.tmp")
            tmp.write_text(json.dumps(payload, indent=2))
            os.replace(tmp, progress_path)

        def save_checkpoint(epoch, next_sample, loss, complete=False):
            training_state = {
                "stage": "mapper",
                "epoch": int(epoch),
                "next_sample": int(next_sample),
                "epochs_target": int(epochs),
                "dataset_rows": int(len(ds)),
                "optimizer": opt.state_dict(),
                "scaler": scaler.state_dict(),
                "complete": bool(complete),
                "last_loss": None if loss is None else float(loss),
            }
            payload = {
                "model": mapper.state_dict(),
                "text_encoder_name": text_encoder_name,
                "latent_dim": latent_dim,
                "target_shape": tuple(target_shape),
                "autoencoder_checkpoint": str(autoencoder_checkpoint_path),
                "max_text_length": max_text_length,
                "freeze_text_encoder": freeze_text_encoder,
                "model_class": "ReportToCTLatentMapper",
                "training_state": training_state,
            }
            tmp = checkpoint_path.with_name(
                f".{checkpoint_path.name}.{uuid.uuid4().hex}.tmp"
            )
            torch.save(payload, tmp)
            os.replace(tmp, checkpoint_path)
            write_progress(epoch, next_sample, loss, complete=complete)

        for epoch in range(start_epoch, epochs):
            generator = torch.Generator().manual_seed(seed + 10000 + epoch)
            order = torch.randperm(len(ds), generator=generator).tolist()
            first_sample = start_sample if epoch == start_epoch else 0
            remaining = Subset(ds, order[first_sample:])
            dl = DataLoader(
                remaining,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )
            mapper.train()
            losses = []
            processed = first_sample

            for step, (x, report_texts, volume_names) in enumerate(dl, start=1):
                x = x.to(device, non_blocking=True)
                tokenized = tokenizer(
                    list(report_texts),
                    padding=True,
                    truncation=True,
                    max_length=max_text_length,
                    return_tensors="pt",
                )
                input_ids = tokenized["input_ids"].to(device, non_blocking=True)
                attention_mask = tokenized["attention_mask"].to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.no_grad():
                    z_true = autoencoder.encode(x)
                with torch.cuda.amp.autocast(enabled=True):
                    z_pred = mapper(input_ids=input_ids, attention_mask=attention_mask)
                    latent_loss = F.mse_loss(z_pred, z_true)
                    x_pred = autoencoder.decode(z_pred)
                    recon_loss = F.l1_loss(x_pred, x)
                    loss = latent_loss + recon_loss_weight * recon_loss
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                loss_value = float(loss.detach().cpu())
                losses.append(loss_value)
                processed = min(len(ds), processed + int(x.shape[0]))

                if step % checkpoint_every == 0:
                    save_checkpoint(epoch, processed, loss_value, complete=False)
                    print(
                        f"[task3-train] mapper epoch={epoch + 1}/{epochs} "
                        f"samples={processed}/{len(ds)} loss={loss_value:.4f} checkpointed",
                        flush=True,
                    )

            mean_loss = float(np.mean(losses)) if losses else None
            save_checkpoint(epoch + 1, 0, mean_loss, complete=(epoch + 1 >= epochs))
            print(
                f"[task3-train] mapper epoch={epoch + 1}/{epochs} complete "
                f"loss={mean_loss}",
                flush=True,
            )
            start_sample = 0

        tokenizer.save_pretrained(mapper_checkpoint_dir)
        return mapper, tokenizer, mapper_checkpoint_dir
    '''
).lstrip()


REPLACEMENTS = {
    "train_task3_autoencoder": NEW_AE,
    "train_task3_report_to_latent": NEW_MAPPER,
}


def install(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    if MARKER in source:
        print(f"OK already patched: {path}")
        return

    tree = ast.parse(source, filename=str(path))
    targets = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in REPLACEMENTS
    }
    missing = sorted(set(REPLACEMENTS) - set(targets))
    if missing:
        raise RuntimeError(f"missing expected functions in {path}: {missing}")

    lines = source.splitlines(keepends=True)
    ordered = sorted(targets.values(), key=lambda node: node.lineno, reverse=True)
    for node in ordered:
        replacement = REPLACEMENTS[node.name]
        if not replacement.endswith("\n"):
            replacement += "\n"
        lines[node.lineno - 1 : node.end_lineno] = [replacement]
    patched = "".join(lines)
    compile(patched, str(path), "exec")

    backup = path.with_name(path.name + ".before_full_training_resume.bak")
    if not backup.exists():
        shutil.copy2(path, backup)

    mode = path.stat().st_mode
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(patched)
        tmp = Path(handle.name)
    os.chmod(tmp, mode)
    os.replace(tmp, path)
    print(f"PATCHED resumable full-data training: {path}")


def main() -> int:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} model_code_from_original.py", file=sys.stderr)
        return 2
    install(Path(sys.argv[1]).expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

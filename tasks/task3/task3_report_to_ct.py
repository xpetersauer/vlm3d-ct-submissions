#!/usr/bin/env python3
"""
Task 3 from your original notebook: report -> generated CT volume.
Official VLM3D/Forithmus track: ct-volume-generation / ctgen.

Starts from your original Task 3 report-to-CT baseline code but wraps it as a
runnable Python script with train, predict, local_eval, and Forithmus modes.

Forithmus contract:
  input:  /input/prompts.json
  output: /output/<input_image_name>.nii.gz, one volume per prompt.
"""

from pathlib import Path
import argparse
import os
import shutil
import zipfile

import pandas as pd

import model_code_from_original as mc
from vlm3d_common import (
    ensure_dir,
    ct_generation_proxy_metrics,
    save_dataframe,
    save_bar_plot,
    save_ct_slices_png,
    write_example_table,
    save_official_metric_note,
)

OFFICIAL_TASK = "ct-volume-generation"


def train(args):
    reports_df = pd.read_csv(args.train_reports_csv)
    train_df = mc.build_task3_report_df(reports_df)

    ae_model, ae_ckpt = mc.train_task3_autoencoder(
        train_df=train_df,
        checkpoint_path=args.autoencoder_checkpoint,
        target_shape=tuple(args.target_shape),
        latent_dim=args.latent_dim,
        base_channels=args.base_channels,
        batch_size=args.batch_size,
        epochs=args.ae_epochs,
        lr=args.ae_lr,
        num_workers=args.num_workers,
        cache_dir=args.cache_dir,
        hf_token=args.hf_token,
        max_rows=args.max_rows,
    )

    mapper_model, mapper_tokenizer, mapper_ckpt_dir = mc.train_task3_report_to_latent(
        train_df=train_df,
        autoencoder_checkpoint_path=args.autoencoder_checkpoint,
        mapper_checkpoint_dir=args.mapper_checkpoint_dir,
        text_encoder_name=args.text_encoder_name,
        target_shape=tuple(args.target_shape),
        latent_dim=args.latent_dim,
        batch_size=args.batch_size,
        epochs=args.mapper_epochs,
        lr=args.mapper_lr,
        num_workers=args.num_workers,
        cache_dir=args.cache_dir,
        hf_token=args.hf_token,
        max_rows=args.max_rows,
        max_text_length=args.max_text_length,
        freeze_text_encoder=args.freeze_text_encoder,
        recon_loss_weight=args.recon_loss_weight,
    )

    print(f"Saved Task 3 autoencoder checkpoint: {ae_ckpt}")
    print(f"Saved Task 3 mapper checkpoint dir: {mapper_ckpt_dir}")
    return ae_ckpt, mapper_ckpt_dir


def predict(args):
    output_dir = ensure_dir(args.output)
    paths = mc.predict_task3_report_to_ct(
        input_path=args.input,
        autoencoder_checkpoint_path=args.autoencoder_checkpoint,
        mapper_checkpoint_dir=args.mapper_checkpoint_dir,
        output_dir=output_dir,
        output_format=args.output_format,
        max_cases=args.max_cases,
    )
    print(f"Wrote {len(paths)} generated CT volumes to {output_dir}")
    return paths


def local_eval(args):
    output_dir = ensure_dir(args.output)
    generated_paths = predict(args)

    save_official_metric_note(
        output_dir,
        task_name="ct_volume_generation",
        official_metrics=["FVD_CTNet", "CT-CLIP Score", "FID_2p5D"],
        local_metrics=["HU min", "HU max", "HU mean", "HU std", "lung fraction", "dense fraction"],
    )

    metrics_df = ct_generation_proxy_metrics(args.output)
    save_dataframe(metrics_df, output_dir / "metrics_ct_generation_proxy.csv")
    if len(metrics_df):
        print(metrics_df.describe(include="all"))
        save_bar_plot(metrics_df.set_index("file")["std"], output_dir / "figures" / "generated_ct_std_by_case.png", "Task 3 generated CT intensity SD", "HU SD")

    prompts = mc.read_task3_prompts(args.input)
    prompt_map = {str(p["case_id"]): str(p["report"]) for p in prompts}
    example_rows = []
    for p in mc.list_ct_files(args.output)[: args.n_show]:
        stem = p.name[:-7] if p.name.endswith(".nii.gz") else p.stem
        png = save_ct_slices_png(p, output_dir / "examples" / f"{stem}_generated_ct.png", title=f"Generated CT: {stem}")
        report_path = output_dir / "examples" / f"{stem}_input_report.txt"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(prompt_map.get(stem, ""))
        example_rows.append({"case_id": stem, "input_report_txt": str(report_path), "generated_ct_png": str(png), "generated_ct_file": str(p)})
    write_example_table(example_rows, output_dir)
    return generated_paths


def make_weights(args):
    weights_dir = ensure_dir(args.weights_dir)
    ae_dest = weights_dir / "task3_autoencoder.pt"
    mapper_dest = weights_dir / "task3_report_to_latent"
    if mapper_dest.exists():
        shutil.rmtree(mapper_dest)
    shutil.copy2(args.autoencoder_checkpoint, ae_dest)
    shutil.copytree(args.mapper_checkpoint_dir, mapper_dest)

    zip_path = Path(args.weights_zip)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for p in weights_dir.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(weights_dir))
    print(f"Wrote {zip_path}; checkpoint files are at ZIP root.")
    return zip_path


def forithmus_process(args):
    args.input = os.environ.get("FORITHMUS_INPUT", "/input")
    args.output = os.environ.get("FORITHMUS_OUTPUT", "/output")
    args.autoencoder_checkpoint = os.environ.get("TASK3_AE", "/weights/task3_autoencoder.pt")
    args.mapper_checkpoint_dir = os.environ.get("TASK3_MAPPER_DIR", "/weights/task3_report_to_latent")
    args.output_format = "nii.gz"
    args.max_cases = None
    return predict(args)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "predict", "local_eval", "make_weights", "forithmus"], default="predict")
    p.add_argument("--input", default="/input")
    p.add_argument("--output", default="/output")
    p.add_argument("--train-reports-csv", default=mc.CT_RATE_CSV_FILES["train"]["reports"])
    p.add_argument("--autoencoder-checkpoint", default="/weights/task3_autoencoder.pt")
    p.add_argument("--mapper-checkpoint-dir", default="/weights/task3_report_to_latent")
    p.add_argument("--output-format", choices=["npy", "nii.gz"], default="nii.gz")
    p.add_argument("--max-cases", type=int, default=None)
    p.add_argument("--target-shape", type=int, nargs=3, default=[64, 96, 96])
    p.add_argument("--latent-dim", type=int, default=512)
    p.add_argument("--base-channels", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-rows", type=int, default=50)
    p.add_argument("--cache-dir", default="./ct_rate_volume_cache")
    p.add_argument("--hf-token", default=os.getenv("HF_TOKEN"))
    p.add_argument("--ae-epochs", type=int, default=3)
    p.add_argument("--mapper-epochs", type=int, default=3)
    p.add_argument("--ae-lr", type=float, default=1e-4)
    p.add_argument("--mapper-lr", type=float, default=1e-4)
    p.add_argument("--text-encoder-name", default="distilbert-base-uncased")
    p.add_argument("--max-text-length", type=int, default=256)
    p.add_argument("--freeze-text-encoder", action="store_true", default=True)
    p.add_argument("--recon-loss-weight", type=float, default=0.25)
    p.add_argument("--n-show", type=int, default=5)
    p.add_argument("--weights-dir", default="weights_task3")
    p.add_argument("--weights-zip", default="weights_task3.zip")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "train":
        train(args)
    elif args.mode == "predict":
        predict(args)
    elif args.mode == "local_eval":
        local_eval(args)
    elif args.mode == "make_weights":
        make_weights(args)
    elif args.mode == "forithmus":
        forithmus_process(args)

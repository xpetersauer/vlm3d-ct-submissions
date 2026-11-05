#!/usr/bin/env python3
"""Compatibility adapter for the recovered Task-2 trainer, not the historical full-data patch.

Adds the external validation-table flags seen in the August run log, removes the
10k row cap, uses the compact CT cache and final inference model definitions.
The recovered original trainer is preserved unchanged under training/original.
"""
import ast
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent


def adapted_source():
    source=(HERE/'original/sota_deadline/train_sota_task2.py').read_text()
    anchor='    ap.add_argument("--out", default="./checkpoints/sota_task2")'
    extra='''    ap.add_argument("--val-reports-csv", required=True)
    ap.add_argument("--val-labels-csv", required=True)
'''
    if source.count(anchor)!=1:raise ValueError('Unsupported recovered trainer version')
    source=source.replace(anchor,extra+anchor)
    source=source.replace('"--max-rows", type=int, default=10000','"--max-rows", type=int, default=None')
    source=source.replace('ap.add_argument("--task1-checkpoint", default=None)',
                          'ap.add_argument("--task1-checkpoint", required=True)')
    start=source.index('    reports = pd.read_csv(args.reports_csv).copy()')
    end=source.index('    train_ds = CTFullReportWithLabels',start)
    block='''    def table(reports_path, labels_path):
        reports = pd.read_csv(reports_path).drop_duplicates("VolumeName")
        labels = pd.read_csv(labels_path).drop_duplicates("VolumeName")
        frame = reports.merge(labels[["VolumeName"] + LABELS], on="VolumeName", how="inner")
        for column in ["Findings_EN", "Impressions_EN"]:
            frame[column] = frame[column].fillna("").astype(str)
        keep = frame["Findings_EN"].str.strip().ne("") & frame["Impressions_EN"].str.strip().ne("")
        return frame[keep].reset_index(drop=True)

    train_df = table(args.reports_csv, args.labels_csv)
    val_df = table(args.val_reports_csv, args.val_labels_csv)
    if args.max_rows is not None:
        train_df = train_df.head(args.max_rows)
    val_df = val_df.head(args.val_cases)
    if len(train_df) == 0 or len(val_df) == 0:
        raise ValueError("Empty training or validation table")
    train_groups = set(train_df["VolumeName"].map(patient_group_from_volume_name))
    valid_groups = set(val_df["VolumeName"].map(patient_group_from_volume_name))
    if train_groups & valid_groups:
        raise ValueError("Training/validation patient overlap")
    print("Compatibility training rows:", len(train_df), "validation rows:", len(val_df), flush=True)

'''
    source=source[:start]+block+source[end:]
    start=source.index('        subfolder, filename = mc.volume_name_to_hf_parts(volume_name)')
    end=source.index('\n\n        findings = ',start)
    source=source[:start]+'        x = mc.get_deadline_cached_ct(volume_name, self.target_shape, None)'+source[end:]
    source=source.replace('        spatial_dims=3,','        spatial_dims=3,\n        img_size=tuple(args.target_shape),',1)
    ast.parse(source)
    return source


if __name__=='__main__':
    project=Path.cwd()
    if not (project/'model_code_from_original.py').is_file():
        raise SystemExit('Run through training/run_stage.py in a prepared workspace.')
    sys.path[:0]=[str(project),str(project/'sota_deadline')]
    exec(compile(adapted_source(),str(HERE/'train_task2_full_data.py'),'exec'),
         {'__name__':'__main__','__file__':str(HERE/'train_task2_full_data.py')})

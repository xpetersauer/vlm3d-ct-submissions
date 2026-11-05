#!/usr/bin/env python3
"""Run a recovered training stage in a separate workspace, with a timestamped log."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

HERE=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--task1-checkpoint',type=Path)
    p.add_argument('--initial-task3-zip',type=Path)
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('stage',choices=['radbert','task1-features','task1-heads','task1-select','baseline-swin','task2-train','task2-rerank','task3-train'])
    a=p.parse_args()
    workspace=a.workspace.expanduser().resolve()
    cfg=json.loads((workspace/'workspace.json').read_text())
    project=Path(cfg['project']);work=Path(cfg['work']);data=Path(cfg['data_root'])
    champion=project/'weeklong_champion/champion'
    py=sys.executable
    labels=data/'dataset/multi_abnormality_labels'
    reports=data/'dataset/radiology_text_reports'
    commands=[]
    if a.stage=='radbert':commands=[[py,str(champion/'radbert_soft_targets.py')]]
    if a.stage=='task1-features':commands=[[py,str(champion/'extract_dino_features.py'),'--n-slices','24','--image-size','224','--image-batch','96']]
    if a.stage=='task1-heads':
        commands=[[py,str(champion/'train_dino_head.py'),'--variant','full','--seed',str(seed),'--epochs','15','--patience','4','--out',str(work/f'dino_full_s{seed}')] for seed in [17,29,43]]
        print('Report soft targets:',(work/'radbert/train_soft_targets.csv').is_file(),flush=True)
    if a.stage=='task1-select':commands=[[py,str(champion/'posthoc_select.py')]]
    if a.stage=='baseline-swin':
        commands=[[py,str(project/'task1_ct_to_abnormality.py'),'--mode','train','--train-labels-csv',str(labels/'train_predicted_labels.csv'),
                   '--checkpoint',str(work/'new_baseline_swin.pt'),'--max-rows','5000','--epochs','5','--target-shape','64','96','96',
                   '--depths','1','1','1','1','--batch-size','1','--feature-size','24']]
        print('This creates a NEW baseline initialization; it is not the historical baseline checkpoint.',flush=True)
    if a.stage=='task2-train':
        if a.task1_checkpoint is None:p.error('task2-train requires --task1-checkpoint with encoder.* keys (a Swin checkpoint, not DINO heads)')
        if not a.task1_checkpoint.expanduser().is_file():p.error('Task-1 initialization checkpoint does not exist')
        commands=[[py,str(HERE/'train_task2_full_data.py'),'--reports-csv',str(reports/'train_reports.csv'),
                  '--labels-csv',str(labels/'train_predicted_labels.csv'),'--val-reports-csv',str(reports/'validation_reports.csv'),
                  '--val-labels-csv',str(labels/'valid_predicted_labels.csv'),'--out',str(project/'checkpoints/sota_task2_ALL_5ep'),
                  '--epochs','5','--batch-size','1','--val-cases','250','--target-shape','64','96','96',
                  '--token-grid','4','6','6','--visual-tokens','32','--feature-size','24','--depths','1','1','1','1',
                  '--num-heads','3','6','12','24','--lm-name','Qwen/Qwen2.5-0.5B-Instruct','--max-target-length','256',
                  '--aux-pathology-weight','0.25','--lr','1e-4','--vision-lr','1e-5','--warmup-ratio','0.05',
                  '--checkpoint-every','5000','--task1-checkpoint',str(a.task1_checkpoint.expanduser().resolve())]]
    if a.stage=='task2-rerank':commands=[[py,str(champion/'task2_rerank.py'),'--max-cases','180','--candidates','4','--max-new-tokens','220']]
    if a.stage=='task3-train':
        root=work/'task3_full_training'
        if a.initial_task3_zip is None:p.error('task3-train requires --initial-task3-zip with baseline weights')
        initial=a.initial_task3_zip.expanduser().resolve()
        if not initial.is_file():p.error('Task-3 initialization ZIP does not exist')
        if not a.dry_run:
            import pandas as pd
            import zipfile
            root.mkdir(parents=True,exist_ok=True)
            checkpoints=root/'checkpoints'
            if not checkpoints.exists():
                with zipfile.ZipFile(initial) as z:
                    for item in z.infolist():
                        parts=Path(item.filename).parts
                        if item.filename.startswith('/') or '..' in parts:raise ValueError('Unsafe initial-weight ZIP')
                    z.extractall(checkpoints)
            rows=len(pd.read_csv(reports/'train_reports.csv'))
        else:rows='FULL_TRAIN_ROWS'
        commands=[[py,str(project/'task3/task3_report_to_ct.py'),'--mode','train','--train-reports-csv',str(reports/'train_reports.csv'),
                   '--autoencoder-checkpoint',str(root/'checkpoints/task3_autoencoder.pt'),
                   '--mapper-checkpoint-dir',str(root/'checkpoints/task3_report_to_latent'),
                   '--ae-epochs','5','--mapper-epochs','5','--ae-lr','1e-4','--mapper-lr','1e-4',
                   '--batch-size','1','--max-rows',str(rows),'--cache-dir',cfg['tensor_cache']],
                  [py,str(project/'task3/task3_report_to_ct.py'),'--mode','make_weights',
                   '--autoencoder-checkpoint',str(root/'checkpoints/task3_autoencoder.pt'),
                   '--mapper-checkpoint-dir',str(root/'checkpoints/task3_report_to_latent'),
                   '--weights-dir',str(root/'packaged_weights'),'--weights-zip',str(root/'task3_weights_retrained.zip')]]
    env=dict(os.environ)
    env['PYTHONPATH']=os.pathsep.join([str(champion),str(project),str(project/'sota_deadline'),env.get('PYTHONPATH','')])
    env['VLM3D_TENSOR_CACHE']=cfg['tensor_cache'];env['VLM3D_RAW_STAGE']=cfg['raw_stage']
    env['TASK3_CHECKPOINT_EVERY_STEPS']='250'
    env['TASK3_TRAIN_PROGRESS_PATH']=str(work/'task3_full_training/progress.json')
    if not a.dry_run:
        import torch
        if not torch.cuda.is_available():raise SystemExit('CUDA is required for these training stages.')
    for command in commands:
        print(shlex.join(command),flush=True)
        if a.dry_run:continue
        logs=workspace/'logs';logs.mkdir(exist_ok=True)
        log=logs/(a.stage+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.log')
        with log.open('w') as output:
            process=subprocess.Popen(command,cwd=project,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            for line in process.stdout:
                print(line,end='',flush=True);output.write(line);output.flush()
            if process.wait():raise SystemExit('Stage failed. See '+str(log))


if __name__=='__main__':main()

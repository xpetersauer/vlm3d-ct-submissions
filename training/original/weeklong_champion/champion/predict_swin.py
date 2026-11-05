#!/usr/bin/env python3
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, torch

PROJECT=Path('/workspace/vlm3d_original_task_split')
sys.path.insert(0,str(Path(__file__).resolve().parent)); sys.path.insert(0,str(PROJECT)); sys.path.insert(0,str(PROJECT/'sota_deadline'))
from common import VALID_LABELS, WORK, load_labels, metrics, save_prediction_npz, write_json, LABELS
from train_structured_swin import build_from_checkpoint, predict


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',required=True); ap.add_argument('--out',required=True); args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    valid=load_labels(VALID_LABELS); y=valid[LABELS].to_numpy(np.float32)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model,_=build_from_checkpoint(Path(args.checkpoint),device)
    names,p=predict(model,valid,device,batch=2)
    save_prediction_npz(out/'valid_predictions.npz',names,p)
    m=metrics(y,p); write_json({k:v for k,v in m.items() if k!='per_label'},out/'valid_metrics.json')
    print('SWIN PREDICTION COMPLETE',args.checkpoint)

if __name__=='__main__': main()

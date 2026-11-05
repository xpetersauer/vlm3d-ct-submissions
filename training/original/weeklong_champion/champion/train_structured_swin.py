#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

PROJECT = Path("/workspace/vlm3d_original_task_split")
sys.path.insert(0, str(PROJECT)); sys.path.insert(0, str(PROJECT / "sota_deadline"))
import model_code_from_original as mc
from sota_models import InteractingLabelTokenClassifier
from transformers import get_cosine_schedule_with_warmup

from common import LABELS, TRAIN_LABELS, VALID_LABELS, WORK, alternate_reconstruction_index, load_labels, metrics, save_prediction_npz, validation_partition, write_json
from models import AsymmetricLoss, DeviceEMA


class StructuredDataset(Dataset):
    def __init__(self, df, alt, soft=None, target_shape=(64,96,96)):
        self.df = df.reset_index(drop=True); self.alt=np.asarray(alt); self.soft=soft; self.target_shape=tuple(target_shape)
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        r=self.df.iloc[int(idx)]; n=str(r.VolumeName)
        x=mc.get_deadline_cached_ct(n,self.target_shape,None)
        y=torch.tensor(r[LABELS].to_numpy(np.float32),dtype=torch.float32)
        q=torch.tensor(self.soft[int(idx)],dtype=torch.float32) if self.soft is not None else torch.full((len(LABELS),),float("nan"))
        return x,y,q,torch.tensor(int(idx),dtype=torch.long),n


class ValDataset(Dataset):
    def __init__(self,df,target_shape=(64,96,96)): self.df=df.reset_index(drop=True); self.target_shape=tuple(target_shape)
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        r=self.df.iloc[i]; n=str(r.VolumeName)
        x=mc.get_deadline_cached_ct(n,self.target_shape,None)
        y=torch.tensor(r[LABELS].to_numpy(np.float32))
        return x,y,n


def checkpoint_candidates(explicit=None):
    p=[]
    if explicit: p.append(Path(explicit))
    p += [
        PROJECT/"checkpoints/sota_task1_ALL_5ep/best.pt",
        PROJECT/"checkpoints/sota_task1_5k_25ep/best.pt",
        PROJECT/"checkpoints/sota_task1_5k/best.pt",
    ]
    return [x for x in p if x.exists()]


def build_from_checkpoint(path, device):
    try: ck=torch.load(path,map_location="cpu",weights_only=False)
    except TypeError: ck=torch.load(path,map_location="cpu")
    extra=ck.get("extra",{})
    vp=dict(extra.get("vision_params",{}))
    if not vp:
        vp=dict(feature_size=24,in_channels=1,out_channels=1,spatial_dims=3,depths=(1,1,1,1),num_heads=(3,6,12,24),drop_rate=0.,attn_drop_rate=0.,dropout_path_rate=.05,use_checkpoint=True)
    vp["img_size"]=(64,96,96)
    model=InteractingLabelTokenClassifier(vp,num_labels=len(LABELS),token_grid=(2,4,4),d_model=384,nhead=8,label_layers=2).to(device)
    with torch.no_grad(): _=model(torch.zeros(1,1,64,96,96,device=device))
    state=ck.get("model",ck)
    miss,unexp=model.load_state_dict(state,strict=False)
    if len(unexp)>10: raise RuntimeError(f"Too many unexpected keys loading {path}: {unexp[:20]}")
    print(f"Initialized from {path}; missing={len(miss)} unexpected={len(unexp)}")
    return model,vp


def load_soft(names):
    p=WORK/"radbert/train_soft_targets.csv"
    if not p.exists(): return None
    d=pd.read_csv(p).drop_duplicates("VolumeName").set_index("VolumeName")
    if any(n not in d.index for n in names): return None
    return d.loc[names,LABELS].to_numpy(np.float32)


@torch.inference_mode()
def predict(model,df,device,batch=2):
    dl=DataLoader(ValDataset(df),batch_size=batch,shuffle=False,num_workers=2,pin_memory=True)
    pp=[]; nn=[]
    model.eval()
    for x,_,names in dl:
        pp.append(torch.sigmoid(model(x.to(device,non_blocking=True))).float().cpu().numpy()); nn.extend(list(names))
    return np.asarray(nn,dtype=str),np.concatenate(pp,0)


def swap_to_ema(model,ema):
    current={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(ema.state_dict(),strict=True)
    return current


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",default=str(WORK/"structured_swin")); ap.add_argument("--init",default=None)
    ap.add_argument("--epochs",type=int,default=3); ap.add_argument("--batch-size",type=int,default=2)
    ap.add_argument("--vision-lr",type=float,default=3e-6); ap.add_argument("--head-lr",type=float,default=3e-5)
    ap.add_argument("--report-weight",type=float,default=.20); ap.add_argument("--recon-weight",type=float,default=.10)
    ap.add_argument("--recon-every",type=int,default=4)
    args=ap.parse_args(); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(17); np.random.seed(17)
    train=load_labels(TRAIN_LABELS); valid=load_labels(VALID_LABELS)
    names=train.VolumeName.astype(str).tolist(); alt=alternate_reconstruction_index(names); soft=load_soft(names)
    print(f"Training rows={len(train):,}; reconstruction-paired rows={(alt>=0).sum():,}; report soft targets={'ON' if soft is not None else 'OFF'}")
    ds=StructuredDataset(train,alt,soft)
    dl=DataLoader(ds,batch_size=args.batch_size,shuffle=True,num_workers=2,pin_memory=True)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cands=checkpoint_candidates(args.init)
    if not cands: raise FileNotFoundError("No SOTA Task1 checkpoint found to initialize structured fine-tune")
    model,vp=build_from_checkpoint(cands[0],device)
    vision_ids={id(p) for p in model.vision.parameters()}
    vision=[p for p in model.vision.parameters() if p.requires_grad]
    other=[p for p in model.parameters() if p.requires_grad and id(p) not in vision_ids]
    opt=torch.optim.AdamW([{"params":vision,"lr":args.vision_lr},{"params":other,"lr":args.head_lr}],weight_decay=1e-2)
    total=max(1,len(dl)*args.epochs); sched=get_cosine_schedule_with_warmup(opt,max(1,int(.05*total)),total)
    lossfn=AsymmetricLoss(4,1,.05); ema=DeviceEMA(model,.999)
    vnames=valid.VolumeName.astype(str).tolist(); vpart=validation_partition(vnames); dev=(vpart=="modeldev") & (np.arange(len(valid))>=500); yv=valid[LABELS].to_numpy(np.float32)
    best=-1.; history=[]; step=0
    for ep in range(args.epochs):
        model.train(); ls=[]
        for x,y,q,idx_batch,_ in dl:
            x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True); opt.zero_grad(set_to_none=True)
            z=model(x); loss=lossfn(z,y)
            if soft is not None:
                q=q.to(device,non_blocking=True); loss=loss+args.report_weight*F.binary_cross_entropy_with_logits(z,q)
            if args.recon_weight>0 and step%args.recon_every==0:
                idx_np=idx_batch.cpu().numpy().astype(int); alt_np=alt[idx_np]; pos=np.where(alt_np>=0)[0]
                if len(pos):
                    xa=torch.stack([mc.get_deadline_cached_ct(names[int(alt_np[k])],(64,96,96),None) for k in pos]).to(device)
                    za=model(xa)
                    pos_t=torch.as_tensor(pos,device=device,dtype=torch.long)
                    loss=loss+args.recon_weight*F.mse_loss(torch.sigmoid(z[pos_t]),torch.sigmoid(za))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); sched.step(); ema.update(model); step+=1; ls.append(float(loss.detach().cpu()))
            if step%2000==0: print(f"epoch={ep+1} step={step:,} loss={np.mean(ls[-200:]):.5f}",flush=True)
        raw=swap_to_ema(model,ema)
        pn,pp=predict(model,valid,device,batch=2); met=metrics(yv[dev],pp[dev])
        row={"epoch":ep+1,"train_loss":float(np.mean(ls)),"modeldev_auroc":met["macro_auroc"],"modeldev_f1":met["macro_f1"],"modeldev_bce":met["bce"]}; history.append(row); print(json.dumps(row),flush=True)
        if met["macro_auroc"]>best:
            best=met["macro_auroc"]
            torch.save({"model":{k:v.detach().cpu() for k,v in model.state_dict().items()},"extra":{"vision_params":vp,"target_shape":(64,96,96),"token_grid":(2,4,4),"labels":LABELS,"training":"ASL+EMA+reconstruction_consistency+optional_report_soft_targets","report_weight":args.report_weight if soft is not None else 0.,"recon_weight":args.recon_weight}},out/"best.pt")
            save_prediction_npz(out/"valid_predictions.npz",pn,pp)
        model.load_state_dict(raw,strict=True)
    pd.DataFrame(history).to_csv(out/"history.csv",index=False)
    if not (out/"valid_predictions.npz").exists(): raise RuntimeError("Structured Swin produced no predictions")
    print("STRUCTURED SWIN COMPLETE",out)

if __name__=="__main__": main()

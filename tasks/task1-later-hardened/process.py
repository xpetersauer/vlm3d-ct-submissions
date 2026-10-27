#!/usr/bin/env python3
from __future__ import annotations
import json, math, os, sys
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel

APP=Path('/opt/app'); WEIGHTS=Path('/weights'); INPUT=Path('/input'); OUTPUT=Path('/output')
sys.path.insert(0,str(APP))
import model_code_from_original as mc
from sota_models import InteractingLabelTokenClassifier
from model_def import QueryPoolClassifier

LABELS=mc.ABNORMALITY_LABELS
VIEWS=[('all',-1000.,1000.),('soft',-150.,250.),('lung',-1000.,400.)]

def logit(p,eps=1e-6):
    p=np.clip(np.asarray(p,float),eps,1-eps); return np.log(p/(1-p))
def sigmoid(x):
    x=np.clip(np.asarray(x,float),-40,40); return 1/(1+np.exp(-x))
def entropy(p):
    p=np.clip(np.asarray(p,float),1e-6,1-1e-6); return -(p*np.log(p)+(1-p)*np.log(1-p))/math.log(2)

def dino_images(x,n_slices,processor,size=224):
    if x.ndim==4:x=x[0]
    inds=torch.linspace(0,x.shape[0]-1,steps=n_slices).round().long().unique(sorted=True)
    hu=x[inds].float()*1000.
    mean=torch.tensor(processor.image_mean); std=torch.tensor(processor.image_std); imgs=[]
    for _,lo,hi in VIEWS:
        v=((hu-lo)/(hi-lo)).clamp(0,1)[:,None].repeat(1,3,1,1)
        v=F.interpolate(v,size=(size,size),mode='bilinear',align_corners=False)
        v=(v-mean[None,:,None,None])/std[None,:,None,None]; imgs.append(v)
    return torch.cat(imgs,0)

class DINOExpert:
    def __init__(self,head_files,device):
        self.device=device; self.base=WEIGHTS/'dinov2-base'
        self.proc=AutoImageProcessor.from_pretrained(str(self.base),local_files_only=True)
        self.dino=AutoModel.from_pretrained(str(self.base),local_files_only=True).to(device).eval()
        if device.type=='cuda': self.dino=self.dino.to(dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
        self.heads=[]; self.n_slices=24
        for f in head_files:
            ck=torch.load(WEIGHTS/f,map_location='cpu')
            h=QueryPoolClassifier(ck.get('feature_dim',768),384,8,len(LABELS)).to(device)
            h.load_state_dict(ck['model'],strict=True); h.eval(); self.heads.append(h); self.n_slices=int(ck.get('n_slices',24))
    @torch.inference_mode()
    def __call__(self,x):
        imgs=dino_images(x,self.n_slices,self.proc); zz=[]
        for s in range(0,len(imgs),96):
            o=self.dino(pixel_values=imgs[s:s+96].to(device=self.device,dtype=next(self.dino.parameters()).dtype)); zz.append(o.last_hidden_state[:,0].float())
        feat=torch.cat(zz,0)[None]
        logits=[h(feat).float().cpu().numpy()[0] for h in self.heads]
        return sigmoid(np.mean(logits,axis=0))

class SwinExpert:
    def __init__(self,file,device):
        ck=torch.load(WEIGHTS/file,map_location='cpu',weights_only=False); ex=ck.get('extra',{}); vp=dict(ex.get('vision_params',{})); vp['img_size']=(64,96,96)
        self.model=InteractingLabelTokenClassifier(vp,len(LABELS),(2,4,4),384,8,2).to(device)
        with torch.no_grad(): _=self.model(torch.zeros(1,1,64,96,96,device=device))
        self.model.load_state_dict(ck.get('model',ck),strict=False); self.model.eval(); self.device=device
    @torch.inference_mode()
    def __call__(self,x): return torch.sigmoid(self.model(x[None].to(self.device))).float().cpu().numpy()[0]

def main():
    OUTPUT.mkdir(parents=True,exist_ok=True); recipe=json.loads((WEIGHTS/'final_recipe.json').read_text()); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base=recipe['base']; dino=None; swin=None
    if base['kind']=='single':
        if base['candidate'].startswith('dino'): dino=DINOExpert(recipe['resolved']['dino_heads'],device)
        else: swin=SwinExpert(recipe['resolved']['swin'],device)
    elif base['kind']=='blend':
        dino=DINOExpert(recipe['resolved']['dino_heads'],device); swin=SwinExpert(recipe['resolved']['swin'],device)
    else: raise ValueError(base)
    g=recipe['graph']; prior=np.asarray(g['prior']); W=np.asarray(g['W']); lam=np.asarray(g['lambda_by_label']); cal=recipe['calibration']; a=np.asarray(cal['a']); b=np.asarray(cal['b'])
    preds=[]
    paths=sorted(list(INPUT.rglob('*.nii.gz'))+list(INPUT.rglob('*.nii')))
    if not paths: raise FileNotFoundError('/input contains no NIfTI volumes')
    for path in paths:
        x=mc.preprocess_ct_for_torch(path,(64,96,96))
        if base['kind']=='single': p=dino(x) if dino is not None else swin(x)
        else:
            pdn=dino(x); ps=swin(x); alpha=np.asarray(base['dino_alpha_by_label']); p=sigmoid(alpha*logit(pdn)+(1-alpha)*logit(ps))
        if g.get('accepted',False):
            msg=(p-prior)@W.T; p=sigmoid(logit(p)+entropy(p)*msg*lam)
        if cal.get('accepted',False): p=sigmoid(logit(p)*a+b)
        case=mc.strip_ct_extension(path.name); preds.append({'input_image_name':case,'probabilities':{lab:float(np.clip(v,0,1)) for lab,v in zip(LABELS,p)}})
        print('classified',case,flush=True)
    obj={'name':'Generated probabilities','type':'Abnormality Classification','version':{'major':1,'minor':0},'predictions':preds}
    (OUTPUT/'results.json').write_text(json.dumps(obj,indent=2))

if __name__=='__main__': main()

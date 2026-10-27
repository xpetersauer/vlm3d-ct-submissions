#!/usr/bin/env python3
from __future__ import annotations
import json, re, sys
from pathlib import Path
import joblib, numpy as np, pandas as pd, torch, torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

APP=Path('/opt/app'); WEIGHTS=Path('/weights'); INPUT=Path('/input'); OUTPUT=Path('/output')
sys.path.insert(0,str(APP))
import model_code_from_original as mc
from sota_models import CTCausalReportGenerator
LABELS=mc.ABNORMALITY_LABELS

class RadBertClassifier(nn.Module):
    def __init__(self,base,n_classes=18):
        super().__init__(); cfg=AutoConfig.from_pretrained(str(base),local_files_only=True); self.model=AutoModel.from_pretrained(str(base),config=cfg,local_files_only=True); self.classifier=nn.Linear(self.model.config.hidden_size,n_classes)
    def forward(self,input_ids,attention_mask):
        o=self.model(input_ids=input_ids,attention_mask=attention_mask); pooled=getattr(o,'pooler_output',None)
        if pooled is None: pooled=o.last_hidden_state[:,0]
        return self.classifier(pooled)

def repetition_ratio(text):
    toks=re.findall(r'\w+',str(text).lower())
    if len(toks)<4:return 0.0
    ng=[tuple(toks[i:i+4]) for i in range(len(toks)-3)]
    return 1-len(set(ng))/max(1,len(ng))

def load_generator(device):
    ck=torch.load(WEIGHTS/'task2_best.pt',map_location='cpu',weights_only=False); ex=ck.get('extra',{}); vp=dict(ex.get('vision_params',{})); vp['img_size']=(64,96,96)
    model=CTCausalReportGenerator(vp,lm_name=str(WEIGHTS/'qwen2.5-0.5b-instruct'),token_grid=tuple(ex.get('token_grid',(4,6,6))),num_visual_tokens=int(ex.get('visual_tokens',32)),aux_pathology_weight=.25,freeze_vision=True,use_lora=True,gradient_checkpointing=False).to(device)
    with torch.no_grad(): _=model.encode_visual(torch.zeros(1,1,64,96,96,device=device))
    model.load_state_dict(ck.get('model',ck),strict=False); model.eval(); return model

def load_reranker(device):
    cfg=json.loads((WEIGHTS/'reranker.json').read_text()) if (WEIGHTS/'reranker.json').exists() else {'accepted':False}
    if not cfg.get('accepted',False): return cfg,None,None,None
    pipe=joblib.load(WEIGHTS/'reranker.joblib'); base=WEIGHTS/'RadBERT-RoBERTa-4m'; tok=AutoTokenizer.from_pretrained(str(base),local_files_only=True,do_lower_case=True); rb=RadBertClassifier(base,len(LABELS))
    state=torch.load(WEIGHTS/'RadBertClassifier.pth',map_location='cpu',weights_only=True); _rb_incompat=rb.load_state_dict(state,strict=False); assert not _rb_incompat.missing_keys and set(_rb_incompat.unexpected_keys) <= {'model.embeddings.position_ids'}, f'RadBERT checkpoint incompatibility: {_rb_incompat}'; print('[task2-weights] VLM3D_TASK2_RADBERT_POSITION_IDS_COMPAT_V1', _rb_incompat); rb.to(device).eval(); return cfg,pipe,tok,rb

@torch.inference_mode()
def radbert_probs(texts,tok,rb,device):
    e=tok(texts,return_tensors='pt',max_length=512,truncation=True,padding=True); return torch.sigmoid(rb(e['input_ids'].to(device),e['attention_mask'].to(device))).float().cpu().numpy()

def main():
    OUTPUT.mkdir(parents=True,exist_ok=True); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); gen=load_generator(device); cfg,reranker,rbtok,rb=load_reranker(device)
    preds=[]; paths=sorted(list(INPUT.rglob('*.nii.gz'))+list(INPUT.rglob('*.nii')))
    if not paths: raise FileNotFoundError('/input contains no NIfTI volumes')
    for path in paths:
        x=mc.preprocess_ct_for_torch(path,(64,96,96))[None].to(device)
        n=int(cfg.get('n_candidates',1)) if cfg.get('accepted',False) else 1
        texts,imagep=gen.generate(x,max_new_tokens=int(cfg.get('max_new_tokens',220)),num_beams=max(1,n),num_return_sequences=max(1,n)); chosen=texts[0]
        if cfg.get('accepted',False) and len(texts)>1:
            tp=radbert_probs(texts,rbtok,rb,device); im=imagep[0].float().cpu().numpy(); rows=[]
            for rank,(t,q) in enumerate(zip(texts,tp)):
                rows.append({'agreement_bce':float(-(im*np.log(np.clip(q,1e-6,1))+(1-im)*np.log(np.clip(1-q,1e-6,1))).mean()),'agreement_l1':float(np.abs(im-q).mean()),'text_confidence':float(np.abs(q-.5).mean()*2),'length_tokens':len(re.findall(r'\w+',t)),'repetition4':repetition_ratio(t),'rank':rank})
            frame=pd.DataFrame(rows)[cfg['features']]; score=reranker.predict(frame); chosen=texts[int(np.argmax(score))]
        preds.append({'input_image_name':mc.strip_ct_extension(path.name),'report':str(chosen).strip()}); print('generated',path.name,flush=True)
    obj=[{'outputs':[{'value':{'name':'Generated reports','type':'Report generation','version':'1.0','generated_reports':preds}}]}]
    (OUTPUT/'predictions.json').write_text(json.dumps(obj,indent=2))
if __name__=='__main__':main()

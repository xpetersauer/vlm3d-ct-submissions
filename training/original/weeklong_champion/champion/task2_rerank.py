#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, os, re, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from huggingface_hub import snapshot_download
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT=Path('/workspace/vlm3d_original_task_split')
sys.path.insert(0,str(PROJECT)); sys.path.insert(0,str(PROJECT/'sota_deadline')); sys.path.insert(0,str(Path(__file__).resolve().parent))
import model_code_from_original as mc
from sota_models import CTCausalReportGenerator
from common import LABELS, VALID_LABELS, VALID_REPORTS, WORK, full_report_text, load_labels, patient_key, deterministic_bucket, write_json
from radbert_soft_targets import load_radbert


def rouge_l_f1(a,b):
    x=re.findall(r"\w+",str(a).lower()); y=re.findall(r"\w+",str(b).lower())
    if not x or not y: return 0.0
    prev=[0]*(len(y)+1)
    for xi in x:
        cur=[0]
        for j,yj in enumerate(y,1): cur.append(prev[j-1]+1 if xi==yj else max(prev[j],cur[-1]))
        prev=cur
    l=prev[-1]; p=l/len(x); r=l/len(y)
    return 2*p*r/(p+r+1e-12)


def repetition_ratio(text):
    toks=re.findall(r"\w+",str(text).lower())
    if len(toks)<4:return 0.0
    ng=[tuple(toks[i:i+4]) for i in range(len(toks)-3)]
    return 1-len(set(ng))/max(1,len(ng))


def case_f1(y,p):
    pred=(p>=.5).astype(int); y=np.asarray(y,int)
    tp=(pred*y).sum(); fp=(pred*(1-y)).sum(); fn=((1-pred)*y).sum()
    return float(2*tp/(2*tp+fp+fn+1e-12))


def build_generator(ckpt_path,device):
    qdir=WORK/'assets/qwen2.5-0.5b-instruct'
    if not (qdir/'config.json').exists(): snapshot_download('Qwen/Qwen2.5-0.5B-Instruct',local_dir=str(qdir),token=os.getenv('HF_TOKEN') or None)
    try: ck=torch.load(ckpt_path,map_location='cpu',weights_only=False)
    except TypeError: ck=torch.load(ckpt_path,map_location='cpu')
    ex=ck.get('extra',{}); vp=dict(ex.get('vision_params',{})); vp['img_size']=(64,96,96)
    model=CTCausalReportGenerator(
        vision_params=vp,lm_name=str(qdir),token_grid=tuple(ex.get('token_grid',(4,6,6))),
        num_visual_tokens=int(ex.get('visual_tokens',32)),aux_pathology_weight=.25,
        freeze_vision=True,use_lora=True,gradient_checkpointing=False,
    ).to(device)
    with torch.no_grad(): _=model.encode_visual(torch.zeros(1,1,64,96,96,device=device))
    missing,unexpected=model.load_state_dict(ck.get('model',ck),strict=False)
    print('Task2 load missing/unexpected',len(missing),len(unexpected))
    model.eval(); return model,qdir


@torch.inference_mode()
def radbert_probs(texts,tok,model,device,batch=64):
    out=[]
    for s in range(0,len(texts),batch):
        enc=tok(texts[s:s+batch],return_tensors='pt',max_length=512,truncation=True,padding=True)
        z=model(enc['input_ids'].to(device),enc['attention_mask'].to(device))
        out.append(torch.sigmoid(z).float().cpu().numpy())
    return np.concatenate(out,0)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--max-cases',type=int,default=180); ap.add_argument('--candidates',type=int,default=4); ap.add_argument('--max-new-tokens',type=int,default=220); args=ap.parse_args()
    out=WORK/'task2_rerank'; out.mkdir(parents=True,exist_ok=True)
    ckpts=[PROJECT/'checkpoints/sota_task2_ALL_5ep/best.pt',PROJECT/'checkpoints/sota_task2_5k/best.pt']
    ckpt=next((p for p in ckpts if p.exists()),None)
    if ckpt is None: raise FileNotFoundError('No SOTA Task2 checkpoint')
    labels=load_labels(VALID_LABELS); reports=pd.read_csv(VALID_REPORTS).drop_duplicates('VolumeName')
    df=labels.merge(reports,on='VolumeName',how='inner'); df['reference_report']=full_report_text(df)
    # The all-data Task2 trainer used the first 250 eligible official validation rows for checkpoint selection.
    # Exclude them from reranker fitting/checking so the reranker is developed on different validation cases.
    f=df['Findings_EN'].fillna('').astype(str).str.strip().ne(''); i=df['Impressions_EN'].fillna('').astype(str).str.strip().ne('')
    eligible=df[f&i].reset_index(drop=True); used_for_generator_val=set(eligible.head(250).VolumeName.astype(str))
    pool=eligible[~eligible.VolumeName.astype(str).isin(used_for_generator_val)].copy()
    pool['bucket']=pool.VolumeName.astype(str).map(lambda x:deterministic_bucket(patient_key(x),'task2-rerank')/10000)
    dev=pool[pool.bucket<.67].head(max(1,int(args.max_cases*.67))); check=pool[pool.bucket>=.67].head(max(1,args.max_cases-len(dev)))
    use=pd.concat([dev.assign(rsplit='dev'),check.assign(rsplit='check')],ignore_index=True)
    if len(use)<40: raise RuntimeError('Too few reranker development cases')
    print('Reranker cases',use.rsplit.value_counts().to_dict())

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); gen,qdir=build_generator(ckpt,device)
    all_text=[]; meta=[]; image_prob_by_case={}
    for ix,row in use.iterrows():
        name=str(row.VolumeName); x=mc.get_deadline_cached_ct(name,(64,96,96),None)[None].to(device)
        texts,image_p=gen.generate(x,max_new_tokens=args.max_new_tokens,num_beams=args.candidates,num_return_sequences=args.candidates)
        image_prob_by_case[name]=image_p[0].float().cpu().numpy()
        for rank,t in enumerate(texts):
            all_text.append(str(t)); meta.append((ix,name,row.rsplit,rank))
        if (ix+1)%10==0: print(f'Generated candidates for {ix+1}/{len(use)} cases',flush=True)

    rbtok,rb=load_radbert(WORK/'radbert/assets',device); tp=radbert_probs(all_text,rbtok,rb,device)
    records=[]
    for k,(ix,name,split,rank) in enumerate(meta):
        row=use.loc[ix]; im=np.asarray(image_prob_by_case[name],float); txt=tp[k]; ref=row.reference_report; y=row[LABELS].to_numpy(np.float32)
        clinical=case_f1(y,txt); rouge=rouge_l_f1(all_text[k],ref); utility=.70*clinical+.30*rouge
        records.append({
            'case':name,'split':split,'rank':rank,'candidate':all_text[k],
            'agreement_bce':float(-(im*np.log(np.clip(txt,1e-6,1))+(1-im)*np.log(np.clip(1-txt,1e-6,1))).mean()),
            'agreement_l1':float(np.abs(im-txt).mean()),'text_confidence':float(np.abs(txt-.5).mean()*2),
            'length_tokens':len(re.findall(r"\w+",all_text[k])),'repetition4':repetition_ratio(all_text[k]),
            'clinical_case_f1':clinical,'rouge_l_f1':rouge,'utility':utility,
        })
    d=pd.DataFrame(records); feats=['agreement_bce','agreement_l1','text_confidence','length_tokens','repetition4','rank']
    tr=d.split=='dev'; model=make_pipeline(StandardScaler(),Ridge(alpha=10.0)); model.fit(d.loc[tr,feats],d.loc[tr,'utility'])
    d['rerank_score']=model.predict(d[feats])

    def chosen(frame,scorecol):
        ids=frame.groupby('case')[scorecol].idxmax(); return frame.loc[ids]
    chk=d[d.split=='check']; baseline=chosen(chk.assign(beam0=-chk['rank'].astype(float)),'beam0'); reranked=chosen(chk,'rerank_score')
    b=float(baseline.utility.mean()); r=float(reranked.utility.mean()); accepted=(r>b+.002)
    print(f'Reranker check utility: beam0={b:.5f} reranked={r:.5f} accepted={accepted}')
    # Export scaler/ridge coefficients in raw standardized form by serializing the small sklearn object via joblib.
    import joblib; joblib.dump(model,out/'reranker.joblib')
    d.to_csv(out/'candidate_scores.csv',index=False)
    cfg={'accepted':bool(accepted),'checkpoint':str(ckpt),'qwen_asset':str(qdir),'features':feats,'n_candidates':args.candidates,'max_new_tokens':args.max_new_tokens,'check_beam0_utility':b,'check_reranked_utility':r,'development_note':'First 250 eligible official validation cases excluded because generator checkpoint selection used them.'}
    write_json(cfg,out/'reranker.json')
    with (out/'candidates.jsonl').open('w') as f:
        for rec in records: f.write(json.dumps(rec)+'\n')
    print('TASK2 RERANK STAGE COMPLETE')

if __name__=='__main__': main()

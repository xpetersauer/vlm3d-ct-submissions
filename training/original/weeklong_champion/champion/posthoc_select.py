#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score

from common import (
    LABELS, TRAIN_LABELS, VALID_LABELS, WORK, binary_entropy, load_labels, load_prediction_npz,
    logit, metrics, save_prediction_npz, sigmoid, validation_partition, write_json,
)


def score_tuple(m):
    return (m["macro_auroc"], m["macro_f1"], -m["bce"])


def subset_metrics(y,p,mask):
    return metrics(y[mask],p[mask])


def load_candidates(valid_names):
    specs = {
        "baseline_swin": WORK/"baseline_swin/valid_predictions.npz",
        "structured_swin": WORK/"structured_swin/valid_predictions.npz",
        "dino_bce": WORK/"dino_bce/valid_predictions.npz",
        "dino_asl": WORK/"dino_asl/valid_predictions.npz",
        "dino_full_s17": WORK/"dino_full_s17/valid_predictions.npz",
        "dino_full_s29": WORK/"dino_full_s29/valid_predictions.npz",
        "dino_full_s43": WORK/"dino_full_s43/valid_predictions.npz",
    }
    c = {}
    for name,path in specs.items():
        if path.exists():
            _,p=load_prediction_npz(path,valid_names); c[name]=p
    full=[c[k] for k in ["dino_full_s17","dino_full_s29","dino_full_s43"] if k in c]
    if len(full)>=2:
        # Average in logit space to preserve confident agreement while diversifying seeds.
        c["dino_full_ensemble"] = sigmoid(np.mean([logit(x) for x in full],axis=0))
    if not c: raise RuntimeError("No validation prediction candidates found")
    return c


def train_graph(train_y, min_joint=25, prior_strength=100.0, topk=6):
    y=np.asarray(train_y,float); n=len(y); prior=(y.sum(0)+1)/(n+2)
    W=np.zeros((len(LABELS),len(LABELS)),dtype=np.float64)
    for a in range(len(LABELS)):
        for b in range(len(LABELS)):
            if a==b: continue
            nb=y[:,b].sum(); joint=(y[:,a]*y[:,b]).sum()
            if nb<min_joint or joint<min_joint: continue
            pab=(joint+1)/(nb+2)
            raw=logit(pab)-logit(prior[a])
            shrink=joint/(joint+prior_strength)
            W[a,b]=float(raw*shrink)
        inds=np.argsort(np.abs(W[a]))[::-1]
        keep=set(inds[:topk])
        for b in range(len(LABELS)):
            if b not in keep: W[a,b]=0
        denom=np.abs(W[a]).sum()
        if denom>0: W[a]/=denom
    return prior,W


def apply_graph(p,prior,W,lambdas):
    z=logit(p); msg=(p-prior[None,:])@W.T; gate=binary_entropy(p)
    return sigmoid(z + gate*msg*lambdas[None,:])


def fit_graph_lambdas(y,p,mask,prior,W):
    grid=np.array([0,.10,.20,.35,.50,.75,1.0,1.5],float)
    z=logit(p); msg=(p-prior[None,:])@W.T; gate=binary_entropy(p)
    lamb=np.zeros(len(LABELS),float)
    yy=y[mask]
    for j in range(len(LABELS)):
        best=(1e9,0.0)
        for lam in grid:
            pp=sigmoid(z[mask,j] + lam*gate[mask,j]*msg[mask,j])
            eps=1e-6; q=np.clip(pp,eps,1-eps)
            bce=float(-(yy[:,j]*np.log(q)+(1-yy[:,j])*np.log(1-q)).mean())
            # Tiny F1 preference only breaks almost-equal likelihood fits.
            f1=float(f1_score(yy[:,j],pp>=.5,zero_division=0))
            objective=bce-.005*f1
            if objective<best[0]: best=(objective,lam)
        lamb[j]=best[1]
    return lamb


def fit_platt(y,p,mask):
    a=np.ones(len(LABELS),float); b=np.zeros(len(LABELS),float)
    x=logit(p)
    for j in range(len(LABELS)):
        yy=y[mask,j]
        if yy.sum()<10 or (1-yy).sum()<10: continue
        lr=LogisticRegression(C=.5,solver="lbfgs",max_iter=1000)
        lr.fit(x[mask,j,None],yy)
        a[j]=float(np.clip(lr.coef_[0,0],.25,4.0)); b[j]=float(np.clip(lr.intercept_[0],-3,3))
    return a,b


def apply_cal(p,a,b): return sigmoid(logit(p)*a[None,:]+b[None,:])


def main():
    out=WORK/"posthoc"; out.mkdir(parents=True,exist_ok=True)
    train=load_labels(TRAIN_LABELS); valid=load_labels(VALID_LABELS)
    names=valid.VolumeName.astype(str).tolist(); y=valid[LABELS].to_numpy(np.int32); part=validation_partition(names); part[:min(500,len(part))]="baseline_seen"
    c=load_candidates(names)

    rows=[]
    for name,p in c.items():
        for split in ["modeldev","selection","graph","calibration","audit","all"]:
            mask=np.ones(len(y),bool) if split=="all" else part==split
            m=subset_metrics(y,p,mask); rows.append({"candidate":name,"split":split,**{k:v for k,v in m.items() if k!="per_label"}})
    pd.DataFrame(rows).to_csv(out/"candidate_metrics.csv",index=False)

    sel=part=="selection"
    best_name=max(c,key=lambda k:score_tuple(subset_metrics(y,c[k],sel)))
    base=c[best_name].copy(); base_components={"kind":"single","candidate":best_name}
    print("Best individual on selection:",best_name,subset_metrics(y,base,sel))

    # Cheap model-diversity test: structured/legacy Swin + DINO full ensemble.
    dino=c.get("dino_full_ensemble")
    swin=c.get("structured_swin",c.get("baseline_swin"))
    if dino is not None and swin is not None:
        grid=np.array([0,.25,.50,.75,1.0])
        global_trials=[]
        for alpha in grid:
            pp=sigmoid(alpha*logit(dino)+(1-alpha)*logit(swin))
            global_trials.append((score_tuple(subset_metrics(y,pp,sel)),alpha,pp))
        _,global_alpha,_=max(global_trials,key=lambda x:x[0])
        alphas=np.full(len(LABELS),global_alpha,float)
        for j in range(len(LABELS)):
            yy=y[sel,j]
            if yy.sum()<15 or (1-yy).sum()<15: continue
            bestj=(-1e9,global_alpha)
            for alpha in grid:
                pp=sigmoid(alpha*logit(dino[:,j])+(1-alpha)*logit(swin[:,j]))
                auc=roc_auc_score(yy,pp[sel]) if len(np.unique(yy))>1 else 0
                if auc>bestj[0]: bestj=(auc,alpha)
            # shrink per-label choice toward global alpha to reduce public-validation overfit
            alphas[j]=.5*bestj[1]+.5*global_alpha
        blend=sigmoid(alphas[None,:]*logit(dino)+(1-alphas[None,:])*logit(swin))
        mb=subset_metrics(y,base,sel); me=subset_metrics(y,blend,sel)
        if me["macro_auroc"] >= mb["macro_auroc"]+.002 and me["macro_f1"] >= mb["macro_f1"]-.01:
            base=blend; best_name="swin_dino_blend"; base_components={"kind":"blend","dino":"dino_full_ensemble","swin":"structured_swin" if "structured_swin" in c else "baseline_swin","dino_alpha_by_label":alphas.tolist()}
            print("Accepted Swin+DINO blend",me)
        else:
            print("Rejected Swin+DINO blend",me,"vs",mb)

    prior,W=train_graph(train[LABELS].to_numpy(np.float32))
    gmask=part=="graph"; lamb=fit_graph_lambdas(y,base,gmask,prior,W); graphed=apply_graph(base,prior,W,lamb)
    mg0=subset_metrics(y,base,gmask); mg1=subset_metrics(y,graphed,gmask)
    # fail-safe: because zero is in the per-label grid this should usually not regress BCE,
    # but reject globally if both threshold and likelihood metrics deteriorate.
    graph_accepted=not (mg1["bce"]>mg0["bce"]+.002 and mg1["macro_f1"]<mg0["macro_f1"]-.005)
    if not graph_accepted:
        lamb[:]=0; graphed=base.copy()
    print("Graph accepted:",graph_accepted,"before",mg0,"after",mg1)

    cmask=part=="calibration"; a,b=fit_platt(y,graphed,cmask); final=apply_cal(graphed,a,b)
    # calibration safeguard on its own development partition
    mc0=subset_metrics(y,graphed,cmask); mc1=subset_metrics(y,final,cmask)
    calibration_accepted = mc1["bce"] <= mc0["bce"]+.002
    if not calibration_accepted:
        a[:]=1; b[:]=0; final=graphed.copy()

    audit=part=="audit"; auditm=subset_metrics(y,final,audit); allm=metrics(y,final)
    pd.DataFrame(allm["per_label"]).to_csv(out/"final_all_per_label.csv",index=False)
    pd.DataFrame(metrics(y[audit],final[audit])["per_label"]).to_csv(out/"audit_per_label.csv",index=False)
    save_prediction_npz(out/"final_validation_predictions.npz",names,final)

    candidate_paths={
        "baseline_swin":str(WORK/"baseline_swin/valid_predictions.npz"),
        "structured_swin":str(WORK/"structured_swin/valid_predictions.npz"),
        "dino_bce":str(WORK/"dino_bce/best.pt"),
        "dino_asl":str(WORK/"dino_asl/best.pt"),
        "dino_full_s17":str(WORK/"dino_full_s17/best.pt"),
        "dino_full_s29":str(WORK/"dino_full_s29/best.pt"),
        "dino_full_s43":str(WORK/"dino_full_s43/best.pt"),
    }
    source_file=WORK/"baseline_swin/source_checkpoint.txt"
    baseline_ckpt=source_file.read_text().strip() if source_file.exists() else str(PROJECT/"checkpoints/sota_task1_ALL_5ep/best.pt")
    recipe={
        "version":1,"labels":LABELS,"selected_name":best_name,"base":base_components,
        "candidate_paths":candidate_paths,
        "swin_checkpoints":{"baseline":baseline_ckpt,"structured":str(WORK/"structured_swin/best.pt")},
        "dino_asset":str(WORK/"assets/dinov2-base"),
        "graph":{"accepted":bool(graph_accepted),"prior":prior.tolist(),"W":W.tolist(),"lambda_by_label":lamb.tolist()},
        "calibration":{"accepted":bool(calibration_accepted),"a":a.tolist(),"b":b.tolist()},
        "audit_metrics":{k:v for k,v in auditm.items() if k!="per_label"},
        "all_validation_metrics":{k:v for k,v in allm.items() if k!="per_label"},
        "validation_partition_counts":{x:int((part==x).sum()) for x in np.unique(part)},
    }
    write_json(recipe,out/"final_recipe.json")
    print("FINAL TASK1 AUDIT",recipe["audit_metrics"])
    print("POSTHOC SELECTION COMPLETE")

if __name__=="__main__": main()

#!/usr/bin/env python3
from __future__ import annotations

import hashlib, json, os, shutil, zipfile
from pathlib import Path
import pandas as pd
from huggingface_hub import snapshot_download

from common import WORK, ROOT, write_json

BUNDLE=Path(__file__).resolve().parents[1]
DEL=WORK/'deliverables'


def cp(src,dst):
    src=Path(src); dst=Path(dst); dst.parent.mkdir(parents=True,exist_ok=True)
    if src.is_dir():
        if dst.exists(): shutil.rmtree(dst)
        shutil.copytree(src,dst)
    else: shutil.copy2(src,dst)


def zip_contents(root:Path,out:Path):
    if out.exists(): out.unlink()
    with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as z:
        for p in sorted(root.rglob('*')):
            if p.is_file(): z.write(p,p.relative_to(root))


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()


def resolve_task1():
    recipe=json.loads((WORK/'posthoc/final_recipe.json').read_text())
    src=DEL/'task1_source'; weights=DEL/'task1_weights_raw'
    cp(BUNDLE/'templates/task1',src)
    if weights.exists(): shutil.rmtree(weights)
    weights.mkdir(parents=True)
    resolved={'dino_heads':[],'swin':None}
    base=recipe['base']
    def add_dino(candidate):
        mapping={
            'dino_bce':WORK/'dino_bce/best.pt','dino_asl':WORK/'dino_asl/best.pt',
            'dino_full_s17':WORK/'dino_full_s17/best.pt','dino_full_s29':WORK/'dino_full_s29/best.pt','dino_full_s43':WORK/'dino_full_s43/best.pt',
        }
        if candidate=='dino_full_ensemble':
            choices=[mapping[k] for k in ['dino_full_s17','dino_full_s29','dino_full_s43'] if mapping[k].exists()]
        else: choices=[mapping[candidate]] if candidate in mapping and mapping[candidate].exists() else []
        if not choices: raise FileNotFoundError(f'No DINO head checkpoint for {candidate}')
        for i,p in enumerate(choices):
            name=f'dino_head_{i}.pt'; cp(p,weights/name); resolved['dino_heads'].append(name)
        cp(WORK/'assets/dinov2-base',weights/'dinov2-base')
    def add_swin(candidate):
        p=WORK/'structured_swin/best.pt' if candidate=='structured_swin' else Path(recipe['swin_checkpoints']['baseline'])
        if not p.exists(): raise FileNotFoundError(p)
        cp(p,weights/'swin_best.pt'); resolved['swin']='swin_best.pt'
    if base['kind']=='single':
        c=base['candidate']; add_dino(c) if c.startswith('dino') else add_swin(c)
    elif base['kind']=='blend':
        add_dino(base['dino']); add_swin(base['swin'])
    else: raise ValueError(base)
    recipe['resolved']=resolved; write_json(recipe,weights/'final_recipe.json')
    zip_contents(weights,DEL/'task1_weights.zip')


def ensure_qwen():
    q=WORK/'assets/qwen2.5-0.5b-instruct'
    if not (q/'config.json').exists(): snapshot_download('Qwen/Qwen2.5-0.5B-Instruct',local_dir=str(q),token=os.getenv('HF_TOKEN') or None)
    return q


def resolve_task2():
    ck=ROOT/'checkpoints/sota_task2_ALL_5ep/best.pt'
    if not ck.exists():
        ck=ROOT/'checkpoints/sota_task2_5k/best.pt'
    if not ck.exists():
        print('Task2 skipped: no checkpoint'); return False
    src=DEL/'task2_source'; weights=DEL/'task2_weights_raw'; cp(BUNDLE/'templates/task2',src)
    if weights.exists(): shutil.rmtree(weights)
    weights.mkdir(parents=True); cp(ck,weights/'task2_best.pt'); cp(ensure_qwen(),weights/'qwen2.5-0.5b-instruct')
    rr=WORK/'task2_rerank/reranker.json'
    if rr.exists():
        cfg=json.loads(rr.read_text()); cp(rr,weights/'reranker.json')
        if cfg.get('accepted',False):
            cp(WORK/'task2_rerank/reranker.joblib',weights/'reranker.joblib')
            cp(WORK/'radbert/assets/RadBERT-RoBERTa-4m',weights/'RadBERT-RoBERTa-4m')
            cp(WORK/'radbert/assets/RadBertClassifier.pth',weights/'RadBertClassifier.pth')
    zip_contents(weights,DEL/'task2_weights.zip'); return True


def resolve_task3():
    source=ROOT/'forithmus_task3_docker'; wz=ROOT/'weights_task3_DEADLINE_5k.zip'
    if source.exists(): cp(source,DEL/'task3_source_baseline')
    if wz.exists(): cp(wz,DEL/'task3_weights_baseline.zip')


def write_results():
    recipe=json.loads((WORK/'posthoc/final_recipe.json').read_text())
    lines=['# VLM3D Weeklong Champion Results','',
           'This bundle was generated automatically after the all-data baseline completed. External real datasets and synthetic data were intentionally **not** used in this queued run; the goal was to isolate architecture/training/fusion effects and stay within the low compute budget.','',
           '## Task 1 method','',
           '- Baseline: completed all-data interacting-label-token Swin model.',
           '- Foundation branch: frozen DINOv2 slice features with learned label-query attention pooling. This is **AnyMC3D-inspired**, not an exact AnyMC3D reproduction: it deliberately omits full-resolution LoRA backbone adaptation to reuse the compact CT cache and control cost.',
           '- Training ablations: BCE; ASL+EMA; ASL+EMA+report-derived soft targets+same-scan reconstruction consistency.',
           '- Structured Swin fine-tune: ASL + EMA + same-scan reconstruction consistency + report soft targets when official RadBERT assets were available.',
           '- Posthoc novelty: directed conditional label graph derived from training-set P(A), P(A|B), and joint support, gated by prediction entropy; calibration uses a separate patient-group validation partition.',
           '- Optional ensemble is accepted only when its separate selection partition improves enough to justify extra inference cost.','']
    lines += ['## Selected Task 1 recipe','',f"- Selected: `{recipe['selected_name']}`",f"- Graph accepted: `{recipe['graph']['accepted']}`",f"- Calibration accepted: `{recipe['calibration']['accepted']}`",f"- Audit metrics: `{json.dumps(recipe['audit_metrics'])}`",'']
    cm=WORK/'posthoc/candidate_metrics.csv'
    if cm.exists():
        d=pd.read_csv(cm); sub=d[d.split.isin(['selection','audit','all'])]
        lines += ['## Candidate metrics (CSV-style)','', '```', sub.to_csv(index=False).strip(), '```','']
    rr=WORK/'task2_rerank/reranker.json'
    if rr.exists():
        cfg=json.loads(rr.read_text()); lines += ['## Task 2 generate-and-rerank','',f"- Accepted: `{cfg.get('accepted')}`",f"- Beam-0 check utility: `{cfg.get('check_beam0_utility')}`",f"- Reranked check utility: `{cfg.get('check_reranked_utility')}`", '- The reranker never sees a reference report at hidden-test inference; references are only development targets. Its deployed features are pathology agreement with the generator auxiliary image head, text confidence, length, repetition, and beam rank.','']
    lines += ['## Submission notes','', '- Run the supplied Mac build script inside each task source directory.', '- Keep `weights.zip` separate from the thin Docker image and place its files at ZIP root.', '- Run `forithmus generate`, `forithmus test`, and `forithmus validate` before the final submission.', '- Task 3 in this unattended bundle remains the known-working baseline rather than risking an unvalidated text-to-CT dependency chain.','']
    (DEL/'WEEKLONG_RESULTS.md').write_text('\n'.join(lines))


def main():
    if DEL.exists(): shutil.rmtree(DEL)
    DEL.mkdir(parents=True)
    resolve_task1(); resolve_task2(); resolve_task3(); write_results()
    manifest={}
    for p in sorted(DEL.rglob('*')):
        if p.is_file() and p.stat().st_size<10*1024**3:
            manifest[str(p.relative_to(DEL))]={'bytes':p.stat().st_size,'sha256':sha256(p)}
    write_json(manifest,DEL/'SHA256_MANIFEST.json')
    archive=Path('/workspace/vlm3d_WEEKLONG_CHAMPION_DELIVERABLES.tar.gz')
    if archive.exists(): archive.unlink()
    import tarfile
    with tarfile.open(archive,'w:gz') as tf: tf.add(DEL,arcname='vlm3d_weeklong_deliverables')
    print('DELIVERABLES COMPLETE'); print(archive); print('size GB',round(archive.stat().st_size/1024**3,2))

if __name__=='__main__': main()

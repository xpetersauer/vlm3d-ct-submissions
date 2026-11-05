#!/usr/bin/env python3
"""Prepare isolated working copies of recovered trainers. Does not start training."""
import argparse
import ast
import json
from pathlib import Path
import shutil
import subprocess
import sys

HERE=Path(__file__).resolve().parent
REPO=HERE.parent


def prepare(output,data_root,tensor_cache=None):
    output=output.expanduser().resolve()
    data_root=data_root.expanduser().resolve()
    if output.exists():
        raise ValueError('Choose a new output directory; existing experiments are never overwritten: '+str(output))
    for name in ['dataset/multi_abnormality_labels/train_predicted_labels.csv',
                 'dataset/multi_abnormality_labels/valid_predicted_labels.csv',
                 'dataset/radiology_text_reports/train_reports.csv',
                 'dataset/radiology_text_reports/validation_reports.csv']:
        if not (data_root/name).is_file():raise ValueError('Missing CT-RATE table: '+str(data_root/name))
    project=output/'project'
    work=output/'work'
    project.mkdir(parents=True)
    work.mkdir()
    (project/'CT-RATE').symlink_to(data_root,target_is_directory=True)
    shutil.copytree(HERE/'original/weeklong_champion',project/'weeklong_champion')
    shutil.copytree(HERE/'original/sota_deadline',project/'sota_deadline')
    for name in ['model_code_from_original.py','vlm3d_common.py','sota_models.py']:
        shutil.copy2(REPO/'tasks/task2'/name,project/name)
    shutil.copy2(REPO/'tasks/task2/sota_models.py',project/'sota_deadline/sota_models.py')
    shutil.copy2(HERE/'original/baseline/task1_ct_to_abnormality.py',project/'task1_ct_to_abnormality.py')
    for p in (project/'weeklong_champion').rglob('*.py'):
        text=p.read_text().replace('/workspace/vlm3d_original_task_split',str(project)).replace('/workspace/vlm3d_weeklong',str(work))
        p.write_text(text)
    posthoc=project/'weeklong_champion/champion/posthoc_select.py'
    text=posthoc.read_text()
    text=text.replace('from pathlib import Path\n', 'from pathlib import Path\nPROJECT = Path('+repr(str(project))+')\n',1)
    posthoc.write_text(text)
    # Keep original files untouched. Compatibility changes apply only to working copies.
    rb=project/'weeklong_champion/champion/radbert_soft_targets.py'
    text=rb.read_text()
    old='    model.load_state_dict(state, strict=True)'
    replacement="    incompatible = model.load_state_dict(state, strict=False)\n    if incompatible.missing_keys or set(incompatible.unexpected_keys) - {'model.embeddings.position_ids'}:\n        raise ValueError('RadBERT state mismatch: ' + str(incompatible))"
    if text.count(old)!=1:raise ValueError('Unexpected RadBERT source version')
    rb.write_text(text.replace(old,replacement))
    shutil.copytree(REPO/'tasks/task3',project/'task3')
    for name in ['install_task3_training_resume_patch.py','install_task3_tensor_cache_patch.py']:
        subprocess.run([sys.executable,str(HERE/'original/task3_runpod'/name),
                        str(project/'task3/model_code_from_original.py')],check=True)
    # Pin MONAI's image-size argument explicitly in the recovered baseline trainer.
    baseline=project/'task1_ct_to_abnormality.py'
    text=baseline.read_text().replace('        spatial_dims=3,','        spatial_dims=3,\n        img_size=tuple(args.target_shape),',1)
    baseline.write_text(text)
    for path in project.rglob('*.py'):ast.parse(path.read_text(),filename=str(path))
    config={'project':str(project),'work':str(work),'data_root':str(data_root),
            'repository':str(REPO),'tensor_cache':str(tensor_cache.expanduser().resolve() if tensor_cache else output/'tensor_cache'),
            'raw_stage':str(output/'raw_stage')}
    (output/'workspace.json').write_text(json.dumps(config,indent=2)+'\n')
    print('Prepared:',output)
    print('Next: python training/run_stage.py --workspace '+str(output)+' task1-features')
    print('This is an isolated reproduction workspace; submitted checkpoints are unchanged.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--tensor-cache',type=Path)
    a=p.parse_args()
    try:prepare(a.output,a.data_root,a.tensor_cache)
    except (OSError,ValueError,subprocess.CalledProcessError) as e:raise SystemExit(str(e))

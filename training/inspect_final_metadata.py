#!/usr/bin/env python3
"""Read saved training metadata from a trusted released ZIP on CPU; never starts training."""
import argparse
import json
from pathlib import Path
import zipfile


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('weights_zip',type=Path)
    p.add_argument('--task',choices=['task1','task2','task3'],required=True)
    a=p.parse_args()
    try:import torch
    except ImportError:raise SystemExit('PyTorch is required. Run in the reproduction environment; a GPU is not required.')
    names={'task1':['dino_head_0.pt','dino_head_1.pt','dino_head_2.pt'],
           'task2':['task2_best.pt'],
           'task3':['task3_autoencoder.pt','task3_report_to_latent/mapper.pt']}[a.task]
    keys=['variant','seed','epoch','best_modeldev_auroc','feature_dim','n_tokens','n_slices',
          'report_weight','recon_weight','best_val_loss','global_step','extra','text_encoder_name',
          'latent_dim','target_shape','base_channels','freeze_text_encoder','max_text_length']
    result={}
    with zipfile.ZipFile(a.weights_zip.expanduser()) as z:
        for name in names:
            print('Reading on CPU:',name,flush=True)
            with z.open(name) as stream:obj=torch.load(stream,map_location='cpu',weights_only=True)
            result[name]={k:obj[k] for k in keys if k in obj}
            state=obj.get('training_state') or {}
            result[name]['training_state']={k:state[k] for k in ['stage','epoch','epochs_target','dataset_rows','complete','last_loss'] if k in state}
            del obj
    print(json.dumps(result,indent=2,default=str))


if __name__=='__main__':main()

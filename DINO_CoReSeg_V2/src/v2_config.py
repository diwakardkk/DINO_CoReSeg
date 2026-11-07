"""V2 configuration and path isolation; never writes to V1 or dataset roots."""
import json
import os
from pathlib import Path
import re
import hashlib
from functools import lru_cache
import torch
import yaml
from .common import RESEARCH,seed_all,fingerprint


def load(path,data_root=None,v1_root=None,allow_cpu=False):
    c=yaml.safe_load(Path(path).read_text())
    c['research']=str(RESEARCH)
    for key,value in [('root',data_root or os.environ.get('CORESEG_DATA_ROOT') or c['data_root']),
                      ('v1_root',v1_root or os.environ.get('CORESEG_V1_ROOT') or c['v1_root'])]:
        p=Path(value);c[key]=str((RESEARCH/p).resolve() if not p.is_absolute() else p.resolve())
    if not re.fullmatch(r'dino_coreseg_v2_[a-zA-Z0-9_]+',c['experiment']):raise ValueError('A new V2 experiment ID is required')
    if c['image_size']%c['patch_size']:raise ValueError('Input must be divisible by patch size')
    if c['patch_size']!=14 or c['feature_dim']!=384:raise ValueError('Frozen DINOv2-S/14 requires patch 14 and feature dimension 384')
    c['grid_size']=c['image_size']//c['patch_size']
    if c['projection_dim']%8 or c['projection_dim']<8:raise ValueError('Projection channels must be a positive multiple of eight')
    if c['variant'] not in ['resolution','rgb_path','boundary_refinement','multilayer']:raise ValueError('Unknown model version')
    if c['variant']=='multilayer' and c['intermediate_layers']!=sorted(set(c['intermediate_layers'])):
        raise ValueError('Intermediate layer indices must be sorted and unique')
    if c['variant']=='multilayer' and (not c['intermediate_layers'] or c['intermediate_layers'][-1]!=11):
        raise ValueError('Multilayer prototype semantics require the last DINO block (index 11)')
    if any(not 0<=i<12 for i in c['intermediate_layers']):raise ValueError('DINO-S has 12 blocks, indexed 0–11')
    if c['device']!='cuda' and not allow_cpu:raise ValueError('Study execution requires CUDA')
    if c['device']=='cuda' and not torch.cuda.is_available():
        if allow_cpu:c['device']='cpu'
        else:raise RuntimeError('CUDA unavailable; no CPU training fallback')
    if c['device']=='cuda' and not c['gpu']['all_visible_devices'] and torch.cuda.device_count()!=1:
        raise ValueError('Single-GPU V2 requires exactly one CUDA_VISIBLE_DEVICES entry; explicitly enable multi-GPU only in an authorized new configuration')
    if c.get('flip_augmentation'):raise ValueError('Initial V2 does not enable cached-feature augmentation')
    if c['gradient_accumulation']<1:raise ValueError('Invalid accumulation factor')
    if c['lr_schedule'] not in ['constant','cosine']:raise ValueError('Unknown learning-rate policy')
    if not 0<c['threshold']<1 or not 0<=c['normal_fpr_limit']<=1:raise ValueError('Invalid decision protocol')
    torch.set_num_threads(c['threads']);os.environ['LOKY_MAX_CPU_COUNT']=str(c['threads']);seed_all(c['seed'])
    if c['device']=='cuda':
        torch.backends.cuda.matmul.allow_tf32=c['gpu']['tf32']
        torch.backends.cudnn.allow_tf32=c['gpu']['tf32']
        torch.backends.cudnn.benchmark=c['gpu']['cudnn_benchmark']
    torch.use_deterministic_algorithms(c['gpu']['deterministic'],warn_only=True)
    return c


def output(c):
    root=Path(c['research']).resolve();dest=root/'outputs'/c['experiment']
    if not dest.resolve().is_relative_to(root) or root==Path(c['v1_root']).resolve():raise ValueError('V2 must be independent of V1')
    return dest


def scientific(c):
    return {k:v for k,v in c.items() if k not in ['root','data_root','v1_root','research','threads','loader']}


def cache_root(c):
    # Resolution/refinement/boundary runs share only identically prepared data.
    preparation={k:c[k] for k in ['image_size','rgb_size','selection_size','patch_size','feature_dim']}
    preparation['layers']=c['intermediate_layers'] if c['variant']=='multilayer' else [11]
    preparation['normalization']={k:c['backbone'][k] for k in ['mean','std','weight_sha256']}
    preparation['precision_policy']={k:c['gpu'][k] for k in ['amp','amp_dtype','tf32']}
    from .runtime import amp_dtype
    preparation['actual_precision']=str(amp_dtype(c));preparation['torch_version']=str(torch.__version__)
    preparation['source']=feature_source_identity()
    return Path(c['research'])/'cache_v2'/fingerprint(preparation)[:20]


@lru_cache(maxsize=1)
def feature_source_identity():
    source=hashlib.sha256((RESEARCH/'src/v2_features.py').read_bytes()).hexdigest()
    vendor=hashlib.sha256();root=RESEARCH/'vendor/dinov2'
    for path in sorted(root.rglob('*.py')):
        vendor.update(str(path.relative_to(root)).encode());vendor.update(path.read_bytes())
    return {'preprocessing_source_sha256':source,'vendor_python_source_sha256':vendor.hexdigest()}


def source_identity():
    from .common import file_hash
    names=['v2_config.py','v2_model.py','v2_features.py','v2_losses.py','v2_prototypes.py',
           'v2_train.py','v2_metrics.py','v2_evaluate.py','v2_data.py','v2_pipeline.py','runtime.py','common.py']
    return {name:file_hash(RESEARCH/'src'/name) for name in names}


def software_identity():
    from importlib.metadata import version,PackageNotFoundError
    result={}
    for name in ['torch','torchvision','numpy','scipy','scikit-learn','matplotlib','Pillow','PyYAML']:
        try:result[name]=version(name)
        except PackageNotFoundError:result[name]=None
    return result

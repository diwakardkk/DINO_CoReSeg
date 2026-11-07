import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import random
import shutil
import signal
import tempfile
from zipfile import ZipFile
import numpy as np
from PIL import Image
import torch
import yaml

RESEARCH=Path(__file__).resolve().parents[1]


def args(description,extras=None):
    p=argparse.ArgumentParser(description=description)
    p.add_argument('--config',default=str(RESEARCH/'configs/gpu_100.yaml'))
    p.add_argument('--data-root',default=None)
    if extras:extras(p)
    return p.parse_args()


def config(path,data_root=None,allow_cpu=False):
    path=Path(path).resolve();c=yaml.safe_load(path.read_text())
    root=data_root or os.environ.get('CORESEG_DATA_ROOT') or c['data_root']
    c['root']=str((RESEARCH/Path(root)).resolve()) if not Path(root).is_absolute() else str(Path(root).resolve())
    c['research']=str(RESEARCH)
    if c['device']!='cuda' and not allow_cpu:raise ValueError('Server training requires device: cuda')
    if c['device']=='cuda' and not torch.cuda.is_available():
        if allow_cpu:c['device']='cpu'
        else:raise RuntimeError('CUDA is unavailable. Install a CUDA-enabled PyTorch build and verify nvidia-smi. No silent CPU fallback.')
    for k,v in dict(image_size=350,rgb_size=100,feature_dim=384,grid_size=25,projection_dim=64,prototype_count=8).items():
        if c[k]!=v:raise ValueError(f'{k} must equal {v} for this architecture')
    if c['flip_augmentation']:raise ValueError('This first GPU version uses no cached-feature augmentation')
    torch.set_num_threads(c['threads']);os.environ['LOKY_MAX_CPU_COUNT']=str(c['threads'])
    seed_all(c['seed'])
    torch.use_deterministic_algorithms(c['gpu']['deterministic'],warn_only=True)
    if c['device']=='cuda':
        torch.backends.cuda.matmul.allow_tf32=c['gpu']['tf32']
        torch.backends.cudnn.allow_tf32=c['gpu']['tf32']
        torch.backends.cudnn.benchmark=c['gpu']['cudnn_benchmark']
    return c


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def fingerprint(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True).encode()).hexdigest()


def scientific_config(c):
    # Location/IO-worker settings can change when a bundle is moved. Scientific
    # parameters, batch policy and precision policy cannot change on resume.
    return {k:v for k,v in c.items() if k not in ['root','data_root','research','loader','threads']}


def file_hash(path):
    with open(path,'rb') as f:
        h=hashlib.sha256()
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
        return h.hexdigest()


def atomic_json(path,obj,immutable=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)
    if immutable and path.exists():
        if path.read_text()!=text:raise RuntimeError(f'Refusing to change finalized artifact {path}')
        return
    with tempfile.NamedTemporaryFile('w',dir=path.parent,delete=False) as f:
        f.write(text);f.flush();os.fsync(f.fileno());temp=f.name
    os.replace(temp,path)


def atomic_torch(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile('wb',dir=path.parent,delete=False) as f:
        temp=f.name
        try:torch.save(obj,f);f.flush();os.fsync(f.fileno())
        except BaseException:
            Path(temp).unlink(missing_ok=True);raise
    os.replace(temp,path)


def atomic_copy(source,destination):
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    with open(source,'rb') as src,tempfile.NamedTemporaryFile('wb',dir=destination.parent,delete=False) as dst:
        shutil.copyfileobj(src,dst);dst.flush();os.fsync(dst.fileno());temp=dst.name
    os.replace(temp,destination)


def read_json(path):return json.loads(Path(path).read_text())
def output(c):return RESEARCH/'outputs'/c['experiment']
def load_split(name):return read_json(RESEARCH/'splits'/f'{name}.json')['rows']


def csv_write(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with tempfile.NamedTemporaryFile('w',newline='',dir=path.parent,delete=False) as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
        f.flush();os.fsync(f.fileno());temp=f.name
    os.replace(temp,path)


def append_jsonl(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',buffering=1) as f:
        f.write(json.dumps(obj,sort_keys=True,allow_nan=False)+'\n');f.flush()


def load_image(root,path,archive=None):
    if archive:
        with ZipFile(Path(root)/archive) as z:
            with Image.open(io.BytesIO(z.read(path))) as im:im.load();return im.copy()
    with Image.open(Path(root)/path) as im:im.load();return im.copy()


def pair_arrays(c,row,include_mask=True):
    im=load_image(c['root'],row['image'],row.get('archive')).convert('RGB');mask=None
    if include_mask:
        if row['mask']:
            m=load_image(c['root'],row['mask'],row.get('archive'))
            if m.size!=im.size:raise ValueError(f"Image/mask dimensions differ: {row['id']}")
            mask=np.asarray(m.convert('L'))>=128
        else:
            if row['positive']:raise ValueError('Positive sample missing mask')
            mask=np.zeros((im.height,im.width),dtype=bool)
    return im,mask


def pair_by_stem(images,masks):
    def index(paths):
        result={}
        for p in sorted(paths):
            if p.stem in result:raise ValueError(f'Duplicate stem {p}')
            result[p.stem]=p
        return result
    a,b=index(images),index(masks)
    if a.keys()!=b.keys():raise ValueError(f'Missing/orphan mask stems: {a.keys()-b.keys()}, {b.keys()-a.keys()}')
    return [(a[k],b[k]) for k in sorted(a)]


class UnionFind:
    def __init__(self):self.parents={}
    def find(self,a):
        self.parents.setdefault(a,a)
        if self.parents[a]!=a:self.parents[a]=self.find(self.parents[a])
        return self.parents[a]
    def union(self,a,b):
        a,b=self.find(a),self.find(b)
        if a!=b:self.parents[max(a,b)]=min(a,b)


class StopRequested(Exception):pass
class StopFlag:
    def __init__(self):self.requested=False
    def install(self):
        def handler(signum,frame):self.requested=True
        signal.signal(signal.SIGTERM,handler);signal.signal(signal.SIGINT,handler)
    def check(self):
        if self.requested:raise StopRequested('Graceful stop requested; committed progress has been saved.')


STOP=StopFlag()

import json
import os
import time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from .common import (RESEARCH,args,config,atomic_json,fingerprint,load_split,pair_arrays,
                     file_hash,read_json,output,STOP)
from .runtime import device,amp_dtype,make_loader,synchronize,tune_batch
from .backbone import load_backbone

CACHE_VERSION='coreseg-gpu-v1-fp16-storage-350-rgb100'


def cache_path(row):return RESEARCH/'cache'/f"{fingerprint(row['id'])}.npz"


def inspect_cache(path,signature=None):
    with np.load(path,allow_pickle=False) as z:
        metadata=json.loads(str(z['metadata']))
        if signature and metadata['signature']!=signature:raise ValueError(f'Cache identity mismatch {path}')
        for name,shape in [('features',(384,25,25)),('rgb',(3,100,100)),('mask',(1,350,350))]:
            if z[name].shape!=shape or not np.isfinite(z[name]).all():raise ValueError(f'Bad {name} cache {path}')
        if not np.isin(z['mask'],[0,1]).all():raise ValueError('Cache target is not binary')
        return metadata


class Prepared(Dataset):
    def __init__(self,c,rows):self.c=c;self.rows=rows
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index]
        if file_hash(Path(self.c['root'])/row['image'])!=row['image_sha256']:raise ValueError('Source changed since audit')
        image,mask=pair_arrays(self.c,row)
        tensor=np.array(image.resize((350,350),Image.Resampling.BICUBIC),dtype=np.float32).transpose(2,0,1)/255.
        small=np.array(image.resize((100,100),Image.Resampling.BILINEAR),dtype=np.uint8).transpose(2,0,1)
        target=np.array(Image.fromarray(mask.astype(np.uint8)*255).resize((350,350),Image.Resampling.NEAREST))>=128
        return {'index':index,'image':torch.from_numpy(tensor.copy()),'rgb':torch.from_numpy(small.copy()),
                'mask':torch.from_numpy(target[None].astype(np.uint8)),'size':torch.tensor([image.width,image.height])}


def extract(c,rows,allow_test=False):
    if any(r['split']=='test' for r in rows):
        marker=output(c)/'holdout/FROZEN.json'
        if not allow_test or not marker.exists():raise RuntimeError('Test extraction requires the completed, frozen holdout checkpoint')
        if file_hash(output(c)/'holdout/best.pt')!=read_json(marker)['checkpoint_sha256']:raise ValueError('Frozen checkpoint changed')
    model,identity=load_backbone(c)
    identity.update(cache_version=CACHE_VERSION,compute_dtype=str(amp_dtype(c)),device_type=c['device'],
                    torch_version=str(torch.__version__),cuda_build=torch.version.cuda,tf32=c['gpu']['tf32'])
    atomic_json(RESEARCH/'cache/backbone_identity.json',identity,immutable=True)
    pending=[];signatures={}
    for original in rows:
        row=dict(original)
        if row['mask']:
            current=file_hash(Path(c['root'])/row['mask'])
            if row.get('mask_sha256') and current!=row['mask_sha256']:raise ValueError('Source mask changed since audit')
            row['mask_sha256']=current
        signature=fingerprint({'row':row,'identity':identity,'mean':c['backbone']['mean'],'std':c['backbone']['std']})
        signatures[row['id']]=signature
        if cache_path(row).exists():inspect_cache(cache_path(row),signature)
        else:pending.append(row)
    if not pending:return {'complete':True,'created':0,'reused':len(rows)}
    batch_size=tune_batch(c,model,'extraction')
    atomic_json(output(c)/'logs/extraction_hardware.json',{'batch_size':batch_size,'precision':str(amp_dtype(c))})
    loader=make_loader(Prepared(c,pending),c,batch_size=batch_size,seed=c['seed'])
    mean=torch.tensor(c['backbone']['mean'],device=device(c)).view(1,3,1,1)
    std=torch.tensor(c['backbone']['std'],device=device(c)).view(1,3,1,1)
    start=time.perf_counter();created=0
    for batch in loader:
        STOP.check();x=batch['image'].to(device(c),non_blocking=True)
        synchronize(c);tick=time.perf_counter()
        with torch.inference_mode():features=model((x-mean)/std).float().cpu().numpy().astype(np.float16)
        synchronize(c);latency=time.perf_counter()-tick
        for j,index in enumerate(batch['index'].tolist()):
            row=pending[index];path=cache_path(row);path.parent.mkdir(parents=True,exist_ok=True)
            metadata={'version':CACHE_VERSION,'row':row,'model':identity,'signature':signatures[row['id']],
                      'original_size':batch['size'][j].tolist(),'preprocessing':'RGB bicubic350, ImageNet normalization; binary nearest350; RGB bilinear100',
                      'normalization':{'mean':c['backbone']['mean'],'std':c['backbone']['std']}}
            temp=path.with_suffix('.tmp')
            with temp.open('wb') as f:
                # Uncompressed NPZ trades a little disk for low-overhead GPU feeding.
                np.savez(f,features=features[j],rgb=batch['rgb'][j].numpy(),mask=batch['mask'][j].numpy(),metadata=np.array(json.dumps(metadata,sort_keys=True)))
                f.flush();os.fsync(f.fileno())
            inspect_cache(temp,metadata['signature'])
            if path.exists():raise RuntimeError('Concurrent cache writer detected')
            os.replace(temp,path);created+=1
        report={'requested':len(rows),'created':created,'reused':len(rows)-len(pending),
                'complete':created==len(pending),'elapsed_seconds':time.perf_counter()-start,
                'last_batch_backbone_seconds_per_image':latency/len(features),'batch_size':batch_size,
                'cache_bytes':sum(p.stat().st_size for p in (RESEARCH/'cache').glob('*.npz'))}
        atomic_json(output(c)/'logs'/f"extraction_{rows[0]['split']}.json",report)
        print(f"Extraction {created}/{len(pending)} new samples; GPU batch {batch_size}",flush=True)
    del model
    if c['device']=='cuda':torch.cuda.empty_cache()
    return report


class CachedDataset(Dataset):
    def __init__(self,rows,flips=False):
        if flips:raise ValueError('No augmentation in this server protocol')
        self.rows=rows;self.identity=read_json(RESEARCH/'cache/backbone_identity.json')
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index]
        with np.load(cache_path(row),allow_pickle=False) as z:
            meta=json.loads(str(z['metadata']))
            if meta['row']['id']!=row['id'] or meta['row']['split']!=row['split'] or meta['model']!=self.identity:
                raise RuntimeError('Cache provenance mismatch')
            item={k:torch.from_numpy(z[k].astype(np.float32)) for k in ['features','rgb','mask']}
        item['rgb']/=255.;item['positive']=torch.tensor(row['positive']);item['id']=row['id'];item['source']=row['source']
        return item


if __name__=='__main__':
    a=args('Batched CUDA frozen-feature extraction',lambda p:p.add_argument('--split',choices=['train','val','test'],required=True))
    c=config(a.config,a.data_root);extract(c,load_split(a.split),allow_test=a.split=='test')

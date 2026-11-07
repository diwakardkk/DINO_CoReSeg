"""Frozen official DINOv2 features, versioned preprocessing and isolated caches."""
import gc
import json
import os
from pathlib import Path
import tempfile
import time
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset
from .common import RESEARCH,atomic_json,file_hash,fingerprint,pair_arrays,read_json,STOP
from .runtime import device,parallelize,autocast,amp_dtype,make_loader,synchronize,rng_state,restore_rng,gpu_indices
from .v2_config import cache_root,output,feature_source_identity


class SpatialDinoV2(nn.Module):
    def __init__(self,backbone,c):super().__init__();self.backbone=backbone;self.c=c
    def forward(self,x):
        size=self.c['image_size'];grid=self.c['grid_size'];channels=self.c['feature_dim']
        if tuple(x.shape[1:])!=(3,size,size):raise ValueError('Incorrect V2 input dimensions')
        if self.c['variant']=='multilayer':
            # Official normalized patch maps; no CLS/register tokens.
            maps=self.backbone.get_intermediate_layers(x,n=self.c['intermediate_layers'],reshape=True,return_class_token=False,norm=True)
            result=torch.stack(maps,1)
        else:
            tokens=self.backbone.forward_features(x)['x_norm_patchtokens']
            if tuple(tokens.shape[1:])!=(grid*grid,channels):raise ValueError('DINO token shape differs from derived mask grid')
            result=tokens.transpose(1,2).reshape(len(x),channels,grid,grid)[:,None]
        if tuple(result.shape[2:])!=(channels,grid,grid):raise ValueError('Invalid DINO spatial output')
        return result


def load_backbone(c):
    os.environ['XFORMERS_DISABLED']='1'
    weights=RESEARCH/c['backbone']['checkpoint'];repo=RESEARCH/c['backbone']['local_repo']
    if file_hash(weights)!=c['backbone']['weight_sha256']:raise ValueError('DINO weights failed SHA-256 verification')
    raw=torch.hub.load(str(repo),c['backbone']['name'],source='local',pretrained=False,verbose=False)
    raw.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True)
    if raw.embed_dim!=c['feature_dim'] or raw.patch_size!=c['patch_size'] or raw.num_register_tokens:raise ValueError('Wrong backbone')
    raw.eval().requires_grad_(False)
    identity={'cache_version':'coreseg-v2-spatial-v1','input_size':c['image_size'],'rgb_size':c['rgb_size'],
              'selection_size':c['selection_size'],'grid_size':c['grid_size'],
              'layers':c['intermediate_layers'] if c['variant']=='multilayer' else [11],
              'weight_sha256':file_hash(weights),'source_commit':(repo/'SOURCE_COMMIT.txt').read_text().strip(),
              'compute_dtype':str(amp_dtype(c)),'torch_version':str(torch.__version__),'tf32':c['gpu']['tf32'],
              'parameters':sum(p.numel() for p in raw.parameters()),'storage_dtype':'float16',
              'mean':c['backbone']['mean'],'std':c['backbone']['std'],
              'feature_source_identity':feature_source_identity()}
    return parallelize(SpatialDinoV2(raw,c),c).eval(),identity


def cache_path(c,row):return cache_root(c)/f"{fingerprint(row['id'])}.npz"


class Prepared(Dataset):
    def __init__(self,c,rows):self.c=c;self.rows=rows
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index];c=self.c
        if file_hash(Path(c['root'])/row['image'])!=row['image_sha256']:raise ValueError('Source image changed')
        if row['mask'] and row.get('mask_sha256') and file_hash(Path(c['root'])/row['mask'])!=row['mask_sha256']:raise ValueError('Source target changed')
        image,truth=pair_arrays(c,row)
        resize=lambda size:np.asarray(Image.fromarray(truth.astype(np.uint8)*255).resize((size,size),Image.Resampling.NEAREST))>=128
        target=resize(c['image_size']);selection=resize(c['selection_size'])
        if row['positive'] and row['split']!='test' and (not target.any() or not selection.any()):raise ValueError(f"Positive development lesion disappeared in preprocessing: {row['id']}")
        return {'index':index,'image':torch.from_numpy(np.array(image.resize((c['image_size'],)*2,Image.Resampling.BICUBIC),np.float32).transpose(2,0,1).copy()/255.),
                'rgb':torch.from_numpy(np.array(image.resize((c['rgb_size'],)*2,Image.Resampling.BILINEAR),np.uint8).transpose(2,0,1).copy()),
                'mask':torch.from_numpy(target[None].astype(np.uint8)),
                'selection_mask':torch.from_numpy(selection[None].astype(np.uint8))}


def tune(c,model,kind='training'):
    requested=c['extraction_batch_size'] if kind=='extraction' else c['batch_size']
    if requested!='auto':
        requested=int(requested)
        if requested<2 or (kind=='training' and requested%2):raise ValueError('Use an even batch >=2')
        return requested
    if c['device']!='cuda':return 2
    ngpu=len(gpu_indices(c));cap=c['gpu']['extraction_cap_per_gpu'] if kind=='extraction' else c['gpu']['batch_cap_per_gpu']
    batch=max(2,cap*ngpu);state=rng_state();training=model.training;probes=[]
    layers=len(c['intermediate_layers']) if c['variant']=='multilayer' else 1
    try:
        while batch>=2:
            x=rgb=target=result=positive=loss=parts=None
            try:
                gc.collect();torch.cuda.empty_cache();synchronize(c)
                for index in gpu_indices(c):torch.cuda.reset_peak_memory_stats(index)
                free={i:torch.cuda.mem_get_info(i)[0] for i in gpu_indices(c)}
                baseline={i:torch.cuda.memory_allocated(i) for i in gpu_indices(c)}
                if kind=='extraction':
                    with torch.inference_mode(),autocast(c):result=model(torch.zeros(batch,3,c['image_size'],c['image_size'],device=device(c)))
                else:
                    from .v2_losses import objective
                    model.train();model.zero_grad(set_to_none=True)
                    x=torch.randn(batch,layers,c['feature_dim'],c['grid_size'],c['grid_size'],device=device(c))
                    rgb=torch.rand(batch,3,c['rgb_size'],c['rgb_size'],device=device(c));target=torch.zeros(batch,1,c['image_size'],c['image_size'],device=device(c))
                    start=c['image_size']//4;target[:batch//2,:,start:start*2,start:start*2]=1
                    positive=torch.arange(batch,device=device(c))<batch//2
                    with autocast(c):result=model(x,rgb);loss,parts=objective(result,target,positive,c,.1)
                    loss.backward()
                synchronize(c)
                reserve=0 if kind=='extraction' else 3*sum(p.numel()*p.element_size() for p in model.parameters())
                extra={i:torch.cuda.max_memory_allocated(i)-baseline[i]+(reserve if i==0 else 0) for i in gpu_indices(c)}
                safe=all(extra[i]<=free[i]*c['gpu']['memory_fraction_for_probe'] for i in free)
                probes.append({'batch':batch,'safe':safe,'free':free,'extra':extra})
                if safe:
                    atomic_json(output(c)/'reports'/f'memory_probe_{kind}.json',{'batch_size':batch,'probes':probes})
                    return batch
            except torch.cuda.OutOfMemoryError:probes.append({'batch':batch,'safe':False,'error':'out_of_memory'})
            finally:
                x=rgb=target=result=positive=loss=parts=None;model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
            batch=(batch//2)//2*2
        raise RuntimeError('Insufficient GPU memory for V2 batch two')
    finally:model.train(training);restore_rng(state)


def extract(c,rows,allow_test=False):
    if any(row['split']=='test' for row in rows):
        marker=output(c)/'checkpoints/FROZEN.json'
        if not allow_test or not marker.exists():raise ValueError('V2 final-test features require a frozen selected model')
        if file_hash(output(c)/'checkpoints/best.pt')!=read_json(marker)['checkpoint_sha256']:raise ValueError('Frozen V2 model changed')
    model,identity=load_backbone(c);folder=cache_root(c);folder.mkdir(parents=True,exist_ok=True)
    atomic_json(folder/'identity.json',identity,immutable=True)
    pending=[]
    for row in rows:
        STOP.check();path=cache_path(c,row)
        if path.exists():
            with np.load(path,allow_pickle=False) as z:
                metadata=json.loads(str(z['metadata']))
                if metadata['identity']!=identity or metadata['row']!=row:raise ValueError('V2 cache identity mismatch')
        else:pending.append(row)
    if not pending:del model;return {'created':0,'reused':len(rows)}
    batch_size=tune(c,model,'extraction');loader=make_loader(Prepared(c,pending),c,batch_size=batch_size)
    mean=torch.tensor(identity['mean'],device=device(c))[None,:,None,None];std=torch.tensor(identity['std'],device=device(c))[None,:,None,None]
    elapsed=0.;created=0
    with torch.inference_mode():
        for batch in loader:
            STOP.check();x=batch['image'].to(device(c),non_blocking=True);synchronize(c);tick=time.perf_counter()
            features=model((x-mean)/std).float().cpu().numpy().astype(np.float16);synchronize(c);elapsed+=time.perf_counter()-tick
            for j,index in enumerate(batch['index'].tolist()):
                row=pending[index];path=cache_path(c,row)
                with tempfile.NamedTemporaryFile(dir=folder,suffix='.tmp',delete=False) as stream:
                    temp=Path(stream.name)
                    np.savez(stream,features=features[j],rgb=batch['rgb'][j].numpy(),mask=batch['mask'][j].numpy(),selection_mask=batch['selection_mask'][j].numpy(),
                             metadata=np.array(json.dumps({'identity':identity,'row':row,
                                'input_target_empty':not bool(batch['mask'][j].any()),
                                'selection_target_empty':not bool(batch['selection_mask'][j].any())},sort_keys=True)))
                    stream.flush();os.fsync(stream.fileno())
                os.replace(temp,path);created+=1
            atomic_json(output(c)/'logs'/f"extraction_{rows[0]['split']}.json",{'requested':len(rows),'created':created,'reused':len(rows)-len(pending),
                        'backbone_seconds':elapsed,'backbone_seconds_per_new_image':elapsed/created,'batch_size':batch_size,'complete':created==len(pending)})
            print(f'V2 features {created}/{len(pending)} new images; batch {batch_size}',flush=True)
    del model
    if c['device']=='cuda':torch.cuda.empty_cache()
    return {'created':created,'reused':len(rows)-len(pending)}


class CachedDataset(Dataset):
    def __init__(self,c,rows):self.c=c;self.rows=rows;self.identity=read_json(cache_root(c)/'identity.json')
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index]
        with np.load(cache_path(self.c,row),allow_pickle=False) as z:
            metadata=json.loads(str(z['metadata']))
            if metadata['identity']!=self.identity or metadata['row']!=row:raise ValueError('Cache membership/identity changed')
            item={key:torch.from_numpy(z[key].astype(np.float32)) for key in ['features','rgb','mask','selection_mask']}
        expected=(self.c['feature_dim'],self.c['grid_size'],self.c['grid_size'])
        if tuple(item['features'].shape[-3:])!=expected:raise ValueError('Incorrect V2 feature shape')
        layers=len(self.c['intermediate_layers']) if self.c['variant']=='multilayer' else 1
        shapes={'features':(layers,*expected),'rgb':(3,self.c['rgb_size'],self.c['rgb_size']),
                'mask':(1,self.c['image_size'],self.c['image_size']),
                'selection_mask':(1,self.c['selection_size'],self.c['selection_size'])}
        for key,shape in shapes.items():
            if tuple(item[key].shape)!=shape or not torch.isfinite(item[key]).all():raise ValueError(f'Invalid V2 {key} cache')
        if not torch.isin(item['mask'],torch.tensor([0.,1.])).all():raise ValueError('Non-binary V2 target')
        item['rgb']/=255.;item.update(id=row['id'],positive=torch.tensor(row['positive']),source=row['source'],
                                    area=torch.tensor(row.get('foreground_fraction',0.),dtype=torch.float32))
        return item

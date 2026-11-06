"""CUDA training with atomic step/epoch checkpoints and exact epoch cursors."""
from collections import defaultdict
import math
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import Sampler
from .common import (RESEARCH,atomic_json,atomic_torch,atomic_copy,append_jsonl,csv_write,file_hash,
                     fingerprint,scientific_config,read_json,seed_all,STOP,StopRequested)
from .model_coreseg import DINOCoReSeg
from .losses import objective
from .extract_features import CachedDataset
from .runtime import (device,parallelize,unwrap,make_loader,move_batch,autocast,amp_dtype,
                      rng_state,restore_rng,optimizer_to,tune_batch,synchronize,hardware,gpu_identity)
from .cv_metrics import monitor_partition


def selection_key(dice,fpr,limit):return (1,dice,-fpr) if fpr<=limit else (0,-fpr,dice)


class BalancedBatches(Sampler):
    def __init__(self,rows,batch_size,seed,epoch=0,first_step=0):
        if batch_size<2 or batch_size%2:raise ValueError('Batch must be an even number >=2')
        self.rows=rows;self.batch_size=batch_size;self.seed=seed
        self.positive=defaultdict(list);self.negative=[]
        for i,row in enumerate(rows):
            if row['positive']:self.positive[row['source']].append(i)
            else:self.negative.append(i)
        if not self.positive or not self.negative:raise ValueError('Both image categories are required')
        self.total_steps=math.ceil(sum(map(len,self.positive.values()))/(batch_size//2));self.set_position(epoch,first_step)
    def set_position(self,epoch,first_step):
        self.epoch=epoch;self.first_step=first_step
    def __len__(self):return max(0,self.total_steps-self.first_step)
    def __iter__(self):
        rng=np.random.default_rng(self.seed+self.epoch);sources=sorted(self.positive)
        normals=rng.permutation(self.negative);cursor=0
        for step in range(self.total_steps):
            batch=[]
            for _ in range(self.batch_size//2):
                source=sources[int(rng.integers(len(sources)))];batch.append(int(rng.choice(self.positive[source])))
                if cursor==len(normals):normals=rng.permutation(self.negative);cursor=0
                batch.append(int(normals[cursor]));cursor+=1
            if step>=self.first_step:yield batch


def implementation_identity():
    return {name:file_hash(RESEARCH/'src'/name) for name in
            ['train.py','runtime.py','common.py','model_coreseg.py','losses.py','prototypes.py','cv_metrics.py','extract_features.py']}


def train_model(c,train,val,prototypes,directory,protocol,seed,on_epoch=None):
    STOP.check()
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    if {r['id'] for r in train}&{r['id'] for r in val}:raise ValueError('Train/validation sample overlap')
    seed_all(seed);raw=DINOCoReSeg(prototypes);model=parallelize(raw,c)
    last=directory/'last.pt';state=torch.load(last,map_location='cpu',weights_only=False) if last.exists() else None
    batch_size=state['batch_size'] if state else tune_batch(c,model,'training')
    evaluation_batch=int(c['evaluation_batch_size']) if c['evaluation_batch_size']!='auto' else batch_size
    identity={'config':fingerprint(scientific_config(c)),'train':fingerprint(train),'validation':fingerprint(val),
              'seed':seed,'software':{'torch':str(torch.__version__),'numpy':np.__version__},
              'prototype_vectors':fingerprint(prototypes.tolist()),'implementation':implementation_identity(),
              'backbone':read_json(RESEARCH/'cache/backbone_identity.json'),
              'batch_size':batch_size,'amp_dtype':str(amp_dtype(c)),
              'visible_gpus':torch.cuda.device_count() if c['device']=='cuda' else 0}
    identity['gpu_devices']=gpu_identity(c)
    optimizer=torch.optim.AdamW(raw.parameters(),lr=c['learning_rate'],weight_decay=c['weight_decay'],fused=c['device']=='cuda')
    scaler=torch.amp.GradScaler('cuda',enabled=c['device']=='cuda' and amp_dtype(c)==torch.float16)
    history=[];epoch=0;next_step=0;global_step=0;bad=0;best_key=(-1,-float('inf'),-float('inf'));running={};running_count=0
    if state:
        if state['identity']!=identity:raise RuntimeError('Resume identity mismatch: data/configuration/source/precision/GPU count changed')
        raw.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);optimizer_to(optimizer,c)
        scaler.load_state_dict(state['scaler']);restore_rng(state['rng'])
        history=state['history'];global_step=state['global_step'];bad=state['bad_epochs'];best_key=tuple(state['best_key'])
        epoch=state['epoch']+1 if state['epoch_complete'] else state['epoch']
        next_step=0 if state['epoch_complete'] else state['next_step']
        running={} if state['epoch_complete'] else state['running'];running_count=0 if state['epoch_complete'] else state['running_count']
        # last.pt is the commit record. Repair derived history/best artifacts if
        # a previous process died between separate atomic file replacements.
        if history:
            winner=max(history,key=lambda r:selection_key(r['val_positive_dice'],r['val_normal_fpr'],c['normal_fpr_limit']))
            selected=directory/f"checkpoints/epoch_{winner['epoch']:03d}.pt"
            if not selected.exists():raise RuntimeError('Committed best epoch checkpoint is missing')
            atomic_copy(selected,directory/'best.pt')
            csv_write(directory/'training_history.csv',history)
            for record in history:atomic_json(directory/f"epochs/epoch_{record['epoch']:03d}.json",record,immutable=True)
    atomic_json(directory/'runtime.json',{'hardware':hardware(c),'batch_size':batch_size,'evaluation_batch_size':evaluation_batch,
               'seed':seed,'epochs':c['max_epochs'],'early_stopping':c['early_stopping'],'optimizer':'AdamW','fused':c['device']=='cuda'})
    if (directory/'FROZEN.json').exists():
        if file_hash(directory/'best.pt')!=read_json(directory/'FROZEN.json')['checkpoint_sha256']:raise RuntimeError('Frozen model changed')
        return read_json(directory/'FROZEN.json')
    sampler=BalancedBatches(train,batch_size,seed,epoch,next_step)
    loader=make_loader(CachedDataset(train),c,batch_sampler=sampler,seed=seed)
    train_monitor_loader=make_loader(CachedDataset(train),c,batch_size=evaluation_batch,seed=seed)
    val_monitor_loader=make_loader(CachedDataset(val),c,batch_size=evaluation_batch,seed=seed)
    last_save=time.monotonic()
    def payload(complete):
        return {'model':raw.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),'rng':rng_state(),
                'config':scientific_config(c),'seed':seed,'model_version':'DINO-CoReSeg-GPU-v1',
                'epoch':epoch,'epoch_complete':complete,'next_step':next_step,'global_step':global_step,
                'running':running,'running_count':running_count,'history':history,'bad_epochs':bad,
                'best_key':list(best_key),'identity':identity,'batch_size':batch_size,'protocol':protocol}
    def save_progress(complete=False):atomic_torch(last,payload(complete))
    try:
        while epoch<c['max_epochs'] and not (c['early_stopping'] and bad>=c['patience']):
            tick=time.perf_counter();sampler.set_position(epoch,next_step);model.train()
            for step,batch in enumerate(loader,start=next_step):
                if STOP.requested:save_progress();STOP.check()
                batch=move_batch(batch,c);optimizer.zero_grad(set_to_none=True)
                with autocast(c):
                    outputs=model(batch['features'],batch['rgb'])
                    loss,parts=objective(outputs,batch['mask'],batch['positive'],c,protocol['positive_area_median'])
                scaler.scale(loss).backward();scaler.unscale_(optimizer)
                for name,p in raw.named_parameters():
                    if p.grad is None:raise FloatingPointError(f'Missing decoder gradient: {name}')
                # A finite total gradient norm detects NaN/Inf while avoiding
                # one host-device synchronization per parameter tensor.
                torch.nn.utils.clip_grad_norm_(raw.parameters(),5.0,error_if_nonfinite=True)
                scaler.step(optimizer);scaler.update()
                values={k:float(v.detach()) for k,v in parts.items()}
                for k,v in values.items():running[k]=running.get(k,0.)+v
                running_count+=1;global_step+=1;next_step=step+1
                append_jsonl(directory/'logs/steps.jsonl',{'epoch':epoch+1,'step':next_step,'global_step':global_step,**values})
                if (global_step%c['checkpoint']['every_steps']==0 or time.monotonic()-last_save>=c['checkpoint']['every_seconds'] or
                    next_step==sampler.total_steps or STOP.requested):
                    save_progress();last_save=time.monotonic()
                atomic_json(directory/'progress.json',{'stage':'training','epoch':epoch+1,'completed_epochs':len(history),
                            'step':next_step,'steps_this_epoch':sampler.total_steps,'global_step':global_step})
                if STOP.requested:STOP.check()
            # Training-batch state is durable before validation starts. An
            # interruption during monitoring repeats monitoring, not SGD updates.
            save_progress()
            train_stats,_=monitor_partition(model,train,c,protocol['positive_area_median'],evaluation_batch,loader=train_monitor_loader)
            val_stats,_=monitor_partition(model,val,c,protocol['positive_area_median'],evaluation_batch,loader=val_monitor_loader)
            record={'epoch':epoch+1,'epoch_seconds':time.perf_counter()-tick,
                    **{'optimization_'+k:v/running_count for k,v in running.items()},
                    **{'train_'+k:v for k,v in train_stats.items()},**{'val_'+k:v for k,v in val_stats.items()}}
            history.append(record);key=selection_key(val_stats['positive_dice'],val_stats['normal_fpr'],c['normal_fpr_limit'])
            improved=key>best_key
            if improved:best_key=key;bad=0
            else:bad+=1
            full=payload(True)
            epoch_checkpoint=directory/f'checkpoints/epoch_{epoch+1:03d}.pt'
            atomic_torch(epoch_checkpoint,full)
            atomic_torch(last,full)
            if improved:atomic_copy(epoch_checkpoint,directory/'best.pt')
            csv_write(directory/'training_history.csv',history)
            atomic_json(directory/f'epochs/epoch_{epoch+1:03d}.json',record,immutable=True)
            append_jsonl(directory/'logs/epochs.jsonl',record)
            print({'run':str(directory.name),**record},flush=True)
            if on_epoch:on_epoch(directory)
            epoch+=1;next_step=0;running={};running_count=0
            if STOP.requested:STOP.check()
        best=torch.load(directory/'best.pt',map_location='cpu',weights_only=False)
        marker={'checkpoint_sha256':file_hash(directory/'best.pt'),'selected_epoch':best['epoch']+1,
                'completed_epochs':len(history),'threshold':c['threshold'],'best_key':list(best_key),
                'FPR_constraint_satisfied':best_key[0]==1,'stop_reason':'early_stopping' if c['early_stopping'] and bad>=c['patience'] else 'max_epochs',
                'protocol':protocol,'identity':identity}
        atomic_json(directory/'FROZEN.json',marker,immutable=True)
        atomic_json(directory/'status.json',{'completed':True,'epochs':len(history),'selected_epoch':marker['selected_epoch']})
        return marker
    except StopRequested:
        atomic_json(directory/'status.json',{'completed':False,'interrupted':True,'completed_epochs':len(history),
                    'epoch':epoch+1,'next_step':next_step,'message':'Resume restarts at last committed batch, or repeats interrupted monitoring.'})
        raise
    except BaseException as exc:
        atomic_json(directory/'status.json',{'completed':False,'error':repr(exc),'completed_epochs':len(history),
                    'latest_checkpoint':str(last),'message':'Periodic committed checkpoints remain available; failed optimizer operations are not declared committed.'})
        raise

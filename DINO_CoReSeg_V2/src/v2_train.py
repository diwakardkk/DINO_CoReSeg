"""Independent V2 engine with durable optimizer boundaries and slim retention."""
from collections import defaultdict
from itertools import islice
import json
import math
from pathlib import Path
import platform
import time
import numpy as np
import torch
from torch.utils.data import Sampler
from .common import atomic_json,atomic_torch,atomic_copy,csv_write,read_json,file_hash,fingerprint,seed_all,STOP,StopRequested
from .v2_config import scientific,source_identity,cache_root,output,software_identity
from .v2_model import DINOCoReSegV2
from .v2_features import CachedDataset,tune
from .v2_losses import objective
from .v2_metrics import monitor
from .runtime import parallelize,make_loader,move_batch,autocast,amp_dtype,rng_state,restore_rng,optimizer_to,hardware,gpu_identity,gpu_indices


class BalancedBatches(Sampler):
    """Copied V1 source-balanced, 50/50 category sampler; cursor is microbatches."""
    def __init__(self,rows,batch_size,seed,epoch=0,first_step=0):
        if batch_size<2 or batch_size%2:raise ValueError('Even batch size >=2 required')
        self.rows=rows;self.batch_size=batch_size;self.seed=seed;self.epoch=epoch;self.first_step=first_step
        self.positive=defaultdict(list);self.negative=[]
        for i,row in enumerate(rows):
            if row['positive']:self.positive[row['source']].append(i)
            else:self.negative.append(i)
        if not self.positive or not self.negative:raise ValueError('Both image categories required')
        self.total_steps=math.ceil(sum(map(len,self.positive.values()))/(batch_size//2))
    def __len__(self):return max(0,self.total_steps-self.first_step)
    def __iter__(self):
        rng=np.random.default_rng(self.seed+self.epoch);sources=sorted(self.positive);normals=rng.permutation(self.negative);cursor=0
        for step in range(self.total_steps):
            batch=[]
            for _ in range(self.batch_size//2):
                source=sources[int(rng.integers(len(sources)))];batch.append(int(rng.choice(self.positive[source])))
                if cursor==len(normals):normals=rng.permutation(self.negative);cursor=0
                batch.append(int(normals[cursor]));cursor+=1
            if step>=self.first_step:yield batch


def selection_key(dice,fpr,limit):return (1,dice,-fpr) if fpr<=limit else (0,-fpr,dice)


def history_files(directory,history):
    csv_write(directory/'logs/training_history.csv',history)
    path=directory/'logs/training_history.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
    import os,tempfile
    with tempfile.NamedTemporaryFile('w',dir=path.parent,delete=False) as stream:
        temporary=stream.name
        for record in history:stream.write(json.dumps(record,sort_keys=True,allow_nan=False)+'\n')
        stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def train(c,train_rows,val_rows,prototypes,protocol,directory=None):
    directory=Path(directory) if directory else output(c);cp=directory/'checkpoints';cp.mkdir(parents=True,exist_ok=True)
    if {r['id'] for r in train_rows}&{r['id'] for r in val_rows}:raise ValueError('Training/validation overlap')
    seed_all(c['seed']);raw=DINOCoReSegV2(c,prototypes);model=parallelize(raw,c)
    last=cp/'last.pt';state=torch.load(last,map_location='cpu',weights_only=False) if last.exists() else None
    batch_size=state['batch_size'] if state else tune(c,model)
    accum=c['gradient_accumulation'];evaluation_batch=batch_size if c['evaluation_batch_size']=='auto' else int(c['evaluation_batch_size'])
    identity={'config':fingerprint(scientific(c)),'train':fingerprint(train_rows),'val':fingerprint(val_rows),
              'source':source_identity(),'cache':read_json(cache_root(c)/'identity.json'),
              'prototypes':fingerprint(prototypes.tolist()),'batch_size':batch_size,'gradient_accumulation':accum,
              'torch':str(torch.__version__),'numpy':np.__version__,'gpu':gpu_identity(c),'amp_dtype':str(amp_dtype(c))}
    identity['scientific_software']={key:value for key,value in software_identity().items() if key in ['torch','numpy','scipy','scikit-learn']}
    optimizer=torch.optim.AdamW(raw.parameters(),lr=c['learning_rate'],weight_decay=c['weight_decay'],fused=c['device']=='cuda')
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=c['max_epochs'],eta_min=c['minimum_learning_rate']) if c['lr_schedule']=='cosine' else None
    scaler=torch.amp.GradScaler('cuda',enabled=c['device']=='cuda' and amp_dtype(c)==torch.float16)
    history=[];epoch=0;cursor=0;global_step=0;running={};running_count=0;best=None;best_key=(-1,-float('inf'),-float('inf'));bad=0
    if state:
        if state['identity']!=identity:raise ValueError('V2 resume refused: exact scientific/source/device/batch identity changed')
        raw.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);optimizer_to(optimizer,c)
        if scheduler:scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler']);restore_rng(state['rng'])
        history=state['history'];epoch=state['epoch']+int(state['epoch_complete']);cursor=0 if state['epoch_complete'] else state['next_microbatch']
        running={} if state['epoch_complete'] else state['running'];running_count=0 if state['epoch_complete'] else state['running_count']
        best=state['best_checkpoint'];best_key=tuple(state['best_key']);bad=state['bad_epochs'];global_step=state['global_step']
        if best:atomic_torch(cp/'best.pt',best)
        history_files(directory,history)
    if (cp/'FROZEN.json').exists():
        marker=read_json(cp/'FROZEN.json')
        if marker['identity']!=identity or file_hash(cp/'best.pt')!=marker['checkpoint_sha256']:raise ValueError('Frozen V2 checkpoint changed')
        return marker
    atomic_json(directory/'config/experiment_config.json',scientific(c),immutable=True)
    atomic_json(directory/'config/hardware.json',{'hardware':hardware(c),'python':platform.python_version(),
                'effective_batch_size':batch_size*accum,'microbatch_size':batch_size,'accumulation':accum,
                'accumulation_semantics':'Mean of microbatch objectives; contrast/positive-weight normalization are local to each microbatch.',
                'lr_schedule':c['lr_schedule'],'trainable_parameters':sum(p.numel() for p in raw.parameters()),
                'frozen_parameters':read_json(cache_root(c)/'identity.json').get('parameters'),
                'evaluation_batch_size':evaluation_batch})
    print({'microbatch_size':batch_size,'gradient_accumulation':accum,'effective_batch_size':batch_size*accum,
           'learning_rate':c['learning_rate'],'lr_schedule':c['lr_schedule'],'automatic_lr_scaling':False},flush=True)
    last_save=time.monotonic()
    sampler=BalancedBatches(train_rows,batch_size,c['seed'],epoch,cursor)
    train_loader=make_loader(CachedDataset(c,train_rows),c,batch_sampler=sampler,seed=c['seed'])
    training_monitor_loader=make_loader(CachedDataset(c,train_rows),c,batch_size=evaluation_batch,seed=c['seed'])
    validation_monitor_loader=make_loader(CachedDataset(c,val_rows),c,batch_size=evaluation_batch,seed=c['seed'])
    def payload(complete):
        return {'model':raw.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict() if scheduler else None,
                'scaler':scaler.state_dict(),'rng':rng_state(),'epoch':epoch,'epoch_complete':complete,'next_microbatch':cursor,
                'global_step':global_step,'running':running,'running_count':running_count,'history':history,
                'best_checkpoint':best,'best_key':list(best_key),'bad_epochs':bad,'identity':identity,'batch_size':batch_size,
                'config':scientific(c),'protocol':protocol,'model_version':'DINO-CoReSeg-V2'}
    try:
        while epoch<c['max_epochs'] and not (c['early_stopping'] and bad>=c['patience']):
            if STOP.requested:atomic_torch(last,payload(False));STOP.check()
            tick=time.perf_counter();lr=optimizer.param_groups[0]['lr'];model.train()
            for index in gpu_indices(c):torch.cuda.reset_peak_memory_stats(index)
            sampler.epoch=epoch;sampler.first_step=cursor
            loader=iter(train_loader)
            while True:
                group=list(islice(loader,accum))
                if not group:break
                optimizer.zero_grad(set_to_none=True)
                for batch in group:
                    batch=move_batch(batch,c)
                    with autocast(c):outs=model(batch['features'],batch['rgb']);loss,parts=objective(outs,batch['mask'],batch['positive'],c,protocol['positive_area_median'])
                    scaler.scale(loss/len(group)).backward()
                    for key,value in parts.items():running[key]=running.get(key,0.)+float(value.detach())
                    running_count+=1
                scaler.unscale_(optimizer)
                if any(p.grad is None for p in raw.parameters()):raise FloatingPointError('A decoder parameter has no gradient')
                torch.nn.utils.clip_grad_norm_(raw.parameters(),c['gradient_clip'],error_if_nonfinite=True)
                scaler.step(optimizer);scaler.update();cursor+=len(group);global_step+=1
                if global_step%c['checkpoint']['every_steps']==0 or time.monotonic()-last_save>=c['checkpoint']['every_seconds'] or cursor==sampler.total_steps or STOP.requested:
                    atomic_torch(last,payload(False));last_save=time.monotonic()
                atomic_json(directory/'logs/progress.json',{'stage':'training','epoch':epoch+1,'completed_epochs':len(history),
                    'microbatch':cursor,'microbatches_this_epoch':sampler.total_steps,'global_step':global_step})
                if STOP.requested:atomic_torch(last,payload(False));STOP.check()
            atomic_torch(last,payload(False))
            training=monitor(c,model,train_rows,protocol,evaluation_batch,training_monitor_loader)
            validation=monitor(c,model,val_rows,protocol,evaluation_batch,validation_monitor_loader)
            key=selection_key(validation['positive_dice'],validation['normal_fpr'],c['normal_fpr_limit']);improved=key>best_key
            record={'epoch':epoch+1,'learning_rate':lr,'epoch_seconds':time.perf_counter()-tick,
                    'microbatch_size':batch_size,'gradient_accumulation':accum,'effective_batch_size':batch_size*accum,
                    'GPU_peak_memory':max((torch.cuda.max_memory_allocated(i) for i in gpu_indices(c)),default=None),
                    'checkpoint_selection_score':json.dumps(list(key)),
                    'checkpoint_constraint_satisfied':key[0]==1,
                    **{'training_'+k:v for k,v in training.items()},**{'validation_'+k:v for k,v in validation.items()},
                    **{'optimization_'+k:v/running_count for k,v in running.items()}}
            history.append(record)
            if improved:
                best_key=key;bad=0
                best={'model':{k:v.detach().cpu().clone() for k,v in raw.state_dict().items()},'epoch':epoch,
                      'history':list(history),'identity':identity,'protocol':protocol,'config':scientific(c),'model_version':'DINO-CoReSeg-V2'}
            else:bad+=1
            if scheduler:scheduler.step()
            atomic_torch(last,payload(True))
            if improved:atomic_torch(cp/'best.pt',best)
            keep=c['checkpoint']['keep_epoch_every']
            if keep and (epoch+1)%keep==0:atomic_copy(last,cp/f'epoch_{epoch+1:03d}.pt')
            history_files(directory,history)
            atomic_json(directory/'config/checkpoint_selection.json',{'selected_epoch':best['epoch']+1,'score':list(best_key),
                        'constraint_satisfied':best_key[0]==1,'policy':'V1 feasible-first Dice, otherwise minimum normal FPR then Dice',
                        'selection_grid':c['selection_size'],'threshold':c['threshold'],'normal_fpr_limit':c['normal_fpr_limit']})
            print({'experiment':c['experiment'],**record},flush=True)
            epoch+=1;cursor=0;running={};running_count=0
        marker={'identity':identity,'checkpoint_sha256':file_hash(cp/'best.pt'),'selected_epoch':best['epoch']+1,
                'completed_epochs':len(history),'FPR_constraint_satisfied':best_key[0]==1,'threshold':c['threshold'],
                'stop_reason':'early_stopping' if c['early_stopping'] and bad>=c['patience'] else 'max_epochs','protocol':protocol}
        atomic_json(cp/'FROZEN.json',marker,immutable=True)
        atomic_json(directory/'logs/progress.json',{'stage':'complete','completed_epochs':len(history),'selected_epoch':marker['selected_epoch']})
        return marker
    except StopRequested:
        atomic_json(directory/'logs/training_status.json',{'completed':False,'interrupted':True,'completed_epochs':len(history)})
        raise

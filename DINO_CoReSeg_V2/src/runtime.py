from contextlib import nullcontext
import gc
import os
import platform
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def device(c):return torch.device(c['device'])
def gpu_indices(c):
    return list(range(torch.cuda.device_count() if c['gpu']['all_visible_devices'] else 1)) if c['device']=='cuda' else []


def synchronize(c):
    for index in gpu_indices(c):torch.cuda.synchronize(index)


def amp_dtype(c):
    if c['device']!='cuda' or not c['gpu']['amp']:return None
    name=c['gpu']['amp_dtype']
    if name=='auto':
        count=torch.cuda.device_count() if c['gpu']['all_visible_devices'] else 1
        native_bf16=all(torch.cuda.get_device_capability(i)[0]>=8 for i in range(count))
        return torch.bfloat16 if native_bf16 and torch.cuda.is_bf16_supported() else torch.float16
    if name=='bfloat16' and any(torch.cuda.get_device_capability(i)[0]<8 for i in gpu_indices(c)):
        raise RuntimeError('BF16 requires Ampere or newer on every selected GPU; use amp_dtype: auto for V100.')
    return {'float16':torch.float16,'bfloat16':torch.bfloat16}[name]


def autocast(c):
    dtype=amp_dtype(c)
    return torch.autocast('cuda',dtype=dtype) if dtype is not None else nullcontext()


class AmpModule(nn.Module):
    # DataParallel uses threads; apply autocast inside each replica explicitly.
    def __init__(self,model,c):super().__init__();self.model=model;self.c=c
    def forward(self,*args,**kwargs):
        with autocast(self.c):return self.model(*args,**kwargs)


def parallelize(model,c):
    wrapped=AmpModule(model.to(device(c)),c)
    if c['device']=='cuda' and c['gpu']['all_visible_devices'] and torch.cuda.device_count()>1:
        wrapped=nn.DataParallel(wrapped,device_ids=list(range(torch.cuda.device_count())))
    return wrapped


def unwrap(model):
    if isinstance(model,nn.DataParallel):model=model.module
    return model.model if isinstance(model,AmpModule) else model


def loader_workers(c):
    value=c['loader']['workers']
    if value!='auto':return int(value)
    cores=len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else (os.cpu_count() or 1)
    # Training plus two persistent monitor loaders can coexist. Leave room for
    # PyTorch/metrics threads and respect scheduler CPU affinity on Linux.
    return min(8,max(0,(cores-int(c['threads']))//3))


def make_loader(dataset,c,batch_size=None,batch_sampler=None,seed=42):
    workers=loader_workers(c);kwargs=dict(num_workers=workers,
        pin_memory=c['device']=='cuda' and c['loader']['pin_memory'],
        generator=torch.Generator().manual_seed(seed))
    if workers:kwargs.update(persistent_workers=True,prefetch_factor=c['loader']['prefetch_factor'],multiprocessing_context='spawn')
    if batch_sampler is not None:kwargs['batch_sampler']=batch_sampler
    else:kwargs.update(batch_size=batch_size,shuffle=False)
    return DataLoader(dataset,**kwargs)


def move_batch(batch,c):
    return {k:(v.to(device(c),non_blocking=True) if torch.is_tensor(v) else v) for k,v in batch.items()}


def rng_state():
    return {'torch':torch.get_rng_state(),'numpy':np.random.get_state(),'python':random.getstate(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    torch.set_rng_state(state['torch'].cpu());np.random.set_state(state['numpy']);random.setstate(state['python'])
    if state['cuda']:
        if len(state['cuda'])!=torch.cuda.device_count():raise RuntimeError('Visible GPU count changed on resume; preserve CUDA_VISIBLE_DEVICES')
        torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda']])


def optimizer_to(optimizer,c):
    def move(value):
        if torch.is_tensor(value):return value.to(device(c))
        if isinstance(value,dict):return {k:move(v) for k,v in value.items()}
        if isinstance(value,list):return [move(v) for v in value]
        return value
    for key in optimizer.state:optimizer.state[key]=move(optimizer.state[key])


def hardware(c):
    return {'python_platform':platform.platform(),'torch':str(torch.__version__),
            'cuda_build':torch.version.cuda,'device':c['device'],'amp_dtype':str(amp_dtype(c)),
            'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
            'compiled_cuda_architectures':torch.cuda.get_arch_list() if c['device']=='cuda' else [],
            'loader_workers_per_loader':loader_workers(c),
            'gpus':[{'index':i,'name':torch.cuda.get_device_name(i),
                     'compute_capability':list(torch.cuda.get_device_capability(i)),
                     'free_memory':torch.cuda.mem_get_info(i)[0],
                     'total_memory':torch.cuda.get_device_properties(i).total_memory}
                    for i in gpu_indices(c)]}


def gpu_identity(c):
    # Exclude transient free memory, but detect a different device set/order.
    return [{'name':torch.cuda.get_device_name(i),'capability':list(torch.cuda.get_device_capability(i)),
             'uuid':str(getattr(torch.cuda.get_device_properties(i),'uuid','unknown'))}
            for i in gpu_indices(c)]


def validate_cuda(c):
    if c['device']!='cuda':return False
    # is_available() alone does not prove that this wheel supports a V100.
    for index in gpu_indices(c):
        try:
            with torch.cuda.device(index):
                x=torch.ones(16,16,device=f'cuda:{index}')
                with autocast(c):value=(x@x).sum()
                if not torch.isfinite(value):raise RuntimeError('Non-finite CUDA matrix test')
                torch.cuda.synchronize(index)
                del x,value
        except RuntimeError as exc:
            raise RuntimeError(f'CUDA kernel check failed on GPU {index} ({torch.cuda.get_device_name(index)}). '
                               'For V2 verify your explicit CUDA_VISIBLE_DEVICES allocation. For V100 use a Volta-compatible wheel '
                               '(PyTorch 2.11 CUDA 12.6, not its CUDA 12.8 build).') from exc
    return True



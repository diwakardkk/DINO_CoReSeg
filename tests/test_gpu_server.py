"""Mocked server allocation tests; these do not execute CUDA kernels."""
from contextlib import nullcontext
import json
import os
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
import yaml

from scripts import select_gpus
from src.common import RESEARCH
import src.runtime as runtime


def settings():
    return yaml.safe_load((RESEARCH/'configs/gpu_100.yaml').read_text())


def inventory():
    names=['NVIDIA GeForce RTX 3090']*8
    names[2]=names[4]='Tesla V100S-PCIE-32GB'
    return '\n'.join(f'{i}, GPU-test-{i}, {name}, {18000 if i==5 else 10000}, 24576, 70'
                     for i,name in enumerate(names))


def test_homogeneous_selection_primary_has_headroom():
    selected=select_gpus.choose_3090(select_gpus.parse_inventory(inventory()))
    assert [gpu['index'] for gpu in selected]==[5,0,1,3,6,7]
    assert all(gpu['name']=='NVIDIA GeForce RTX 3090' for gpu in selected)


def test_selection_is_persisted_when_free_memory_changes(tmp_path,monkeypatch,capsys):
    monkeypatch.setattr(select_gpus,'ROOT',tmp_path)
    for key in ['CUDA_VISIBLE_DEVICES','SLURM_JOB_ID','PBS_JOBID','LSB_JOBID']:
        monkeypatch.delenv(key,raising=False)
    monkeypatch.setattr(select_gpus.subprocess,'run',lambda *a,**kw:SimpleNamespace(stdout=inventory()))
    select_gpus.main();first=capsys.readouterr().out.strip()
    changed=inventory().replace('18000','1000')
    monkeypatch.setattr(select_gpus.subprocess,'run',lambda *a,**kw:SimpleNamespace(stdout=changed))
    select_gpus.main();second=capsys.readouterr().out.strip()
    assert first==second and first.startswith('GPU-test-5,')
    assert len(json.loads((tmp_path/'outputs/launcher/rtx3090_selection.json').read_text())['uuids'])==6


def test_user_and_scheduler_mask_is_preserved(monkeypatch,capsys):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','3,5')
    monkeypatch.setenv('SLURM_JOB_ID','fixture')
    monkeypatch.setattr(select_gpus.subprocess,'run',lambda *a,**kw:pytest.fail('Must not query outside allocation'))
    select_gpus.main()
    assert capsys.readouterr().out.strip()=='3,5'
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES')
    with pytest.raises(RuntimeError,match='scheduler job'):select_gpus.main()


@pytest.mark.parametrize('existing_mask',[None,'3,5'])
def test_launcher_passes_selected_mask_to_pipeline(tmp_path,existing_mask):
    for name in ['run_3090.sh','run_server.sh']:
        shutil.copy2(RESEARCH/name,tmp_path/name)
    python=tmp_path/'fake_python'
    python.write_text(f'#!{sys.executable}\n'+'''import os,sys,json
if sys.argv[1:]==['scripts/select_gpus.py']:
    assert 'CUDA_VISIBLE_DEVICES' not in os.environ or os.environ['CUDA_VISIBLE_DEVICES']
    print(os.environ.get('CUDA_VISIBLE_DEVICES','GPU-test-a,GPU-test-b'))
else:
    print(json.dumps({'mask':os.environ['CUDA_VISIBLE_DEVICES'],
                     'allocator':os.environ['PYTORCH_ALLOC_CONF'],'args':sys.argv[1:]}))
''')
    python.chmod(0o755)
    env=dict(os.environ,CORESEG_PYTHON=str(python))
    env.pop('CUDA_VISIBLE_DEVICES',None);env.pop('PYTORCH_ALLOC_CONF',None)
    if existing_mask is not None:env['CUDA_VISIBLE_DEVICES']=existing_mask
    result=subprocess.run(['bash',str(tmp_path/'run_3090.sh'),'check'],env=env,
                          capture_output=True,text=True,check=True)
    data=json.loads(result.stdout)
    assert data['mask']==(existing_mask or 'GPU-test-a,GPU-test-b')
    assert data['allocator']=='expandable_segments:True'
    assert '--check-only' in data['args']


def test_mixed_hardware_uses_fp16_and_rejects_forced_bf16(monkeypatch):
    c=settings()
    monkeypatch.setattr(torch.cuda,'device_count',lambda:2)
    monkeypatch.setattr(torch.cuda,'get_device_capability',lambda i:(8,6) if i==0 else (7,0))
    monkeypatch.setattr(torch.cuda,'is_bf16_supported',lambda:True)
    assert runtime.amp_dtype(c)==torch.float16
    c['gpu']['amp_dtype']='bfloat16'
    with pytest.raises(RuntimeError,match='BF16 requires'):runtime.amp_dtype(c)
    c['gpu']['amp_dtype']='auto'
    monkeypatch.setattr(torch.cuda,'get_device_capability',lambda i:(8,6))
    assert runtime.amp_dtype(c)==torch.bfloat16


def test_auto_workers_respect_cpu_allocation(monkeypatch):
    c=settings();c['threads']=8
    monkeypatch.setattr(runtime.os,'sched_getaffinity',lambda pid:set(range(32)),raising=False)
    assert runtime.loader_workers(c)==8
    monkeypatch.setattr(runtime.os,'sched_getaffinity',lambda pid:set(range(10)))
    assert runtime.loader_workers(c)==0
    c['loader']['workers']=2
    assert runtime.loader_workers(c)==2


def test_batch_probe_reduces_for_secondary_gpu_pressure(monkeypatch):
    c=settings();c['gpu']['extraction_cap_per_gpu']=2;c['gpu']['amp']=False
    trial={'number':0};records=[]
    monkeypatch.setattr(torch.cuda,'device_count',lambda:2)
    monkeypatch.setattr(torch.cuda,'empty_cache',lambda:None)
    def reset(index):
        if index==0:trial['number']+=1
    monkeypatch.setattr(torch.cuda,'reset_peak_memory_stats',reset)
    monkeypatch.setattr(torch.cuda,'mem_get_info',lambda i:((100 if i==1 and trial['number']==1 else 1000),2000))
    monkeypatch.setattr(torch.cuda,'memory_allocated',lambda i:0)
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda i:300)
    monkeypatch.setattr(runtime,'device',lambda config:torch.device('cpu'))
    monkeypatch.setattr(runtime,'synchronize',lambda config:None)
    monkeypatch.setattr(runtime,'rng_state',lambda:{})
    monkeypatch.setattr(runtime,'restore_rng',lambda state:None)
    monkeypatch.setattr(runtime,'atomic_json',lambda path,value:records.append(value))
    model=torch.nn.Identity()
    assert runtime.tune_batch(c,model,'extraction')==2
    assert records[0]['probes'][0]['batch_size']==4
    assert not records[0]['probes'][0]['safe']
    assert records[0]['probes'][1]['safe']
    assert model.training


def test_cuda_preflight_checks_each_selected_device(monkeypatch):
    c=settings();c['gpu']['amp']=False;checked=[];real_ones=torch.ones
    monkeypatch.setattr(torch.cuda,'device_count',lambda:3)
    monkeypatch.setattr(torch.cuda,'device',lambda index:nullcontext())
    monkeypatch.setattr(torch.cuda,'synchronize',lambda index:checked.append(index))
    monkeypatch.setattr(torch,'ones',lambda *a,**kw:real_ones(*a))
    assert runtime.validate_cuda(c)
    assert checked==[0,1,2]


def test_cuda_preflight_reports_unsupported_secondary_device(monkeypatch):
    c=settings();c['gpu']['amp']=False;real_ones=torch.ones
    monkeypatch.setattr(torch.cuda,'device_count',lambda:2)
    monkeypatch.setattr(torch.cuda,'device',lambda index:nullcontext())
    monkeypatch.setattr(torch.cuda,'synchronize',lambda index:None)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda index:'Tesla V100S')
    def ones(*args,**kwargs):
        if kwargs['device']=='cuda:1':raise RuntimeError('no kernel image is available')
        return real_ones(*args)
    monkeypatch.setattr(torch,'ones',ones)
    with pytest.raises(RuntimeError,match='GPU 1.*V100S'):runtime.validate_cuda(c)

"""One independent SSH-server command: audit -> CUDA smoke -> CV -> final study."""
import csv
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import traceback
import torch
from .common import (RESEARCH,args,config,output,read_json,atomic_json,file_hash,
                     load_split,STOP,StopRequested)


class Tee:
    def __init__(self,console,log):self.console=console;self.log=log;self.lock=threading.Lock()
    def write(self,text):
        with self.lock:self.console.write(text);self.log.write(text);self.log.flush()
    def flush(self):self.console.flush();self.log.flush()
    def isatty(self):return False
    def __getattr__(self,name):return getattr(self.console,name)


def preflight(c):
    from .runtime import hardware,device
    root=Path(c['root'])
    for name in ['dataset-1','dataset-2','dataset-3']:
        if not (root/name).is_dir():raise FileNotFoundError(f'Missing {root/name}; pass --data-root /path/to/dataset-parent')
    weights=RESEARCH/c['backbone']['checkpoint']
    if file_hash(weights)!=c['backbone']['weight_sha256']:raise RuntimeError('Bundled DINO weights failed checksum')
    cuda_executed=False
    if c['device']=='cuda':
        x=torch.ones(16,16,device=device(c));value=(x@x).sum()
        if not torch.isfinite(value):raise RuntimeError('CUDA matrix test failed')
        torch.cuda.synchronize();cuda_executed=True
    report={'passed':True,'CUDA_execution_checked':cuda_executed,'hardware':hardware(c),
            'data_root':str(root),'weights_sha256':file_hash(weights),'max_epochs':c['max_epochs'],
            'early_stopping':c['early_stopping'],'CV_models':5 if c['cross_validation']['enabled'] else 0,
            'holdout_models':1,'package':str(RESEARCH)}
    atomic_json(output(c)/'reports/preflight.json',report);print(json.dumps(report,indent=2),flush=True)
    return report


def _audit_and_splits(c):
    from .storage import prepare_data
    from .audit_data import run as audit
    from .make_splits import run as split
    prepare_data(c);STOP.check()
    if not (RESEARCH/'manifests/all.json').exists():audit(c)
    if not (RESEARCH/'splits/protocol.json').exists():split(c)
    # Retain a clearly labeled report of correlated held-out source collections.
    allrows={r['id']:r for r in read_json(RESEARCH/'manifests/all.json')['rows']}
    test={r['id'] for r in load_split('test')};links=[]
    for edge in read_json(RESEARCH/'manifests/audit.json')['links']:
        if edge['a'] in test and edge['b'] in test and allrows[edge['a']]['collection']!=allrows[edge['b']]['collection']:links.append(edge)
    atomic_json(output(c)/'reports/heldout_source_overlap.json',{'links':links,'count':len(links),
                'note':'Held-out source partitions can be correlated; these links do not enter development.'})


def run(c,mode='all',skip_tests=False):
    from .extract_features import extract
    from .smoke_test import run as smoke
    from .cross_validate import run as cross_validate,protocol_for
    from .prototypes import build
    from .train import train_model
    from .evaluate import evaluate
    from .benchmark_runtime import run as benchmark
    from .visualize import run as visualize,history_plots
    from .report import run as report
    from .cv_plots import run as cv_plots
    base=output(c);base.mkdir(parents=True,exist_ok=True);stage='preflight'
    lock_path=RESEARCH/'outputs/.pipeline.lock';lock_path.parent.mkdir(parents=True,exist_ok=True)
    with lock_path.open('w') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another pipeline process is already running in this package directory')
        STOP.install()
        def mark(name):
            nonlocal stage
            stage=name
            atomic_json(base/'pipeline_status.json',{'completed':False,'mode':mode,'stage':name,'pid':os.getpid()})
        try:
            mark('preflight');preflight(c)
            if mode=='plots':
                visualize(c);atomic_json(base/'pipeline_status.json',{'completed':True,'mode':mode,'stage':'complete'});return
            mark('audit_and_splits');_audit_and_splits(c)
            if mode=='audit':
                atomic_json(base/'pipeline_status.json',{'completed':True,'mode':mode,'stage':'complete'});return
            if not skip_tests:
                mark('unit_tests');testlog=base/'logs/unit_tests.txt';testlog.parent.mkdir(parents=True,exist_ok=True)
                with testlog.open('w') as f:
                    result=subprocess.run([sys.executable,'-m','pytest','-q','tests'],cwd=RESEARCH,stdout=f,stderr=subprocess.STDOUT)
                if result.returncode:raise RuntimeError(f'Self-tests failed: {testlog}')
            mark('GPU_smoke_test');smoke(c);STOP.check()
            if mode!='evaluate':
                mark('development_feature_extraction')
                for split in ['train','val']:extract(c,load_split(split));STOP.check()
            if mode in ['all','cv'] and c['cross_validation']['enabled']:
                mark('five_fold_CV');cross_validate(c);STOP.check()
            if mode in ['all','holdout']:
                mark('holdout_training');directory=base/'holdout';train=load_split('train');val=load_split('val')
                directory.mkdir(parents=True,exist_ok=True)
                protos=build(c,train,[r['id'] for r in train],directory/'prototypes.pt',c['seed'])
                def plot_epoch(directory):
                    with (directory/'training_history.csv').open() as f:history=list(csv.DictReader(f))
                    epoch=int(history[-1]['epoch'])
                    if epoch==1 or epoch%c['artifacts']['plot_every_epochs']==0 or epoch==c['max_epochs']:history_plots(directory)
                train_model(c,train,val,protos,directory,protocol_for(train),c['seed'],on_epoch=plot_epoch)
            if mode in ['all','holdout','evaluate']:
                mark('final_test_extraction');extract(c,load_split('test'),allow_test=True)
                mark('final_six_partition_evaluation');directory=base/'holdout'
                evaluate(c,load_split('test'),directory,read_json(directory/'FROZEN.json')['protocol'])
                mark('end_to_end_runtime');benchmark(c,load_split('test'),directory)
            atomic_json(base/'pipeline_status.json',{'completed':True,'mode':mode,'stage':'complete'})
        except StopRequested as exc:
            atomic_json(base/'pipeline_status.json',{'completed':False,'stage':stage,'interrupted':True,'message':str(exc)})
            print(str(exc),flush=True)
            raise SystemExit(130)
        except BaseException as exc:
            atomic_json(base/'pipeline_status.json',{'completed':False,'stage':stage,'error':repr(exc)})
            traceback.print_exc();raise
        finally:
            # Plot/report failures must not erase the original failure/status.
            try:
                visualize(c)
                cvbase=RESEARCH/'outputs'/(c['experiment']+'_cv5')
                if cvbase.exists():cv_plots(cvbase,5,c['cross_validation']['roc_grid_points'])
                report(c)
            except Exception as exc:
                atomic_json(base/'reports/artifact_error.json',{'error':repr(exc),'stage':stage});traceback.print_exc()


if __name__=='__main__':
    a=args('Independent NVIDIA GPU 100-epoch study',lambda p:(p.add_argument('--mode',choices=['all','cv','holdout','evaluate','plots','audit'],default='all'),
        p.add_argument('--check-only',action='store_true'),p.add_argument('--allow-cpu-check',action='store_true'),p.add_argument('--skip-tests',action='store_true')))
    if a.allow_cpu_check and not a.check_only:raise SystemExit('--allow-cpu-check is only allowed with --check-only, never for experiment training')
    c=config(a.config,a.data_root,allow_cpu=a.allow_cpu_check)
    logs=output(c)/'logs';logs.mkdir(parents=True,exist_ok=True)
    with (logs/'pipeline.log').open('a',buffering=1) as log:
        original_out,original_err=sys.stdout,sys.stderr;sys.stdout=Tee(original_out,log);sys.stderr=Tee(original_err,log)
        try:
            if a.check_only:preflight(c)
            else:run(c,a.mode,a.skip_tests)
        finally:sys.stdout,sys.stderr=original_out,original_err

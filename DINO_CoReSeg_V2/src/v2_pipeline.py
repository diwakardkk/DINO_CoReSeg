"""Independent V2: preserve/audit -> smoke -> extract -> train -> frozen test."""
import argparse
import copy
import csv
import fcntl
import json
import os
from pathlib import Path
import sys
import subprocess
import ast
import traceback
import numpy as np
import torch
from .common import RESEARCH,STOP,StopRequested,atomic_json,read_json,file_hash
from .v2_config import load,output,cache_root,scientific,source_identity,software_identity
from .v2_preserve import capture,verify
from .v2_data import prepare
from .v2_features import extract
from .v2_prototypes import build
from .v2_train import train
from .v2_evaluate import evaluate
from .v2_benchmark import benchmark
from .v2_reporting import corrected_v1_report,report
from .runtime import hardware,validate_cuda


class Tee:
    def __init__(self,console,log):self.console=console;self.log=log
    def write(self,value):self.console.write(value);self.log.write(value);self.log.flush()
    def flush(self):self.console.flush();self.log.flush()
    def __getattr__(self,name):return getattr(self.console,name)


def protocol_for(rows):
    areas=[r['foreground_fraction'] for r in rows if r['positive']]
    return {'positive_area_median':float(np.median(areas)),
            'small_polyp_fraction_max':float(np.quantile(areas,1/3)),
            'medium_polyp_fraction_max':float(np.quantile(areas,2/3)),
            'definition':'Training-only native mask foreground-fraction tertiles'}


def smoke(c,splits):
    dest=output(c)/'smoke';marker=dest/'smoke_status.json'
    if marker.exists():
        status=read_json(marker)
        if status.get('source')!=source_identity():raise ValueError('Smoke identity changed; use a new experiment ID')
        if status['passed']:return status
    rows=splits['train'];positive=[r for r in rows if r['positive']];normal=[];patients=set()
    for row in rows:
        if not row['positive'] and row['patient'] not in patients:
            normal.append(row);patients.add(row['patient'])
            if len(normal)==4:break
    chosen=positive[:4]+normal
    if len(chosen)!=8:raise ValueError('Eight development samples required for real-data smoke test')
    extract(c,chosen)
    train_rows=chosen[:3]+chosen[4:7];val_rows=[chosen[3],chosen[7]]
    test=copy.deepcopy(c);test['max_epochs']=1;test['batch_size']=2;test['evaluation_batch_size']=2
    test['loader']['workers']=0;test['checkpoint']['every_steps']=1
    dest.mkdir(parents=True,exist_ok=True)
    protos=build(test,train_rows,[r['id'] for r in train_rows],dest/'prototypes.pt')
    first=train(test,train_rows,val_rows,protos,protocol_for(train_rows),dest)
    second=train(test,train_rows,val_rows,protos,protocol_for(train_rows),dest)
    if first['checkpoint_sha256']!=second['checkpoint_sha256']:raise ValueError('V2 checkpoint resume smoke failed')
    result={'passed':True,'source':source_identity(),'device':c['device'],'samples':8,'epochs':1,
            'shape':[c['feature_dim'],c['grid_size'],c['grid_size']],'hardware':hardware(c),
            'purpose':'CUDA integration only; not a scientific performance result','checkpoint_resume_checked':True}
    atomic_json(marker,result);return result


def cv(c,splits):
    from .cv_splits import canonical_groups,partition_check
    from .common import csv_write
    from .v2_evaluate import PixelROC
    from .plot_paper import generate
    c=copy.deepcopy(c);c['experiment']+='_cv5';base=output(c)
    rows=sorted(splits['train']+splits['val'],key=lambda r:r['id']);lookup={r['id']:r for r in rows}
    frozen_folder=Path(c['v1_root'])/'outputs/dino_coreseg_gpu_100_cv5/splits'
    folds=[read_json(frozen_folder/f'fold_{i:02d}.json') for i in range(1,6)];results=[]
    indices={row['id']:i for i,row in enumerate(rows)};groups=canonical_groups(rows)
    heldout_ids=[]
    for fold in folds:
        partition_check(rows,[[indices[key] for key in fold[field]] for field in ['train_ids','inner_val_ids','heldout_ids']],groups)
        heldout_ids+=fold['heldout_ids']
    if len(heldout_ids)!=len(set(heldout_ids)) or set(heldout_ids)!=set(indices):raise ValueError('Frozen V1 outer folds do not cover the retained development pool once')
    atomic_json(base/'config/fold_manifest.json',folds,immutable=True)
    for fold in folds:
        STOP.check();test=copy.deepcopy(c);test['seed']=fold['seed'];directory=base/f"fold_{fold['fold']:02d}"
        training=[lookup[k] for k in fold['train_ids']];validation=[lookup[k] for k in fold['inner_val_ids']];heldout=[lookup[k] for k in fold['heldout_ids']]
        protos=build(test,training,fold['train_ids'],directory/'checkpoints/prototypes.pt')
        protocol=protocol_for(training);train(test,training,validation,protos,protocol,directory)
        evaluate(test,heldout,protocol,directory,outer=True)
        with (directory/'evaluation/metrics/final_metrics.csv').open() as stream:
            results.extend({'fold':fold['fold'],**row} for row in csv.DictReader(stream))
        generate(base,base/'figures')
    csv_write(base/'evaluation/metrics/fold_metrics.csv',results)
    summary={'completed':True,'folds':5,'membership':'Exact imported V1 frozen fold IDs','evaluation_grid':'native'}
    for target in ['positive','normal']:
        subset=[row for row in results if row['target']==target]
        for key in ['dice','iou','recall','fp_frame','frame_specificity']:
            values=[float(row[key]) for row in subset if row.get(key) not in ['',None]]
            if values:summary[target+'_'+key]={'mean':float(np.mean(values)),'std':float(np.std(values,ddof=1))}
    atomic_json(base/'evaluation/metrics/cv_summary.json',summary)
    aucs=[];pooled=PixelROC(c['evaluation']['roc_bins'])
    for fold in folds:
        with np.load(base/f"fold_{fold['fold']:02d}/evaluation/metrics/outer_heldout_roc.npz",allow_pickle=False) as z:
            aucs.append(float(z['auc']));pooled.positive+=z['positive_hist'];pooled.negative+=z['negative_hist']
    summary['pixel_auc']={'mean':float(np.mean(aucs)),'std':float(np.std(aucs,ddof=1))}
    pooled.save(base/'evaluation/metrics/pooled_heldout_roc.npz')
    atomic_json(base/'evaluation/metrics/cv_summary.json',summary)
    atomic_json(base/'logs/status.json',{'completed':True,'folds':5})


def run(c,stage):
    base=output(c);base.mkdir(parents=True,exist_ok=True)
    lock_path=base/'logs/.pipeline.lock';lock_path.parent.mkdir(parents=True,exist_ok=True)
    with lock_path.open('w') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('This V2 experiment already has a writer')
        STOP.install();current='preflight'
        def mark(name,**kwargs):
            nonlocal current
            current=name;atomic_json(base/'logs/pipeline_status.json',{'completed':False,'stage':name,'pid':os.getpid(),**kwargs})
        try:
            mark('preflight');validate_cuda(c)
            for source in (RESEARCH/'src').glob('*.py'):ast.parse(source.read_text(),filename=str(source))
            if file_hash(RESEARCH/c['backbone']['checkpoint'])!=c['backbone']['weight_sha256']:raise ValueError('V2 pretrained weights failed verification')
            atomic_json(base/'config/resolved_paths.json',{'root':c['root'],'v1_root':c['v1_root'],'v2_root':c['research']})
            atomic_json(base/'config/environment.json',{'software':software_identity(),'source':source_identity(),'hardware':hardware(c)})
            atomic_json(base/'reports/preflight.json',{'CUDA_checked':True,'hardware':hardware(c),'source':source_identity()})
            if stage=='check':return
            mark('preserving_v1');guard=capture(c)
            if stage=='cleanup-dry-run':
                from .v2_cleanup import dry_run
                dry_run(c);return
            if stage=='report-v1':corrected_v1_report(c);verify(c['v1_root'],guard);return
            mark('verifying_frozen_dataset_splits');splits=prepare(c)
            if stage=='prepare':return
            if stage in ['all','smoke']:
                mark('CPU_unit_tests')
                log=base/'logs/unit_tests.log'
                with log.open('w') as stream:
                    result=subprocess.run([sys.executable,'-m','pytest','-q','tests'],cwd=RESEARCH,
                        env={**os.environ,'CUDA_VISIBLE_DEVICES':''},stdout=stream,stderr=subprocess.STDOUT)
                if result.returncode:raise RuntimeError(f'V2 unit tests failed; see {log}')
                mark('CUDA_real_data_smoke');smoke(c,splits)
                if stage=='smoke':return
            if stage in ['all','extract','cv']:
                mark('development_features');extract(c,splits['train']);extract(c,splits['val'])
                if stage=='extract':return
            if stage=='cv':mark('five_fold_CV');cv(c,splits);return
            protocol=protocol_for(splits['train'])
            if stage in ['all','train']:
                if not (base/'smoke/smoke_status.json').exists():raise ValueError('Run the CUDA smoke stage before full training')
                mark('holdout_training')
                protos=build(c,splits['train'],[row['id'] for row in splits['train']],base/'checkpoints/prototypes.pt')
                train(c,splits['train'],splits['val'],protos,protocol)
                if stage=='train':return
            if stage in ['all','evaluate']:
                mark('frozen_test_features');extract(c,splits['test'],allow_test=True)
                mark('final_test_evaluation');evaluate(c,splits['test'],protocol)
                mark('live_end_to_end_benchmark');benchmark(c,splits['test'])
                mark('paper_figures')
                from .plot_paper import generate
                generate(base,base/'figures',c['root'])
                report(c);verify(c['v1_root'],guard)
                atomic_json(base/'logs/pipeline_status.json',{'completed':True,'stage':'complete','mode':stage})
        except StopRequested:
            atomic_json(base/'logs/pipeline_status.json',{'completed':False,'interrupted':True,'stage':current});raise SystemExit(130)
        except BaseException as exc:
            atomic_json(base/'logs/pipeline_status.json',{'completed':False,'stage':current,'error':repr(exc)})
            traceback.print_exc();raise
        finally:
            # A stage may finish successfully without declaring the full study done.
            primary_error=sys.exc_info()[0] is not None
            try:
                report(c)
                if not primary_error:
                    status=read_json(base/'logs/pipeline_status.json')
                    if not status.get('completed'):
                        atomic_json(base/'logs/pipeline_status.json',{'completed':False,'stage_completed':True,
                                    'stage':stage+'_complete','mode':stage,'note':'This requested stage finished; the full study is not declared complete.'})
            except Exception as exc:
                atomic_json(base/'reports/artifact_error.json',{'error':repr(exc),'stage':current})
                if not primary_error:raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(RESEARCH/'configs/v2_resolution_448.yaml'))
    parser.add_argument('--data-root');parser.add_argument('--v1-root')
    parser.add_argument('--stage',choices=['check','prepare','smoke','extract','train','evaluate','cv','report-v1','cleanup-dry-run','all'],default='all')
    args=parser.parse_args()
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:parser.error('Set CUDA_VISIBLE_DEVICES to your allocated GPU(s); V2 does not select shared GPUs automatically')
    c=load(args.config,args.data_root,args.v1_root);base=output(c);logs=base/'logs';logs.mkdir(parents=True,exist_ok=True)
    with (logs/'pipeline.log').open('a',buffering=1) as stream:
        stdout,stderr=sys.stdout,sys.stderr;sys.stdout=Tee(stdout,stream);sys.stderr=Tee(stderr,stream)
        try:run(c,args.stage)
        finally:sys.stdout,sys.stderr=stdout,stderr


if __name__=='__main__':main()

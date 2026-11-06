from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import resource
import sys
import time
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from .common import (RESEARCH,read_json,atomic_json,csv_write,file_hash,fingerprint,pair_arrays,STOP)
from .model_coreseg import DINOCoReSeg
from .extract_features import CachedDataset
from .metrics import positive_metrics,normal_metrics,macro
from .cv_metrics import PixelROC
from .runtime import device,parallelize,make_loader,move_batch,synchronize,hardware


def export_binary(path,pred):Image.fromarray(np.asarray(pred,bool).astype(np.uint8)*255).save(path)


def size_bin(area,protocol):
    if area<=protocol['small_polyp_fraction_max']:return 'small'
    if area<=protocol['medium_polyp_fraction_max']:return 'medium'
    return 'large'


def patient_bootstrap(rows,metric,seed,n=500):
    groups=defaultdict(list)
    for row in rows:
        if row.get('patient') and metric in row:groups[row['patient']].append(float(row[metric]))
    if len(groups)<2:return None
    values=list(groups.values());rng=np.random.default_rng(seed);estimates=[]
    for _ in range(n):
        sample=rng.integers(len(values),size=len(values))
        estimates.append(np.mean([v for i in sample for v in values[int(i)]]))
    low,high=np.percentile(estimates,[2.5,97.5])
    return {'lower':float(low),'upper':float(high),'replicates':n,'unit':'patient clusters'}


def _evaluate_one(c,row,prob350,counter,directory,checkpoint_sha,protocol,seconds,partition):
    key=fingerprint(row['id']);dest=directory/'predictions'/partition;dest.mkdir(parents=True,exist_ok=True)
    result_path=dest/f'{key}.json';prob_path=dest/f'{key}.npz';png=dest/f'{key}.png'
    image,truth=pair_arrays(c,row)
    if result_path.exists() and png.exists() and prob_path.exists():
        existing=read_json(result_path)
        if existing['checkpoint_sha256']!=checkpoint_sha:raise RuntimeError('Evaluation prediction checkpoint changed')
        with np.load(prob_path,allow_pickle=False) as z:prob=z['probability'].astype(np.float32)
        record=existing['metrics']
    else:
        prob=F.interpolate(torch.from_numpy(prob350)[None,None],size=(image.height,image.width),mode='bilinear',align_corners=False)[0,0].numpy()
        pred=prob>=c['threshold']
        record={'id':row['id'],'partition':partition,'patient':row['patient'],'positive':row['positive'],
                'source':row['source'],
                'width':image.width,'height':image.height,'prediction_key':key,'decoder_seconds_per_image':seconds,
                'mask_kind':row['mask_kind']}
        record.update(pixel_accuracy=float((pred==truth).mean()),TP=int((pred&truth).sum()),
                      FP=int((pred&~truth).sum()),FN=int((~pred&truth).sum()),TN=int((~pred&~truth).sum()))
        if row['positive']:
            record.update(positive_metrics(prob,truth,c['threshold'],c['boundary_tolerance_pixels'],c['boundary_iou_fraction']))
            record['size_bin']=size_bin(float(truth.mean()),protocol)
        else:
            record.update(normal_metrics(prob,c['threshold'],c['normal_component_gates']))
            p256=F.interpolate(torch.from_numpy(prob350)[None,None],size=(256,256),mode='bilinear',align_corners=False)[0,0].numpy()
            record.update({'exploratory_grid256_'+k:v for k,v in normal_metrics(p256,c['threshold'],c['normal_component_gates']).items()})
        export_binary(png,pred)
        np.savez_compressed(prob_path,probability=prob,counter_evidence=counter.astype(np.float16))
        atomic_json(result_path,{'row':row,'metrics':record,'checkpoint_sha256':checkpoint_sha,'threshold':c['threshold']})
    roc=PixelROC(c['evaluation']['roc_bins']);roc.update(prob,truth)
    return record,roc.positive,roc.negative


def evaluate(c,rows,directory,protocol,partition_mode='source'):
    directory=Path(directory);marker=read_json(directory/'FROZEN.json');checkpoint=directory/'best.pt'
    sha=file_hash(checkpoint)
    if sha!=marker['checkpoint_sha256']:raise ValueError('Frozen checkpoint checksum mismatch')
    if marker['threshold']!=c['threshold']:raise ValueError('Frozen threshold mismatch')
    status=directory/'evaluation_status.json'
    if status.exists() and read_json(status).get('completed',False):
        if read_json(status)['checkpoint_sha256']!=sha:raise ValueError('Completed evaluation identity mismatch')
        return read_json(status)
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if state['identity']['backbone']!=read_json(RESEARCH/'cache/backbone_identity.json'):raise ValueError('Evaluation cache model differs from training')
    raw=DINOCoReSeg(state['model']['prototypes']);raw.load_state_dict(state['model']);model=parallelize(raw,c).eval()
    batch_size=read_json(directory/'runtime.json')['evaluation_batch_size']
    loader=make_loader(CachedDataset(rows),c,batch_size=batch_size,seed=c['seed']);lookup={r['id']:r for r in rows}
    workers=c['evaluation']['metric_workers'];records=[];rocs={};pending=[];times=[]
    def collect(future):
        record,pos,neg=future.result();records.append(record)
        name=record['partition'];rocs.setdefault(name,PixelROC(c['evaluation']['roc_bins']))
        rocs[name].positive+=pos;rocs[name].negative+=neg
    with ThreadPoolExecutor(max_workers=workers) as pool,torch.inference_mode():
        for batch in loader:
            STOP.check();batch=move_batch(batch,c);synchronize(c);tick=time.perf_counter()
            out=model(batch['features'],batch['rgb']);synchronize(c);elapsed=time.perf_counter()-tick
            probabilities=out['final'].float().sigmoid().cpu().numpy();counter=out['counter_evidence'].float().cpu().numpy()
            times.append((elapsed,len(probabilities)))
            for i,key in enumerate(batch['id']):
                row=lookup[key];partition='outer_heldout' if partition_mode=='outer' else ('FAR' if row['domain']=='FAR' else row['source'])
                pending.append(pool.submit(_evaluate_one,c,row,probabilities[i,0],counter[i,0],directory,sha,protocol,elapsed/len(probabilities),partition))
                if len(pending)>=2*workers:collect(pending.pop(0))
            atomic_json(directory/'evaluation_progress.json',{'completed_images':len(records),'total':len(rows)})
        for future in pending:collect(future)
    records.sort(key=lambda r:r['id']);metrics=directory/'metrics';metrics.mkdir(parents=True,exist_ok=True)
    csv_write(metrics/'per_image.csv',records);summary=[];sizes=[];failures=[]
    for partition in sorted(rocs):
        group=[r for r in records if r['partition']==partition]
        for positive in [True,False]:
            chosen=[r for r in group if r['positive']==positive]
            if not chosen:continue
            values=[{k:v for k,v in r.items() if k not in ['id','partition','patient','positive','width','height','prediction_key','size_bin','mask_kind']} for r in chosen]
            summary.append({'partition':partition,'target':'positive' if positive else 'normal','images':len(chosen),**macro(values)})
        for size in ['small','medium','large']:
            chosen=[r for r in group if r.get('size_bin')==size]
            if chosen:sizes.append({'partition':partition,'size_bin':size,'images':len(chosen),**macro([{k:r[k] for k in ['dice','iou','recall','hd95','assd']} for r in chosen])})
        failures.append({'partition':partition,'surface_failure_count':sum(r.get('surface_failure',0) for r in group),
                         'empty_positive_prediction_count':sum(r.get('empty_prediction',0) for r in group)})
        roc=rocs[partition];roc.save(metrics/f'{partition}_roc.npz')
    csv_write(metrics/'final_metrics.csv',summary);csv_write(metrics/'lesion_size_metrics.csv',sizes);csv_write(metrics/'surface_failures.csv',failures)
    source_summary=[]
    for source in sorted({r['source'] for r in records}):
        for positive in [True,False]:
            group=[r for r in records if r['source']==source and r['positive']==positive]
            if not group:continue
            values=[{k:v for k,v in r.items() if k not in ['id','partition','patient','source','positive','width','height','prediction_key','size_bin','mask_kind']} for r in group]
            source_summary.append({'source':source,'target':'positive' if positive else 'normal','images':len(group),**macro(values)})
    csv_write(metrics/'source_metrics.csv',source_summary)
    validation_dice=state['history'][-1]['val_positive_dice']
    csv_write(metrics/'domain_gaps.csv',[{'partition':r['partition'],'selected_validation_dice':validation_dice,
              'test_dice':r['dice'],'validation_minus_test':validation_dice-r['dice'],
              'interpretation':'Descriptive mixed-source validation gap, not a causal estimate of domain shift'}
              for r in summary if r['target']=='positive'])
    far=[r for r in records if r['patient']]
    atomic_json(metrics/'patient_bootstrap.json',{'dice':patient_bootstrap([r for r in far if r['positive']],'dice',c['seed'],c['evaluation']['bootstrap_replicates']),
                'normal_fpr':patient_bootstrap([r for r in far if not r['positive']],'fp_frame',c['seed'],c['evaluation']['bootstrap_replicates'])})
    atomic_json(metrics/'runtime.json',{'hardware':hardware(c),'cached_decoder_seconds_per_image':sum(t for t,n in times)/sum(n for t,n in times),
                'trainable_parameters':sum(p.numel() for p in raw.parameters()),
                'backbone_parameters':read_json(RESEARCH/'cache/backbone_identity.json')['parameters'],
                'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated() if c['device']=='cuda' else None,
                'gpu_memory':[{'device':i,'peak_allocated_bytes':torch.cuda.max_memory_allocated(i),
                               'peak_reserved_bytes':torch.cuda.max_memory_reserved(i)}
                              for i in range(torch.cuda.device_count())] if c['device']=='cuda' else [],
                'process_lifetime_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
                'cache_bytes':sum(p.stat().st_size for p in (RESEARCH/'cache').glob('*.npz')),
                'timing_scope':'Cached decoder only; end-to-end inference is measured separately by benchmark_runtime.py.'})
    atomic_json(status,{'completed':True,'images':len(records),'checkpoint_sha256':sha,'partitions':sorted(rocs)})
    return read_json(status)

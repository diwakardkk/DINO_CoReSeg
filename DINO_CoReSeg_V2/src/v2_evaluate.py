"""Frozen V2 native/standard-grid evaluation with bounded permanent artifacts."""
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from pathlib import Path
import json
import time
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from .common import atomic_json,atomic_torch,csv_write,read_json,file_hash,fingerprint,pair_arrays,STOP
from .metrics import positive_metrics,normal_metrics,macro
from .runtime import parallelize,device,make_loader,move_batch,synchronize,hardware
from .v2_config import output,cache_root,scientific
from .v2_model import DINOCoReSegV2
from .v2_features import CachedDataset
from .v2_metrics import detection
from .v2_reporting import selected_record


class PixelROC:
    def __init__(self,bins=65536):self.bins=bins;self.positive=np.zeros(bins,np.int64);self.negative=np.zeros(bins,np.int64)
    def update(self,prob,truth):
        if prob.shape!=truth.shape or not np.isfinite(prob).all():raise ValueError('Invalid ROC inputs')
        labels=np.asarray(truth,bool);indices=np.minimum((np.clip(prob,0,1)*self.bins).astype(np.int64),self.bins-1)
        self.positive+=np.bincount(indices[labels],minlength=self.bins);self.negative+=np.bincount(indices[~labels],minlength=self.bins)
    def save(self,path):
        if not self.positive.sum() or not self.negative.sum():raise ValueError('ROC requires positive and negative pixels')
        tpr=np.r_[0,np.cumsum(self.positive[::-1])/self.positive.sum()];fpr=np.r_[0,np.cumsum(self.negative[::-1])/self.negative.sum()]
        np.savez_compressed(path,positive_hist=self.positive,negative_hist=self.negative,fpr=fpr,tpr=tpr,auc=np.trapezoid(tpr,fpr),bins=self.bins)


def resize(prob,shape):
    return F.interpolate(torch.as_tensor(prob,dtype=torch.float32)[None,None],size=shape,mode='bilinear',align_corners=False)[0,0].numpy()


def choose_qualitative(records,c):
    choices=[]
    for partition in ['Kvasir','CVC-ClinicDB','CVC-ColonDB','ETIS-LaribPolypDB','FAR']:
        group=sorted((r for r in records if r['partition']==partition and r['positive']),key=lambda row:(row['dice'],row['id']))
        if group:
            q=c['evaluation']['qualitative_quantiles'][partition];row=dict(group[int(q*(len(group)-1))]);row['selection_reason']=f'Predeclared positive Dice quantile {q:g}';choices.append(row)
    normals=sorted((r for r in records if not r['positive']),key=lambda row:row['id'])
    if normals:
        row=dict(normals[(len(normals)-1)//2]);row['selection_reason']='Predeclared median identifier among healthy images; no accuracy selection';choices.append(row)
    return choices


def evaluate_one(c,row,prob_model,directory,checkpoint_sha,protocol,seconds,partition):
    key=fingerprint(row['id']);folder=directory/'predictions'/partition;folder.mkdir(parents=True,exist_ok=True)
    image,truth=pair_arrays(c,row);prob=resize(prob_model,(image.height,image.width));pred=prob>=c['threshold']
    sidecar=folder/(key+'.json');png=folder/(key+'.png')
    if sidecar.exists() and png.exists():
        saved=read_json(sidecar)
        if saved['checkpoint_sha256']!=checkpoint_sha or saved['row']!=row:raise ValueError('V2 saved prediction identity mismatch')
        if not np.array_equal(np.asarray(Image.open(png))>0,pred):raise ValueError('Repeated inference differs from committed binary prediction')
        record=saved['metrics']
    else:
        record={'id':row['id'],'source':row['source'],'partition':partition,'patient':row['patient'],'positive':row['positive'],
                'width':image.width,'height':image.height,'mask_kind':row['mask_kind'],'prediction_key':key,
                'decoder_seconds_per_image':seconds,'pixel_accuracy':float((pred==truth).mean())}
        record.update(TP=int((pred&truth).sum()),FP=int((pred&~truth).sum()),FN=int((~pred&truth).sum()),TN=int((~pred&~truth).sum()))
        if row['positive']:
            record.update(positive_metrics(prob,truth,c['threshold'],c['boundary_tolerance_pixels'],c['boundary_iou_fraction']))
            record.update(detection(prob,truth,c['threshold'],c['lesion_detection_iou']))
            area=float(truth.mean());record['size_bin']='small' if area<=protocol['small_polyp_fraction_max'] else ('medium' if area<=protocol['medium_polyp_fraction_max'] else 'large')
            standard=c['evaluation']['standard_size'];p=resize(prob_model,(standard,standard))
            y=np.asarray(Image.fromarray(truth.astype(np.uint8)*255).resize((standard,standard),Image.Resampling.NEAREST))>0
            if y.any():
                standard_metrics=positive_metrics(p,y,c['threshold'],c['boundary_tolerance_pixels'],c['boundary_iou_fraction'])
                record.update({'standard_'+k:v for k,v in standard_metrics.items()});record['standard_target_empty']=0
            else:record['standard_target_empty']=1
        else:
            record.update(normal_metrics(prob,c['threshold'],c['normal_component_gates']))
            standard=c['evaluation']['standard_size']
            record.update({'standard_'+k:v for k,v in normal_metrics(resize(prob_model,(standard,standard)),c['threshold'],c['normal_component_gates']).items()})
        Image.fromarray(pred.astype(np.uint8)*255).save(png)
        atomic_json(sidecar,{'row':row,'checkpoint_sha256':checkpoint_sha,'threshold':c['threshold'],'metrics':record})
    roc=PixelROC(c['evaluation']['roc_bins']);roc.update(prob,truth)
    return record,roc.positive,roc.negative


def patient_bootstrap(rows,metric,seed,count):
    groups=defaultdict(list)
    for row in rows:
        if row.get('patient') and row.get(metric) is not None:groups[row['patient']].append(float(row[metric]))
    if len(groups)<2:return None
    values=list(groups.values());rng=np.random.default_rng(seed);estimates=[]
    for _ in range(count):
        draws=rng.integers(len(values),size=len(values));estimates.append(float(np.mean([value for index in draws for value in values[index]])))
    low,high=np.quantile(estimates,[.025,.975]);return {'lower':float(low),'upper':float(high),'replicates':count,'unit':'patient-cluster resampling; image-weighted statistic'}


def evaluate(c,rows,protocol,directory=None,outer=False):
    directory=Path(directory) if directory else output(c);frozen=read_json(directory/'checkpoints/FROZEN.json');checkpoint=directory/'checkpoints/best.pt'
    sha=file_hash(checkpoint)
    if sha!=frozen['checkpoint_sha256'] or frozen['threshold']!=c['threshold']:raise ValueError('Frozen V2 model/threshold mismatch')
    status=directory/'evaluation/evaluation_status.json'
    if status.exists() and read_json(status).get('completed'):
        if read_json(status)['checkpoint_sha256']!=sha:raise ValueError('Completed evaluation checkpoint differs')
        return read_json(status)
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if state['identity']['cache']!=read_json(cache_root(c)/'identity.json') or state['identity']['config']!=fingerprint(scientific(c)):raise ValueError('V2 evaluation configuration/cache differs')
    raw=DINOCoReSegV2(c,state['model']['prototypes']);raw.load_state_dict(state['model']);model=parallelize(raw,c).eval()
    batch_size=read_json(directory/'config/hardware.json')['evaluation_batch_size'];lookup={row['id']:row for row in rows}
    groups=[rows[i:i+batch_size] for i in range(0,len(rows),batch_size)]
    atomic_json(directory/'evaluation/batch_membership.json',[[row['id'] for row in group] for group in groups],immutable=True)
    resume=directory/'evaluation/resume.pt';records=[];rocs={};first_batch=0
    if resume.exists():
        saved=torch.load(resume,map_location='cpu',weights_only=False)
        if saved['checkpoint_sha256']!=sha or saved['row_ids']!=[row['id'] for row in rows]:raise ValueError('Evaluation resume identity differs')
        records=saved['records'];first_batch=saved['next_batch']
        for partition,hists in saved['histograms'].items():
            roc=PixelROC(c['evaluation']['roc_bins']);roc.positive=hists[0];roc.negative=hists[1];rocs[partition]=roc
    with ThreadPoolExecutor(max_workers=c['evaluation']['metric_workers']) as pool,torch.inference_mode():
        for batch_number,group in enumerate(groups):
            if batch_number<first_batch:continue
            STOP.check();loader=make_loader(CachedDataset(c,group),{**c,'loader':{**c['loader'],'workers':0}},batch_size=len(group))
            batch=move_batch(next(iter(loader)),c);synchronize(c);tick=time.perf_counter()
            logits=model(batch['features'],batch['rgb'])['final'];synchronize(c);seconds=(time.perf_counter()-tick)/len(group)
            probabilities=logits.float().sigmoid().cpu().numpy()[:,0]
            def task(item):
                row,prob=item;partition='outer_heldout' if outer else ('FAR' if row['domain']=='FAR' else row['source'])
                return evaluate_one(c,row,prob,directory,sha,protocol,seconds,partition)
            for record,positive_hist,negative_hist in pool.map(task,zip(group,probabilities)):
                records.append(record);roc=rocs.setdefault(record['partition'],PixelROC(c['evaluation']['roc_bins']))
                roc.positive+=positive_hist;roc.negative+=negative_hist
            atomic_torch(resume,{'checkpoint_sha256':sha,'row_ids':[row['id'] for row in rows],'next_batch':batch_number+1,
                                'records':records,'histograms':{key:(roc.positive,roc.negative) for key,roc in rocs.items()}})
            atomic_json(directory/'evaluation/progress.json',{'stage':'evaluating','completed_images':len(records),'total':len(rows)})
    records.sort(key=lambda r:r['id']);metrics=directory/'evaluation/metrics';metrics.mkdir(parents=True,exist_ok=True)
    csv_write(metrics/'per_image.csv',records);summary=[];sizes=[];failures=[];valid_surfaces=[]
    exclude={'id','source','partition','patient','positive','width','height','mask_kind','prediction_key','size_bin'}
    for partition in sorted(rocs):
        group=[row for row in records if row['partition']==partition]
        for positive in [True,False]:
            chosen=[r for r in group if r['positive']==positive]
            if chosen:summary.append({'partition':partition,'target':'positive' if positive else 'normal','images':len(chosen),
                                      **macro([{k:v for k,v in r.items() if k not in exclude} for r in chosen])})
        pos=[r for r in group if r['positive']]
        for size in ['small','medium','large']:
            chosen=[r for r in pos if r['size_bin']==size]
            if chosen:sizes.append({'partition':partition,'size_bin':size,'images':len(chosen),**macro([{key:r[key] for key in ['dice','iou','recall','empty_prediction','lesion_detection_recall']} for r in chosen])})
        valid=[r for r in pos if not r['surface_failure']]
        if valid:valid_surfaces.append({'partition':partition,'valid_predictions':len(valid),'excluded_empty_predictions':len(pos)-len(valid),
                                       **macro([{key:r[key] for key in ['hd95','assd','boundary_f1','boundary_iou']} for r in valid])})
        failures.extend({'partition':partition,'id':r['id'],'reason':'Empty positive prediction; native diagonal penalty retained','hd95':r['hd95'],'assd':r['assd']} for r in pos if r['surface_failure'])
        rocs[partition].save(metrics/f'{partition}_roc.npz')
    csv_write(metrics/'final_metrics.csv',summary);csv_write(metrics/'lesion_size_metrics.csv',sizes)
    source_summary=[]
    for source in sorted({r['source'] for r in records}):
        for positive in [True,False]:
            chosen=[r for r in records if r['source']==source and r['positive']==positive]
            if chosen:source_summary.append({'source':source,'target':'positive' if positive else 'normal','images':len(chosen),
                                            **macro([{k:v for k,v in row.items() if k not in exclude} for row in chosen])})
    csv_write(metrics/'source_metrics.csv',source_summary)
    csv_write(metrics/'surface_failures.csv',failures or [{'partition':'none','reason':'No empty positive predictions'}])
    csv_write(metrics/'valid_surface_only.csv',valid_surfaces)
    validation=selected_record(state,frozen)
    csv_write(metrics/'domain_gaps.csv',[{'partition':row['partition'],'selected_epoch':frozen['selected_epoch'],
        'selected_validation_dice':validation['validation_positive_dice'],'native_test_dice':row['dice'],
        'validation_minus_test':validation['validation_positive_dice']-row['dice'],
        'interpretation':'Descriptive mixed-source validation/native-test gap; different grids, not a causal domain-shift estimate'} for row in summary if row['target']=='positive'])
    far=[r for r in records if r['patient']]
    atomic_json(metrics/'patient_bootstrap.json',{'dice':patient_bootstrap([r for r in far if r['positive']],'dice',c['seed'],c['evaluation']['bootstrap_replicates']),
                                               'normal_fpr':patient_bootstrap([r for r in far if not r['positive']],'fp_frame',c['seed'],c['evaluation']['bootstrap_replicates'])})
    atomic_json(metrics/'metric_protocol.json',{'native_distance_units':'original-image pixels','standardized_size':c['evaluation']['standard_size'],
        'standard_prefix':'standard_','empty_surface_penalty':'image diagonal','valid_surface_only':'separate descriptive subset; failures must be reported',
        'lesion_detection':'8-connected GT/predicted components; greedy one-to-one matching at configured IoU',
        'lesion_iou_threshold':c['lesion_detection_iou'],'threshold':c['threshold'],'ROC_bins':c['evaluation']['roc_bins']})
    atomic_json(metrics/'runtime.json',{'hardware':hardware(c),'trainable_parameters':sum(p.numel() for p in raw.parameters()),
        'frozen_backbone_parameters':read_json(cache_root(c)/'identity.json')['parameters'],
        'cached_decoder_seconds_per_image':float(np.mean([r['decoder_seconds_per_image'] for r in records])),
        'scope':'Cached-feature decoder forward only; not end-to-end throughput'})
    if not outer:
        selected=choose_qualitative(records,c);selected_ids={row['id'] for row in selected}
        with torch.inference_mode():
            for group in groups:
                if not selected_ids&{row['id'] for row in group}:continue
                batch=next(iter(make_loader(CachedDataset(c,group),{**c,'loader':{**c['loader'],'workers':0}},batch_size=len(group))))
                batch=move_batch(batch,c);probabilities=model(batch['features'],batch['rgb'])['final'].float().sigmoid().cpu().numpy()[:,0]
                for row,prob_model in zip(group,probabilities):
                    if row['id'] not in selected_ids:continue
                    partition='FAR' if row['domain']=='FAR' else row['source'];key=fingerprint(row['id'])
                    image,_=pair_arrays(c,row,include_mask=False);prob=resize(prob_model,(image.height,image.width))
                    path=directory/'predictions'/partition/key
                    if not np.array_equal(np.asarray(Image.open(path.with_suffix('.png')))>0,prob>=c['threshold']):
                        raise ValueError('Selected probability map does not reproduce the committed native mask')
                    np.savez_compressed(path.with_suffix('.npz'),probability=prob)
        atomic_json(directory/'evaluation/qualitative_candidates.json',{'checkpoint_sha256':sha,'records':selected})
    result={'completed':True,'images':len(records),'checkpoint_sha256':sha,'partitions':sorted(rocs)}
    atomic_json(status,result);atomic_json(directory/'evaluation/progress.json',{'stage':'complete','completed_images':len(records),'total':len(rows)})
    return result

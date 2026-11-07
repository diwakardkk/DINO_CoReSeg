"""Explicit overlap, component-detection and small-lesion monitoring."""
import numpy as np
import torch
from scipy import ndimage as ndi
from torch.nn import functional as F
from .v2_losses import objective
from .v2_features import CachedDataset
from .runtime import make_loader,move_batch,autocast
from .common import STOP


def detection(prob,truth,threshold=.5,iou_threshold=.1):
    truth=np.asarray(truth,bool);pred=np.asarray(prob)>=threshold
    gt,n=ndi.label(truth,structure=np.ones((3,3)));pr,m=ndi.label(pred,structure=np.ones((3,3)))
    if not n:return {'lesion_detection_recall':None,'lesions':0,'detected_lesions':0,'predicted_components':m}
    intersections=np.bincount((gt.ravel()*(m+1)+pr.ravel()),minlength=(n+1)*(m+1)).reshape(n+1,m+1)[1:,1:]
    area_gt=np.bincount(gt.ravel(),minlength=n+1)[1:,None]
    area_pr=np.bincount(pr.ravel(),minlength=m+1)[None,1:]
    overlaps=intersections/np.maximum(area_gt+area_pr-intersections,1)
    candidates=[(float(overlaps[i,j]),i,j) for i,j in zip(*np.where(overlaps>=iou_threshold))]
    used_gt=set();used_pred=set()
    for _,i,j in sorted(candidates,reverse=True):
        if i not in used_gt and j not in used_pred:used_gt.add(i);used_pred.add(j)
    return {'lesion_detection_recall':len(used_gt)/n,'lesions':n,'detected_lesions':len(used_gt),'predicted_components':m}


def overlap(prob,truth,threshold=.5):
    truth=np.asarray(truth,bool);pred=np.asarray(prob)>=threshold
    if not truth.any():raise ValueError('Positive metrics require a nonempty target')
    tp=int((truth&pred).sum());fp=int((~truth&pred).sum());fn=int((truth&~pred).sum())
    return {'positive_dice':2*tp/(2*tp+fp+fn),'positive_iou':tp/(tp+fp+fn),
            'positive_recall':tp/(tp+fn),'positive_pixel_accuracy':float((pred==truth).mean()),
            'empty_prediction_rate':float(not pred.any())}


@torch.inference_mode()
def monitor(c,model,rows,protocol,batch_size,loader=None):
    model.eval();parts=[];positive=[];small=[];normal=[];accuracies=[];detected=lesions=0
    pos_sum=pos_weight=neg_sum=boundary_sum=0.;pos_count=neg_count=0
    loader=loader or make_loader(CachedDataset(c,rows),c,batch_size=batch_size,seed=c['seed'])
    lookup={row['id']:row for row in rows}
    for batch in loader:
        STOP.check();batch=move_batch(batch,c)
        with autocast(c):outputs=model(batch['features'],batch['rgb']);_,terms=objective(outputs,batch['mask'],batch['positive'],c,protocol['positive_area_median'])
        parts.append((len(batch['id']),{key:float(value) for key,value in terms.items()}))
        logits=outputs['final'].float();target=batch['mask'].float();pos=batch['positive'].bool();dims=(1,2,3)
        bce=F.binary_cross_entropy_with_logits(logits,target,reduction='none').mean(dims)
        if pos.any():
            p=logits[pos].sigmoid();y=target[pos]
            soft=1-(2*(p*y).sum(dims)+1e-6)/(p.sum(dims)+y.sum(dims)+1e-6)
            weight=(protocol['positive_area_median']/y.mean(dims).clamp_min(1/(target.shape[-1]*target.shape[-2]))).clamp(1,c['small_weight_cap'])
            pos_sum+=float(((bce[pos]+soft)*weight).sum());pos_weight+=float(weight.sum())
            n=int(pos.sum());pos_count+=n;boundary_sum+=n*float(terms['boundary_loss'])
        if (~pos).any():neg_sum+=float(bce[~pos].sum());neg_count+=int((~pos).sum())
        probabilities=F.interpolate(outputs['final'].float().sigmoid(),size=(c['selection_size'],)*2,mode='bilinear',align_corners=False).cpu().numpy()[:,0]
        targets=batch['selection_mask'].cpu().numpy()[:,0]>0
        for key,prob,truth in zip(batch['id'],probabilities,targets):
            row=lookup[key];pred=prob>=c['threshold'];accuracies.append(float((pred==truth).mean()))
            if row['positive']:
                metrics=overlap(prob,truth,c['threshold']);positive.append(metrics)
                if row['foreground_fraction']<=protocol['small_polyp_fraction_max']:small.append(metrics)
                lesion=detection(prob,truth,c['threshold'],c['lesion_detection_iou']);detected+=lesion['detected_lesions'];lesions+=lesion['lesions']
            else:
                if truth.any():raise ValueError('Healthy image has nonempty target')
                normal.append(float(pred.any()))
    total=sum(n for n,_ in parts)
    result={key:sum(n*row[key] for n,row in parts)/total for key in parts[0][1]}
    # Whole-partition category balance matches V1's predictive monitor rather
    # than allowing mostly-normal validation batches to dominate the loss plot.
    result['positive_loss']=pos_sum/pos_weight if pos_weight else 0.
    result['negative_loss']=neg_sum/neg_count if neg_count else 0.
    result['segmentation_loss']=(result['positive_loss']+result['negative_loss'])/max(int(pos_count>0)+int(neg_count>0),1)
    result['boundary_loss']=boundary_sum/pos_count if pos_count else 0.
    result['total_loss']=result['segmentation_loss']+c['contrast_weight']*result['contrastive_loss']+c['auxiliary_weight']*result['auxiliary_loss']+c['boundary_weight']*result['boundary_loss']
    for key in ['positive_dice','positive_iou','positive_recall','positive_pixel_accuracy','empty_prediction_rate']:
        result[key]=float(np.mean([row[key] for row in positive])) if positive else None
    result.update(small_polyp_dice=float(np.mean([r['positive_dice'] for r in small])) if small else None,
                  small_polyp_recall=float(np.mean([r['positive_recall'] for r in small])) if small else None,
                  normal_fpr=float(np.mean(normal)) if normal else None,macro_pixel_accuracy=float(np.mean(accuracies)),
                  lesion_detection_recall=detected/lesions if lesions else None)
    return result

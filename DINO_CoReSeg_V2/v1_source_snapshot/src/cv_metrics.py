import numpy as np
import torch
from torch.nn import functional as F
from .common import STOP
from .extract_features import CachedDataset
from .runtime import make_loader,move_batch


def bounded_loss(loss):
    if not np.isfinite(loss) or loss<0:raise ValueError('Invalid loss')
    return float(loss/(1+loss))


class PixelROC:
    def __init__(self,bins=65536):
        self.bins=bins;self.positive=np.zeros(bins,np.int64);self.negative=np.zeros(bins,np.int64)
    def update(self,prob,truth):
        prob=np.asarray(prob,float);truth=np.asarray(truth,bool)
        if prob.shape!=truth.shape or not np.isfinite(prob).all() or prob.min()<0 or prob.max()>1:raise ValueError('Invalid ROC arrays')
        bins=np.minimum((prob*self.bins).astype(np.int64),self.bins-1)
        self.positive+=np.bincount(bins[truth],minlength=self.bins)
        self.negative+=np.bincount(bins[~truth],minlength=self.bins)
    def curve(self):
        if not self.positive.sum() or not self.negative.sum():raise ValueError('ROC requires both pixel classes')
        tpr=np.r_[0,np.cumsum(self.positive[::-1])/self.positive.sum()]
        fpr=np.r_[0,np.cumsum(self.negative[::-1])/self.negative.sum()]
        return fpr,tpr,float(np.trapezoid(tpr,fpr))
    def save(self,path):
        fpr,tpr,auc=self.curve()
        np.savez_compressed(path,positive_hist=self.positive,negative_hist=self.negative,fpr=fpr,tpr=tpr,auc=auc,bins=self.bins)


class Monitor:
    def __init__(self,c,area_median):self.c=c;self.area_median=area_median;self.parts=[]
    def update(self,logits,target,positive):
        # Vectorized GPU reductions, one compact CPU transfer per batch.
        logits=logits.float();target=target.float()
        if not torch.isfinite(logits).all():raise FloatingPointError('Nonfinite monitor logits')
        p=logits.sigmoid();pred=p>=self.c['threshold'];truth=target.bool();dims=(1,2,3)
        bce=F.binary_cross_entropy_with_logits(logits,target,reduction='none').mean(dims)
        soft=1-(2*(p*target).sum(dims)+1e-6)/(p.sum(dims)+target.sum(dims)+1e-6)
        area=target.mean(dims);weight=(self.area_median/area.clamp_min(1/122500)).clamp(1,self.c['small_weight_cap'])
        tp=(pred&truth).sum(dims).float();fp=(pred&~truth).sum(dims).float();fn=(~pred&truth).sum(dims).float()
        accuracy=(pred==truth).float().mean(dims)
        dice=2*tp/(2*tp+fp+fn).clamp_min(1);iou=tp/(tp+fp+fn).clamp_min(1)
        vals=torch.stack([positive.float(),bce,soft,weight,accuracy,dice,iou,pred.flatten(1).any(1).float()],1).detach().cpu().numpy()
        self.parts.append(vals)
    def summary(self):
        a=np.concatenate(self.parts);pos=a[:,0]>0;neg=~pos
        if not pos.any() or not neg.any():raise ValueError('Monitor requires positive and normal samples')
        pl=float(np.sum((a[pos,1]+a[pos,2])*a[pos,3])/a[pos,3].sum());nl=float(a[neg,1].mean());raw=(pl+nl)/2
        return {'raw_segmentation_loss':raw,'loss_0_1':bounded_loss(raw),'accuracy':float(a[:,4].mean()),
                'positive_accuracy':float(a[pos,4].mean()),'normal_accuracy':float(a[neg,4].mean()),
                'positive_dice':float(a[pos,5].mean()),'positive_iou':float(a[pos,6].mean()),
                'normal_fpr':float(a[neg,7].mean()),'positive_loss':pl,'normal_loss':nl}


@torch.inference_mode()
def monitor_partition(model,rows,c,area_median,batch_size,roc=None,loader=None):
    model.eval();monitor=Monitor(c,area_median);scores=[]
    loader=loader or make_loader(CachedDataset(rows),c,batch_size=batch_size,seed=c['seed'])
    for batch in loader:
        STOP.check();batch=move_batch(batch,c)
        logits=model(batch['features'],batch['rgb'])['final'].float()
        monitor.update(logits,batch['mask'],batch['positive'])
        if roc is not None:
            prob=logits.sigmoid().cpu().numpy();truth=batch['mask'].cpu().numpy()
            for i,key in enumerate(batch['id']):
                roc.update(prob[i,0],truth[i,0]);scores.append({'id':key,'maximum_probability':float(prob[i].max())})
    return monitor.summary(),scores

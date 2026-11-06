import torch
from torch.nn import functional as F
from .prototypes import occupancy


def segmentation(logits,target,positive,c,area_median):
    bce=F.binary_cross_entropy_with_logits(logits,target,reduction='none').mean((1,2,3))
    pos=positive.bool();zero=logits.sum()*0
    if pos.any():
        prob=logits[pos].sigmoid();truth=target[pos]
        dice=1-(2*(prob*truth).sum((1,2,3))+1e-6)/(prob.sum((1,2,3))+truth.sum((1,2,3))+1e-6)
        fraction=truth.mean((1,2,3)).clamp_min(1/122500)
        weight=(area_median/fraction).clamp(min=1,max=c['small_weight_cap'])
        positive_loss=((bce[pos]+dice)*weight).sum()/weight.sum()
    else:positive_loss=zero
    negative_loss=bce[~pos].mean() if (~pos).any() else zero
    # Fixed equal category weighting when both are present; no frame-count scaling.
    present=int(pos.any())+int((~pos).any())
    total=(positive_loss+negative_loss)/max(present,1)
    return total,positive_loss,negative_loss


def contrast(projected,occ,c):
    flat=projected.permute(0,2,3,1).reshape(-1,64)
    labels=occ.reshape(-1)
    indices=[];classes=[]
    for label,mask in [(0,labels<=c['background_occupancy']),(1,labels>=c['foreground_occupancy'])]:
        ii=mask.nonzero().flatten()
        if len(ii):
            # Deterministic evenly spaced sampling; bounded quadratic contrast cost.
            ii=ii[torch.linspace(0,len(ii)-1,min(64,len(ii))).long()]
            indices.append(ii);classes.append(torch.full((len(ii),),label,device=ii.device))
    if len(indices)<2:return projected.sum()*0
    z=F.normalize(flat[torch.cat(indices)],dim=1);y=torch.cat(classes)
    logits=z@z.T/.1;diag=torch.eye(len(z),dtype=torch.bool,device=z.device)
    logits=logits.masked_fill(diag,-1e4)
    same=(y[:,None]==y[None,:])&~diag
    logprob=logits-torch.logsumexp(logits,dim=1,keepdim=True)
    valid=same.sum(1)>0
    return -(logprob*same).sum(1)[valid].div(same.sum(1)[valid]).mean()


def objective(outputs,target,positive,c,area_median):
    seg,pos,neg=segmentation(outputs['final'],target,positive,c,area_median)
    occ=occupancy(target)
    supcon=contrast(outputs['projected'],occ,c)
    confident=(occ<=c['background_occupancy'])|(occ>=c['foreground_occupancy'])
    labels=(occ>=c['foreground_occupancy']).float()
    ce=F.binary_cross_entropy_with_logits(outputs['counter_logits'],labels,reduction='none')
    # Equal fg/bg confident-patch contributions when available.
    counter_terms=[ce[confident&(labels==v)].mean() for v in [0,1] if (confident&(labels==v)).any()]
    counter=sum(counter_terms)/len(counter_terms) if counter_terms else seg*0
    aux=[]
    for name in ['initial','fused','refined']:
        logits=F.interpolate(outputs[name],size=(350,350),mode='bilinear',align_corners=False)
        aux.append(segmentation(logits,target,positive,c,area_median)[0])
    auxiliary=(sum(aux)/len(aux)+counter)/2
    total=seg+c['contrast_weight']*supcon+c['auxiliary_weight']*auxiliary
    parts={'loss':total,'positive_seg':pos,'normal_bce':neg,'contrast':supcon,'auxiliary':auxiliary}
    if not all(torch.isfinite(v) for v in parts.values()):raise FloatingPointError('Nonfinite loss')
    return total,parts

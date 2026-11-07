"""Resolution-derived V1 objective plus an independently controlled edge loss."""
import torch
from torch.nn import functional as F


def occupancy(mask,patch_size,grid_size):
    if mask.shape[-2:]!=(patch_size*grid_size,)*2:raise ValueError('Mask and DINO token grid do not align')
    return F.avg_pool2d(mask,patch_size,patch_size)


def background_eligible(mask,c):
    halo=F.max_pool2d(mask,5,1,2)
    return (occupancy(mask,c['patch_size'],c['grid_size'])<=c['background_occupancy']) & (occupancy(halo,c['patch_size'],c['grid_size'])<=c['background_occupancy'])


def segmentation(logits,target,positive,c,median):
    bce=F.binary_cross_entropy_with_logits(logits.float(),target.float(),reduction='none').mean((1,2,3))
    pos=positive.bool();zero=logits.sum()*0
    if pos.any():
        p=logits[pos].float().sigmoid();y=target[pos].float();dims=(1,2,3)
        soft=1-(2*(p*y).sum(dims)+1e-6)/(p.sum(dims)+y.sum(dims)+1e-6)
        weight=(median/y.mean(dims).clamp_min(1/(target.shape[-1]*target.shape[-2]))).clamp(1,c['small_weight_cap'])
        pl=((bce[pos]+soft)*weight).sum()/weight.sum()
    else:pl=zero
    nl=bce[~pos].mean() if (~pos).any() else zero
    return (pl+nl)/max(int(pos.any())+int((~pos).any()),1),pl,nl


def contrast(projected,occ,c):
    flat=projected.permute(0,2,3,1).reshape(-1,projected.shape[1]);labels=occ.reshape(-1)
    indices=[];classes=[]
    for label,mask in [(0,labels<=c['background_occupancy']),(1,labels>=c['foreground_occupancy'])]:
        ii=mask.nonzero().flatten()
        if len(ii):
            ii=ii[torch.linspace(0,len(ii)-1,min(64,len(ii)),device=ii.device).long()]
            indices.append(ii);classes.append(torch.full((len(ii),),label,device=ii.device))
    if len(indices)<2:return projected.sum()*0
    z=F.normalize(flat[torch.cat(indices)].float(),dim=1);y=torch.cat(classes)
    logits=z@z.T/.1;diag=torch.eye(len(z),dtype=torch.bool,device=z.device)
    logits=logits.masked_fill(diag,-1e4);same=(y[:,None]==y[None,:])&~diag;valid=same.sum(1)>0
    if not valid.any():return projected.sum()*0
    logprob=logits-torch.logsumexp(logits,1,keepdim=True)
    return -(logprob*same).sum(1)[valid].div(same.sum(1)[valid]).mean()


def boundary_weights(target,radius=4,emphasis=2.):
    """w=1+beta*max(1-d_inf(edge)/(R+1),0), via square dilations."""
    dilation=F.max_pool2d(target,3,1,1)
    erosion=-F.max_pool2d(-F.pad(target,(1,1,1,1),value=0),3,1,0)
    edge=(dilation-erosion)>0
    bands=sum(F.max_pool2d(edge.float(),2*r+1,1,r) for r in range(radius+1))
    return 1+emphasis*bands/(radius+1)


def boundary_loss(logits,target,positive,c):
    pos=positive.bool()
    if not pos.any():return logits.sum()*0
    weight=boundary_weights(target[pos].float(),c['boundary_radius'],c['boundary_emphasis'])
    bce=F.binary_cross_entropy_with_logits(logits[pos].float(),target[pos].float(),reduction='none')
    return ((weight*bce).sum((1,2,3))/weight.sum((1,2,3))).mean()


def objective(outputs,target,positive,c,median):
    seg,pos,neg=segmentation(outputs['final'],target,positive,c,median)
    occ=occupancy(target,c['patch_size'],c['grid_size'])
    supcon=contrast(outputs['projected'],occ,c)
    confident=(occ<=c['background_occupancy'])|(occ>=c['foreground_occupancy'])
    labels=(occ>=c['foreground_occupancy']).float()
    ce=F.binary_cross_entropy_with_logits(outputs['counter_logits'].float(),labels,reduction='none')
    terms=[ce[confident&(labels==v)].mean() for v in [0,1] if (confident&(labels==v)).any()]
    counter=sum(terms)/len(terms) if terms else seg*0
    aux=[segmentation(F.interpolate(outputs[name].float(),size=target.shape[-2:],mode='bilinear',align_corners=False),target,positive,c,median)[0]
         for name in ['initial','fused','refined']]
    auxiliary=(sum(aux)/len(aux)+counter)/2
    boundary=boundary_loss(outputs['final'],target,positive,c)
    total=seg+c['contrast_weight']*supcon+c['auxiliary_weight']*auxiliary+c['boundary_weight']*boundary
    parts={'total_loss':total,'segmentation_loss':seg,'positive_loss':pos,'negative_loss':neg,
           'contrastive_loss':supcon,'auxiliary_loss':auxiliary,'boundary_loss':boundary}
    if not all(torch.isfinite(value) for value in parts.values()):raise FloatingPointError('Non-finite V2 objective')
    return total,parts

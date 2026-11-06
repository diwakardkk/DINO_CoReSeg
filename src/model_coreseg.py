import torch
from torch import nn
from torch.nn import functional as F


def block(cin,cout,k=3):
    return nn.Sequential(nn.Conv2d(cin,cout,k,padding=k//2),nn.GroupNorm(8,cout),nn.GELU())


class DINOCoReSeg(nn.Module):
    def __init__(self,prototypes):
        super().__init__()
        if prototypes.shape!=(8,384):raise ValueError('Expected 8 normal prototypes in 384D DINO space')
        self.register_buffer('prototypes',F.normalize(prototypes.float(),dim=1))
        self.projection=block(384,64,1)
        self.initial_head=nn.Sequential(block(64,32),nn.Conv2d(32,1,1))
        self.cef=nn.Sequential(block(66,32),block(32,16),nn.Conv2d(16,1,1))
        self.gamma_raw=nn.Parameter(torch.tensor(.01))
        self.rgb_branch=nn.Sequential(block(3,16),block(16,16))
        self.light_features=block(64,16,1)
        self.rpr=nn.Sequential(block(33,32),block(32,16),nn.Conv2d(16,1,1))
        self.rpr_scale=nn.Parameter(torch.tensor(.01))
        self.counter_scale=nn.Parameter(torch.tensor(1.0));self.counter_bias=nn.Parameter(torch.tensor(-.5))

    def forward(self,features,rgb):
        if features.ndim!=4 or tuple(features.shape[1:])!=(384,25,25):raise ValueError('Expected features [B,384,25,25]')
        if tuple(rgb.shape)!=(len(features),3,100,100):raise ValueError('Expected RGB [B,3,100,100]')
        p=self.projection(features)
        similarity=torch.einsum('bchw,kc->bkhw',F.normalize(features,dim=1),self.prototypes).amax(1,keepdim=True)
        counter=(1-similarity).clamp(0,2)/2
        initial=self.initial_head(p)
        residual=self.cef(torch.cat([p,counter,initial],dim=1))
        fused=initial+torch.tanh(self.gamma_raw)*residual
        up=F.interpolate(fused,size=(100,100),mode='bilinear',align_corners=False)
        light=F.interpolate(self.light_features(p),size=(100,100),mode='bilinear',align_corners=False)
        refinement=self.rpr(torch.cat([up,light,self.rgb_branch(rgb)],dim=1))
        refined=up+torch.tanh(self.rpr_scale)*refinement
        final=F.interpolate(refined,size=(350,350),mode='bilinear',align_corners=False)
        return {'final':final,'initial':initial,'counter_evidence':counter,'fused':fused,
                'refined':refined,'projected':p,
                'counter_logits':F.softplus(self.counter_scale)*counter+self.counter_bias}

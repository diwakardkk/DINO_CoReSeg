"""Versioned resolution, RGB-context and multiscale DINO-CoReSeg decoders."""
import torch
from torch import nn
from torch.nn import functional as F


def block(cin,cout,k=3,stride=1):
    return nn.Sequential(nn.Conv2d(cin,cout,k,stride=stride,padding=k//2),nn.GroupNorm(8,cout),nn.GELU())


class DINOCoReSegV2(nn.Module):
    def __init__(self,c,prototypes):
        super().__init__();self.c=c
        channels=c['projection_dim'];feature=c['feature_dim']
        if prototypes.shape!=(c['prototype_count'],feature):raise ValueError('Prototype shape differs from V2 configuration')
        self.register_buffer('prototypes',F.normalize(prototypes.float(),dim=1))
        self.projection=block(feature,channels,1)
        self.initial_head=nn.Sequential(block(channels,32),nn.Conv2d(32,1,1))
        self.cef=nn.Sequential(block(channels+2,32),block(32,16),nn.Conv2d(16,1,1))
        self.gamma_raw=nn.Parameter(torch.tensor(.01))
        self.rgb_branch=nn.Sequential(block(3,16),block(16,16))
        self.light_features=block(channels,16,1)
        self.rpr=nn.Sequential(block(33,32),block(32,16),nn.Conv2d(16,1,1))
        self.rpr_scale=nn.Parameter(torch.tensor(.01))
        self.counter_scale=nn.Parameter(torch.tensor(1.0));self.counter_bias=nn.Parameter(torch.tensor(-.5))
        if c['variant']!='resolution':
            self.rgb_context=nn.Sequential(block(16,24,stride=2),block(24,16))
            self.rgb_context_scale=nn.Parameter(torch.tensor(.01))
        if c['variant'] in ['boundary_refinement','multilayer']:
            self.coarse_rpr=nn.Sequential(block(33,32),block(32,16),nn.Conv2d(16,1,1))
            self.coarse_scale=nn.Parameter(torch.tensor(.01))
        if c['variant']=='multilayer':
            self.layer_projections=nn.ModuleList([block(feature,channels,1) for _ in c['intermediate_layers'][:-1]])
            self.layer_weights=nn.Parameter(torch.zeros(len(c['intermediate_layers'])))

    def forward(self,features,rgb):
        grid=self.c['grid_size'];size=self.c['rgb_size']
        expected=(self.c['feature_dim'],grid,grid)
        layers=features if features.ndim==5 else features[:,None]
        if tuple(layers.shape[2:])!=expected:raise ValueError(f'Expected spatial features {expected}')
        expected_layers=len(self.c['intermediate_layers']) if self.c['variant']=='multilayer' else 1
        if layers.shape[1]!=expected_layers or tuple(rgb.shape)!=(len(layers),3,size,size):raise ValueError('V2 feature/RGB dimensions differ')
        final_features=layers[:,-1];p=self.projection(final_features)
        if self.c['variant']=='multilayer':
            projected=[module(layers[:,i]) for i,module in enumerate(self.layer_projections)]+[p]
            p=sum(weight*value for weight,value in zip(self.layer_weights.softmax(0),projected))
        similarity=torch.einsum('bchw,kc->bkhw',F.normalize(final_features,dim=1),self.prototypes).amax(1,keepdim=True)
        counter=(1-similarity).clamp(0,2)/2
        initial=self.initial_head(p);fused=initial+torch.tanh(self.gamma_raw)*self.cef(torch.cat([p,counter,initial],1))
        up=F.interpolate(fused,size=rgb.shape[-2:],mode='bilinear',align_corners=False)
        light=F.interpolate(self.light_features(p),size=rgb.shape[-2:],mode='bilinear',align_corners=False)
        rgb_features=self.rgb_branch(rgb);coarse=None
        if self.c['variant']!='resolution':
            context=self.rgb_context(rgb_features)
            rgb_features=rgb_features+torch.tanh(self.rgb_context_scale)*F.interpolate(context,size=rgb.shape[-2:],mode='bilinear',align_corners=False)
            if self.c['variant'] in ['boundary_refinement','multilayer']:
                resize=lambda x:F.interpolate(x,size=context.shape[-2:],mode='bilinear',align_corners=False)
                coarse=resize(up)+torch.tanh(self.coarse_scale)*self.coarse_rpr(torch.cat([resize(up),resize(light),context],1))
                up=up+F.interpolate(coarse-resize(up),size=up.shape[-2:],mode='bilinear',align_corners=False)
        refined=up+torch.tanh(self.rpr_scale)*self.rpr(torch.cat([up,light,rgb_features],1))
        final=F.interpolate(refined,size=(self.c['image_size'],)*2,mode='bilinear',align_corners=False)
        return {'final':final,'initial':initial,'counter_evidence':counter,'fused':fused,'refined':refined,
                'projected':p,'counter_logits':F.softplus(self.counter_scale)*counter+self.counter_bias}

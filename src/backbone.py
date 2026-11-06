import os
from pathlib import Path
import torch
from torch import nn
from .common import RESEARCH,file_hash
from .runtime import parallelize


class SpatialDino(nn.Module):
    def __init__(self,backbone):super().__init__();self.backbone=backbone
    def forward(self,x):
        if tuple(x.shape[1:])!=(3,350,350):raise ValueError('DINO expects [B,3,350,350]')
        tokens=self.backbone.forward_features(x)['x_norm_patchtokens']
        if tuple(tokens.shape[1:])!=(625,384):raise ValueError('Incorrect DINO patch/register token dimensions')
        return tokens.transpose(1,2).reshape(len(x),384,25,25).contiguous()


def load_backbone(c):
    os.environ['XFORMERS_DISABLED']='1'
    repo=RESEARCH/c['backbone']['local_repo'];weights=RESEARCH/c['backbone']['checkpoint']
    if not (repo/'hubconf.py').exists() or not weights.exists():raise FileNotFoundError('Bundled official DINO source/weights missing')
    sha=file_hash(weights)
    if sha!=c['backbone']['weight_sha256']:raise RuntimeError('DINO checkpoint SHA-256 mismatch')
    model=torch.hub.load(str(repo),c['backbone']['name'],source='local',pretrained=False,verbose=False)
    model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True)
    if model.embed_dim!=384 or model.patch_size!=14 or model.num_register_tokens!=0:raise ValueError('Expected non-register DINOv2-S/14')
    model.eval().requires_grad_(False)
    identity={'name':c['backbone']['name'],'weight_sha256':sha,
              'source_commit':(repo/'SOURCE_COMMIT.txt').read_text().strip(),
              'parameters':sum(p.numel() for p in model.parameters())}
    return parallelize(SpatialDino(model),c).eval(),identity

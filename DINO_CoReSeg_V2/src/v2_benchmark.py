"""Separately measured cached-decoder and live full-pipeline latency."""
import time
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from .common import read_json,atomic_json,pair_arrays
from .runtime import device,parallelize,synchronize,hardware
from .v2_config import output
from .v2_features import load_backbone
from .v2_model import DINOCoReSegV2


@torch.inference_mode()
def benchmark(c,rows):
    directory=output(c);state=torch.load(directory/'checkpoints/best.pt',map_location='cpu',weights_only=False)
    raw=DINOCoReSegV2(c,state['model']['prototypes']);raw.load_state_dict(state['model']);decoder=parallelize(raw,c).eval()
    backbone,identity=load_backbone(c);times=[]
    mean=torch.tensor(c['backbone']['mean'],device=device(c))[None,:,None,None];std=torch.tensor(c['backbone']['std'],device=device(c))[None,:,None,None]
    for row in rows[:8]:
        synchronize(c);tick=time.perf_counter();image,_=pair_arrays(c,row,include_mask=False)
        x=torch.from_numpy(np.array(image.resize((c['image_size'],)*2,Image.Resampling.BICUBIC),np.float32).transpose(2,0,1).copy()/255.)[None].to(device(c))
        rgb=torch.from_numpy(np.array(image.resize((c['rgb_size'],)*2,Image.Resampling.BILINEAR),np.float32).transpose(2,0,1).copy()/255.)[None].to(device(c))
        synchronize(c);a=time.perf_counter();features=backbone((x-mean)/std).half().float();synchronize(c);b=time.perf_counter()
        logits=decoder(features,rgb)['final'].float();synchronize(c);d=time.perf_counter()
        prediction=F.interpolate(logits.sigmoid(),size=(image.height,image.width),mode='bilinear',align_corners=False)>=c['threshold'];prediction.cpu();synchronize(c);e=time.perf_counter()
        times.append({'id':row['id'],'backbone_seconds':b-a,'decoder_seconds':d-b,'end_to_end_seconds':e-tick})
    atomic_json(directory/'evaluation/metrics/end_to_end_runtime.json',{'hardware':hardware(c),'samples':times,
        'means':{key:float(np.mean([r[key] for r in times])) for key in ['backbone_seconds','decoder_seconds','end_to_end_seconds']},
        'scope':'Batch one: image loading, resizing/normalization, transfers, live DINO, decoder, native restore and host prediction transfer. Model loading excluded; first sample included.',
        'backbone_parameters':identity['parameters']})

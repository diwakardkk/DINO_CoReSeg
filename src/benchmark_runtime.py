import time
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from .common import read_json,atomic_json,pair_arrays
from .backbone import load_backbone
from .model_coreseg import DINOCoReSeg
from .runtime import device,parallelize,synchronize,hardware


def run(c,rows,directory,count=8):
    state=torch.load(directory/'best.pt',map_location='cpu',weights_only=False)
    raw=DINOCoReSeg(state['model']['prototypes']);raw.load_state_dict(state['model']);decoder=parallelize(raw,c).eval()
    backbone,identity=load_backbone(c);times=[]
    mean=torch.tensor(c['backbone']['mean'],device=device(c)).view(1,3,1,1)
    std=torch.tensor(c['backbone']['std'],device=device(c)).view(1,3,1,1)
    with torch.inference_mode():
        for row in rows[:count]:
            synchronize(c);start=time.perf_counter();image,_=pair_arrays(c,row,include_mask=False)
            x=torch.from_numpy(np.array(image.resize((350,350),Image.Resampling.BICUBIC),np.float32).transpose(2,0,1).copy()/255.)[None].to(device(c))
            rgb=torch.from_numpy(np.array(image.resize((100,100),Image.Resampling.BILINEAR),np.float32).transpose(2,0,1).copy()/255.)[None].to(device(c))
            synchronize(c);a=time.perf_counter();feature=backbone((x-mean)/std).to(torch.float16).float();synchronize(c);b=time.perf_counter()
            logits=decoder(feature,rgb)['final'].float();synchronize(c);d=time.perf_counter()
            prediction=F.interpolate(logits.sigmoid(),size=(image.height,image.width),mode='bilinear',align_corners=False)>=c['threshold']
            prediction.cpu();synchronize(c);e=time.perf_counter()
            times.append({'id':row['id'],'backbone_seconds':b-a,'decoder_seconds':d-b,'end_to_end_seconds':e-start})
    atomic_json(directory/'metrics/end_to_end_runtime.json',{'hardware':hardware(c),'samples':times,
                'means':{k:float(np.mean([r[k] for r in times])) for k in ['backbone_seconds','decoder_seconds','end_to_end_seconds']},
                'scope':'Batch-one image load/preprocess, host/device transfers, backbone, decoder, native prediction restoration and transfer to host; model loading excluded.',
                'first_sample_included':True,'backbone_parameters':identity['parameters']})

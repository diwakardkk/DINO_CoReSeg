from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.cluster import MiniBatchKMeans
from .common import RESEARCH,atomic_json,atomic_torch,fingerprint,read_json,STOP
from .extract_features import CachedDataset


def occupancy(mask):return F.avg_pool2d(mask,14,14)
def background_eligible(mask,maximum=.01):
    return (occupancy(mask)<=maximum)&(occupancy(F.max_pool2d(mask,5,1,2))<=maximum)


def build(c,rows,allowed_training_ids,destination,seed):
    allowed=set(allowed_training_ids)
    if not rows or {r['id'] for r in rows}!=allowed:raise ValueError('Prototype membership must match the declared training partition')
    destination=Path(destination)
    identity={'training_ids':fingerprint(sorted(allowed)),'backbone':read_json(RESEARCH/'cache/backbone_identity.json'),
              'seed':seed,'count':8,'dimension':384,'occupancy':c['background_occupancy'],
              'per_image':c['prototype_patches_per_image'],'category_cap':c['prototype_max_per_category']}
    if destination.exists():
        bank=torch.load(destination,map_location='cpu',weights_only=True)
        if bank['identity']!=identity:raise RuntimeError('Existing prototype bank identity mismatch')
        if not set(sum(bank['sample_ids'].values(),[]))<=allowed:raise ValueError('Prototype training membership mismatch')
        centers=bank['prototypes']
        if centers.shape!=(8,384) or not torch.isfinite(centers).all() or not torch.allclose(centers.norm(dim=1),torch.ones(8),atol=1e-5):
            raise ValueError('Invalid stored prototypes')
        return centers
    rng=np.random.default_rng(seed);categories=defaultdict(list)
    for row in rows:categories['positive_background' if row['positive'] else 'FAR_healthy'].append(row)
    if len(categories)!=2:raise ValueError('Both prototype categories are required')
    count=min(c['prototype_max_per_category'],min(map(len,categories.values())));vectors={};used={}
    for category,items in categories.items():
        chunks=[];used[category]=[]
        for index in rng.choice(len(items),count,replace=False):
            STOP.check();row=items[int(index)];item=CachedDataset([row])[0]
            candidates=np.flatnonzero(background_eligible(item['mask'][None],c['background_occupancy']).flatten().numpy())
            if not len(candidates):continue
            take=rng.choice(candidates,min(len(candidates),c['prototype_patches_per_image']),replace=False)
            patches=item['features'].permute(1,2,0).reshape(-1,384)[take].numpy().copy()
            patches/=np.maximum(np.linalg.norm(patches,axis=1,keepdims=True),1e-8)
            chunks.append(patches);used[category].append(row['id'])
        if not chunks:raise ValueError('No confident background available for prototypes')
        vectors[category]=np.concatenate(chunks)
    n=min(map(len,vectors.values()));balanced=np.concatenate([v[rng.choice(len(v),n,replace=False)] for v in vectors.values()])
    km=MiniBatchKMeans(n_clusters=8,random_state=seed,n_init=3,batch_size=512,max_iter=100).fit(balanced)
    centers=F.normalize(torch.from_numpy(km.cluster_centers_.astype(np.float32)),dim=1)
    atomic_torch(destination,{'prototypes':centers,'identity':identity,'sample_ids':used,'patches_per_category':n})
    atomic_json(destination.with_suffix('.json'),{'identity':identity,'sample_ids':used,'patches_per_category':n},immutable=True)
    return centers

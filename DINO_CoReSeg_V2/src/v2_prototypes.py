"""Training-only prototype fitting with exact token-to-mask spatial alignment."""
from collections import defaultdict
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.cluster import MiniBatchKMeans
from .common import atomic_json,atomic_torch,fingerprint,read_json,STOP
from .v2_config import cache_root
from .v2_features import CachedDataset
from .v2_losses import background_eligible


def build(c,rows,allowed_ids,path):
    allowed=set(allowed_ids)
    if not rows or {r['id'] for r in rows}!=allowed:raise ValueError('Prototype membership must equal training IDs')
    identity={'train_ids':fingerprint(sorted(allowed)),'cache':read_json(cache_root(c)/'identity.json'),
              'seed':c['seed'],'count':c['prototype_count'],'patch_size':c['patch_size'],'grid':c['grid_size'],
              'background_max':c['background_occupancy'],'halo_pixels':2,'per_image':c['prototype_patches_per_image'],
              'category_cap':c['prototype_max_per_category']}
    if path.exists():
        bank=torch.load(path,map_location='cpu',weights_only=False)
        if bank['identity']!=identity:raise ValueError('V1/different-resolution prototype reuse refused')
        if not set(sum(bank['sample_ids'].values(),[]))<=allowed:raise ValueError('Prototype membership leak')
        centers=bank['prototypes']
        if centers.shape!=(c['prototype_count'],c['feature_dim']) or not torch.isfinite(centers).all() or not torch.allclose(centers.norm(dim=1),torch.ones(c['prototype_count']),atol=1e-5):raise ValueError('Invalid normalized V2 prototype bank')
        return centers
    categories=defaultdict(list)
    for row in rows:categories['positive_background' if row['positive'] else 'healthy'].append(row)
    if len(categories)!=2:raise ValueError('Both prototype categories required')
    rng=np.random.default_rng(c['seed']);count=min(c['prototype_max_per_category'],min(map(len,categories.values())))
    vectors={};used={}
    for category,items in categories.items():
        chunks=[];used[category]=[]
        for index in rng.choice(len(items),count,replace=False):
            STOP.check();row=items[index];item=CachedDataset(c,[row])[0]
            eligible=background_eligible(item['mask'][None],c).flatten().numpy();indices=np.flatnonzero(eligible)
            if not len(indices):continue
            take=rng.choice(indices,min(len(indices),c['prototype_patches_per_image']),replace=False)
            patches=item['features'][-1].permute(1,2,0).reshape(-1,c['feature_dim'])[take].numpy().copy()
            patches/=np.maximum(np.linalg.norm(patches,axis=1,keepdims=True),1e-8);chunks.append(patches);used[category].append(row['id'])
        if not chunks:raise ValueError('No eligible training background')
        vectors[category]=np.concatenate(chunks)
    n=min(map(len,vectors.values()));balanced=np.concatenate([v[rng.choice(len(v),n,replace=False)] for v in vectors.values()])
    km=MiniBatchKMeans(n_clusters=c['prototype_count'],random_state=c['seed'],n_init=3,batch_size=512,max_iter=100).fit(balanced)
    centers=F.normalize(torch.from_numpy(km.cluster_centers_.astype(np.float32)),dim=1)
    atomic_torch(path,{'identity':identity,'prototypes':centers,'sample_ids':used})
    atomic_json(path.with_suffix('.json'),{'identity':identity,'sample_ids':used,'patches_per_category':n},immutable=True)
    return centers

"""Image-only duplicate search; final test mask contents are never inspected here."""
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
import csv
import hashlib
import json
import time

import numpy as np
from PIL import Image
from scipy.fft import dctn

from .common import (RESEARCH, args, config, atomic_json, file_hash, fingerprint,
                     pair_arrays, pair_by_stem, csv_write, UnionFind, STOP)

EXT = {'.png', '.bmp', '.jpg', '.jpeg'}


def phash(rgb):
    gray = np.asarray(rgb.convert('L').resize((32,32), Image.Resampling.LANCZOS), dtype=float)
    coeff = dctn(gray, norm='ortho')[:8,:8].ravel()
    bits = coeff > np.median(coeff[1:]); bits[0] = False
    return int.from_bytes(np.packbits(bits).tobytes(), 'big')


def near_pairs(hashes, radius=4):
    """Pigeonhole candidates in r+1 disjoint chunks; exact Hamming verification."""
    buckets = defaultdict(list); pairs = set()
    edges = np.linspace(0,64,radius+2,dtype=int)
    for i, h in enumerate(hashes):
        candidates = set()
        keys = [(j,(h>>int(edges[j])) & ((1<<int(edges[j+1]-edges[j]))-1)) for j in range(radius+1)]
        for key in keys: candidates.update(buckets[key])
        for k in candidates:
            if (h ^ hashes[k]).bit_count() <= radius: pairs.add((k,i))
        for key in keys: buckets[key].append(i)
    return sorted(pairs)


def inventory(c):
    root = Path(c['root']); rows = []
    # Discover pairs on the server; no sample manifest is published with code.
    benchmark=root/'dataset-3/PolypDataset'
    collections=[('train_base',benchmark/'NewTRimage',benchmark/'NewTRmask')]
    tests=benchmark/'TestDataset/TestDataset'
    for source in ['CVC-300','CVC-ClinicDB','CVC-ColonDB','ETIS-LaribPolypDB','Kvasir']:
        collections.append(('test_'+source,tests/source/'images',tests/source/'masks'))
    for group,images_dir,masks_dir in collections:
        images=sorted(p for p in images_dir.glob('*') if p.suffix.lower() in EXT)
        masks=sorted(p for p in masks_dir.glob('*') if p.suffix.lower() in EXT)
        for image,mask in pair_by_stem(images,masks):
            sid=image.stem
            source=group[5:] if group.startswith('test_') else 'unverified_numeric_source'
            rows.append(dict(id=f'd3/{group}/{sid}',image=str(image.relative_to(root)),
                             mask=str(mask.relative_to(root)),patient='',source=source,
                             domain='benchmark',collection=group,positive=True,
                             mask_kind='supplied',archive=None))
    patient_root = root/'dataset-2/Dataset'
    for patient in sorted(patient_root.glob('Patient*')):
        if not patient.is_dir(): continue
        images=[p for p in (patient/'Polyp').glob('*') if p.suffix.lower() in EXT]
        masks=[p for p in (patient/'BinaryMask').glob('*') if p.suffix.lower() in EXT]
        for im, mask in pair_by_stem(images,masks):
            rows.append(dict(id=f'far/{patient.name}/Polyp/{im.stem}',image=str(im.relative_to(root)),
                             mask=str(mask.relative_to(root)),patient=patient.name, source='FAR',domain='FAR',
                             collection='far',positive=True,mask_kind='supplied',archive=None))
        for im in sorted((patient/'Healthy').glob('*')):
            if im.suffix.lower() not in EXT: continue
            rows.append(dict(id=f'far/{patient.name}/Healthy/{im.stem}',image=str(im.relative_to(root)),
                             mask=None,patient=patient.name,source='FAR',domain='FAR',collection='far',
                             positive=False,mask_kind='derived_empty',archive=None))
    assert Counter(r['collection'] for r in rows)['train_base']==1450
    assert sum(r['domain']=='FAR' for r in rows)==8181
    assert sum(r['collection'].startswith('test_') for r in rows)==798
    assert len({r['id'] for r in rows})==len(rows)
    return rows


def run(c):
    root = Path(c['root']); rows = inventory(c)
    # Resolve provenance against the user's local Kvasir archive.
    from zipfile import ZipFile
    with ZipFile(root/'dataset-1/kvasir-seg.zip') as archive:
        known={Path(name).stem for name in archive.namelist()
               if name.startswith('Kvasir-SEG/images/') and Path(name).suffix.lower() in EXT}
    for r in rows:
        if r['collection']=='train_base':
            sid=Path(r['image']).stem
            r['source']='Kvasir' if sid in known else 'unverified_numeric_source'
    baseline = RESEARCH/'manifests/inventory.json'
    atomic_json(baseline, {'rows':rows}, immutable=True)
    hashed=[]; start=time.perf_counter()
    for i,r in enumerate(rows):
        STOP.check()
        # Development masks are decoded; benchmark masks only have their header dimensions checked.
        im,m = pair_arrays(c,r,include_mask=not r['collection'].startswith('test_'))
        if r['mask'] and r['collection'].startswith('test_'):
            with Image.open(root/r['mask']) as header:
                if header.size!=im.size: raise ValueError(f"Bad test pair dimensions {r['id']}")
        if m is not None and r['positive'] and not m.any(): raise ValueError(f"Empty positive mask {r['id']}")
        rgb=np.asarray(im)
        norm=np.asarray(im.resize((64,64),Image.Resampling.BILINEAR))
        r.update(width=im.width,height=im.height)
        if m is not None: r['foreground_fraction']=float(m.mean())
        hashes={'file_sha256':file_hash(root/r['image']),
                'decoded_sha256':hashlib.sha256(str(rgb.shape).encode()+rgb.tobytes()).hexdigest(),
                'normalized_sha256':hashlib.sha256(norm.tobytes()).hexdigest(),'phash':phash(im)}
        r['image_sha256']=hashes['file_sha256']
        if r['mask'] and not r['collection'].startswith('test_'): r['mask_sha256']=file_hash(root/r['mask'])
        hashed.append(dict(id=r['id'],**hashes))
        if (i+1)%1000==0: print(f'Audited {i+1}/{len(rows)}',flush=True)
    edges={}; exactgroups={}
    for kind in ['file_sha256','decoded_sha256','normalized_sha256']:
        groups=defaultdict(list)
        for i,h in enumerate(hashed): groups[h[kind]].append(i)
        groups=[g for g in groups.values() if len(g)>1];exactgroups[kind]=groups
        for g in groups:
            for a,b in combinations(g,2): edges.setdefault((a,b),[]).append(kind)
    for a,b in near_pairs([h['phash'] for h in hashed],c['phash_hamming_threshold']):
        edges.setdefault((a,b),[]).append('phash_candidate')
    links=[dict(a=rows[a]['id'],b=rows[b]['id'],reasons=k,
                hamming=(hashed[a]['phash']^hashed[b]['phash']).bit_count()) for (a,b),k in sorted(edges.items())]
    report={'rows':len(rows),'counts':dict(Counter(r['collection'] for r in rows)),
            'exact_group_counts':{k:len(v) for k,v in exactgroups.items()},
            'links':links,'phash_radius':c['phash_hamming_threshold'],
            'policy':'Conservative candidate grouping/exclusion before splits. pHash is a heuristic, not clinical identity proof.',
            'test_masks':'Headers only; no test-mask pixels or statistics used.',
            'audit_input_sha256':fingerprint({'inventory_schema':'server-discovery-v1','counts':dict(Counter(r['collection'] for r in rows))}),
            'limitations':['pHash cannot establish the absence of all near-duplicates.',
                          'Numeric development source remains unverified; not labeled ClinicDB without provenance.',
                          'Patient independence of benchmark datasets cannot be established.']}
    atomic_json(RESEARCH/'manifests/audit.json',report,immutable=True)
    atomic_json(RESEARCH/'manifests/audit_runtime.json',{'seconds':time.perf_counter()-start})
    atomic_json(RESEARCH/'manifests/hashes.json',hashed,immutable=True)
    atomic_json(RESEARCH/'manifests/all.json',{'rows':rows},immutable=True)
    csv_write(RESEARCH/'manifests/dataset_manifest.csv',rows)
    csv_write(RESEARCH/'manifests/duplicate_links.csv',links)
    print(f'Audit complete: {len(rows)} images; {len(links)} duplicate/near-duplicate links',flush=True)
    return report


if __name__=='__main__':
    a=args('Audit development/FAR and image-only test duplicate content');run(config(a.config,a.data_root))

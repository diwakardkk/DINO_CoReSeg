"""Verify actual frozen V1 membership and report physical collection use."""
from collections import Counter,defaultdict
from pathlib import Path
from zipfile import ZipFile
import hashlib
import numpy as np
from PIL import Image
from scipy.fft import dctn
from .common import read_json,atomic_json,csv_write,file_hash,pair_by_stem,STOP
from .v2_config import output

EXT={'.png','.jpg','.jpeg','.bmp'}


def verify_disjoint(splits,hashes=None,links=()):
    membership={};patients={};components={};identities={}
    for split,rows in splits.items():
        for row in rows:
            sid=row['id']
            if row['collection'].startswith('test_') and split!='test':raise ValueError('Official dataset-3 test sample entered development')
            if row['collection'] not in ['far','train_base'] and not row['collection'].startswith('test_'):raise ValueError('Augmented/unknown collection entered initial V2')
            if row['domain']=='FAR' and not row['patient']:raise ValueError('FAR patient ID is required')
            if not row['positive'] and (row['mask'] is not None or row['mask_kind']!='derived_empty'):raise ValueError('Healthy targets must be explicitly derived empty')
            if sid in membership:raise ValueError('Image ID leakage/duplication')
            membership[sid]=split
            if row['patient']:
                previous=patients.setdefault(row['patient'],split)
                if previous!=split:raise ValueError('FAR patient leakage')
            component=row['duplicate_component'];previous=components.setdefault(component,split)
            if previous!=split:raise ValueError('Duplicate-component leakage')
            for key in ['image_sha256']:
                if row.get(key):
                    previous=identities.setdefault((key,row[key]),split)
                    if previous!=split:raise ValueError('Image-identity leakage')
    for row in hashes or []:
        if row['id'] not in membership:continue
        for key in ['file_sha256','decoded_sha256','normalized_sha256']:
            if row.get(key):
                previous=identities.setdefault((key,row[key]),membership[row['id']])
                if previous!=membership[row['id']]:raise ValueError('Decoded/normalized image leakage')
    for edge in links:
        if edge['a'] in membership and edge['b'] in membership and membership[edge['a']]!=membership[edge['b']]:
            raise ValueError('Detected image-link leakage')
    return {'samples':{key:len(rows) for key,rows in splits.items()},'patient_disjoint':True,
            'detected_identity_disjoint':True,'limitation':'No detected leakage is not proof of all possible duplicate/patient identities in public benchmarks.'}


def used_counts(splits,predicate):return {key:sum(bool(predicate(row)) for row in rows) for key,rows in splits.items()}


def _files(path):return sorted(p for p in path.glob('*') if p.suffix.lower() in EXT)


def _pairs(images,masks):return len(pair_by_stem(_files(images),_files(masks)))


def _phash(image):
    coeff=dctn(np.asarray(image.convert('L').resize((32,32),Image.Resampling.LANCZOS),float),norm='ortho')[:8,:8].ravel()
    bits=coeff>np.median(coeff[1:]);bits[0]=False
    return int.from_bytes(np.packbits(bits).tobytes(),'big')


def prepare(c):
    v1=Path(c['v1_root']);root=Path(c['root']);dest=output(c)
    splits={key:read_json(v1/'splits'/f'{key}.json')['rows'] for key in ['train','val','test']}
    hashes=read_json(v1/'manifests/hashes.json');audit=read_json(v1/'manifests/audit.json')
    checks=verify_disjoint(splits,hashes,audit['links'])
    by_id={row['id']:row for rows in splits.values() for row in rows};test_ids={row['id'] for row in splits['test']}
    within_test=[edge for edge in audit['links'] if edge['a'] in test_ids and edge['b'] in test_ids and by_id[edge['a']]['collection']!=by_id[edge['b']]['collection']]
    atomic_json(dest/'reports/benchmark_partition_overlap.json',{'links':within_test,'count':len(within_test),
                'interpretation':'Detected image-only candidates within the reserved test set; benchmark partitions need not be independent. No such links enter development.'},immutable=True)
    for split,rows in splits.items():
        for row in rows:
            STOP.check()
            if file_hash(root/row['image'])!=row['image_sha256']:raise ValueError(f"Source image differs from frozen V1 audit: {row['id']}")
            if row['mask']:
                current=file_hash(root/row['mask'])
                if row.get('mask_sha256') and current!=row['mask_sha256']:raise ValueError('Source mask differs from frozen manifest')
                # Hashing test bytes here protects identity without fitting mask
                # statistics. Test pixels are decoded only after V2 is frozen.
                row['mask_sha256']=current
        atomic_json(dest/'config/splits'/f'{split}.json',{'rows':rows},immutable=True)
    protocol=read_json(v1/'splits/protocol.json')
    for key in ['threshold','normal_fpr_limit']:
        if c[key]!=protocol[key]:raise ValueError('Initial V2 decision protocol must match frozen V1')
    if [c['selection_size']]*2!=protocol['selection_grid']:raise ValueError('V2 initial checkpoint selection must preserve the V1 monitoring grid')
    atomic_json(dest/'config/v1_split_protocol.json',protocol,immutable=True)
    atomic_json(dest/'reports/leakage_verification.json',checks,immutable=True)
    exclusions=[]
    with (v1/'splits/exclusions.csv').open() as f:
        import csv
        exclusions=list(csv.DictReader(f))
    inventory={row['id']:row for row in read_json(v1/'manifests/all.json')['rows']}
    usage=[]
    def add(dataset,collection,available,predicate,reason=''):
        used=used_counts(splits,predicate)
        excluded=[r for r in exclusions if r['id'] in inventory and predicate(inventory[r['id']])]
        reasons=Counter(r['reason'] for r in excluded)
        usage.append({'dataset':dataset,'collection':collection,'available_pairs':available,
                      'training_used':used['train'],'validation_used':used['val'],'final_test_used':used['test'],
                      'excluded_duplicates':len(excluded),'reason_for_exclusion':reason or '; '.join(f'{key}: {n}' for key,n in sorted(reasons.items()))})
        usage[-1]['not_used_for_model']=available-sum(used.values())
    far=root/'dataset-2/Dataset';patients=[p for p in far.glob('Patient*') if p.is_dir()]
    positive=sum(_pairs(p/'Polyp',p/'BinaryMask') for p in patients);healthy=sum(len(_files(p/'Healthy')) for p in patients)
    if positive!=432 or healthy!=7749:raise ValueError('FAR source counts differ from the declared full dataset')
    add('dataset-2','FAR_polyp',positive,lambda row:row['domain']=='FAR' and row['positive'])
    add('dataset-2','FAR_healthy_derived_empty',healthy,lambda row:row['domain']=='FAR' and not row['positive'])
    add('dataset-2','Annotated_overlays',sum(len(_files(p/'Annotated')) for p in patients),lambda row:False,'Visualization overlays never used as model input or target')
    benchmark=root/'dataset-3/PolypDataset'
    base_count=_pairs(benchmark/'NewTRimage',benchmark/'NewTRmask')
    if base_count!=1450:raise ValueError('Expected 1450 unaugmented development pairs')
    add('dataset-3','NewTRimage/NewTRmask',base_count,lambda row:row['collection']=='train_base')
    for images,masks in [('NewTRimageDA','NewTRmaskDA'),('OLDTRimage','OLDTRimageMask')]:
        add('dataset-3',images+'/'+masks,_pairs(benchmark/images,benchmark/masks),lambda row:False,'Repeated/augmented source material; excluded from initial V2')
    for name,expected in [('Kvasir',100),('CVC-ClinicDB',62),('CVC-ColonDB',380),('CVC-300',60),('ETIS-LaribPolypDB',196)]:
        folder=benchmark/'TestDataset/TestDataset'/name;available=_pairs(folder/'images',folder/'masks')
        if available!=expected:raise ValueError(f'{name} collection is incomplete')
        add('dataset-3',name,available,lambda row,name=name:row['collection']=='test_'+name)
    primary_by_stem={Path(row['image']).stem:(split,row) for split,rows in splits.items() for row in rows
                     if row['collection']=='train_base' or row['collection']=='test_Kvasir'}
    provenance=[];sessile=[]
    for archive_name,label,expected in [('kvasir-seg.zip','Kvasir-SEG',1000),('kvasir-sessile.zip','Kvasir-Sessile',196)]:
        with ZipFile(root/'dataset-1'/archive_name) as archive:
            names=sorted(name for name in archive.namelist() if '/images/' in name and Path(name).suffix.lower() in EXT)
            masks=sorted(name for name in archive.namelist() if '/masks/' in name and Path(name).suffix.lower() in EXT)
            if len(names)!=expected:raise ValueError('Dataset-1 archive image count differs')
            if {Path(name).stem for name in names}!={Path(name).stem for name in masks}:raise ValueError('Dataset-1 image/mask provenance pairing differs')
            linked=Counter()
            for name in names:
                sid=Path(name).stem
                if sid not in primary_by_stem:
                    if label=='Kvasir-Sessile':sessile.append({'image_id':sid,'split':'excluded','reason':'Not in retained frozen membership'})
                    continue
                split,row=primary_by_stem[sid];linked[split]+=1
                if label=='Kvasir-SEG':
                    import io
                    with Image.open(io.BytesIO(archive.read(name))) as reference,Image.open(root/row['image']) as actual:
                        distance=(_phash(reference)^_phash(actual)).bit_count()
                    provenance.append({'image_id':sid,'split':split,'operational_id':row['id'],
                                       'original_file_sha256':hashlib.sha256(archive.read(name)).hexdigest(),
                                       'operational_file_sha256':row['image_sha256'],'phash_hamming':distance,
                                       'interpretation':'Filename identity plus image-similarity audit; resized/reencoded files need not be byte-identical'})
                else:sessile.append({'image_id':sid,'split':split,'operational_id':row['id'],'reason':'Provenance-linked subgroup; no duplicate training copy'})
        add('dataset-1',label,expected,lambda row:False,'Provenance/subgroup reference only; images already represented in dataset-3 are not duplicated')
        usage[-1].update({f'provenance_linked_{key}':linked[key] for key in ['train','val','test']})
    csv_write(dest/'reports/dataset_usage_report.csv',usage)
    csv_write(dest/'reports/provenance_links.csv',provenance);csv_write(dest/'reports/sessile_subgroup.csv',sessile)
    return splits

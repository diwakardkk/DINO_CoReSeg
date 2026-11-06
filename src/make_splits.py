"""Freeze splits only after conservative duplicate reconciliation."""
from collections import Counter, defaultdict
import random
from pathlib import Path

import numpy as np
from .common import RESEARCH, args, config, read_json, atomic_json, fingerprint, UnionFind, csv_write, scientific_config


def grouped_assign(groups, ratios, rng, weight):
    """Assign whole duplicate/patient components; balance label strata separately."""
    result={}; names=list(ratios)
    strata=defaultdict(list)
    for key,rows in groups.items():
        label=any(r['positive'] for r in rows)
        strata[(label,tuple(sorted({r['source'] for r in rows})))].append((key,rows))
    for values in strata.values():
        rng.shuffle(values); values.sort(key=lambda kv:-weight(kv[1]))
        total=sum(weight(v) for _,v in values); counts={k:0 for k in names}
        for key,rows in values:
            n=weight(rows)
            choice=min(names,key=lambda k: ((counts[k]+n-ratios[k]*total)**2-(counts[k]-ratios[k]*total)**2)/(ratios[k]*total+1))
            # Equivalent marginal normalized squared deficit cost.
            result[key]=choice;counts[choice]+=n
    return result


def run(c):
    audit=read_json(RESEARCH/'manifests/audit.json')
    rows=read_json(RESEARCH/'manifests/all.json')['rows'];byid={r['id']:r for r in rows}
    uf=UnionFind()
    for r in rows:uf.find(r['id'])
    patient_first={}
    for r in rows:
        if r['patient']:
            first=patient_first.setdefault(r['patient'],r['id']);uf.union(first,r['id'])
    for link in audit['links']:uf.union(link['a'],link['b'])
    components=defaultdict(list)
    for r in rows:components[uf.find(r['id'])].append(r)
    excluded={}
    for group in components.values():
        tests=[r for r in group if r['collection'].startswith('test_')]
        dev=[r for r in group if r['collection']=='train_base']
        fars=[r for r in group if r['domain']=='FAR']
        if tests:
            for r in dev+fars:excluded[r['id']]='component_linked_to_official_test_image'
        elif dev and fars:
            for r in fars:excluded[r['id']]='FAR_component_linked_to_benchmark_development'
    # One copy of exact decoded/normalized duplicate image content; do not silently
    # choose between contradictory positive/negative labels.
    hashes=read_json(RESEARCH/'manifests/hashes.json');exact=defaultdict(list)
    for h in hashes:
        r=byid[h['id']]
        if not r['collection'].startswith('test_') and r['id'] not in excluded:
            exact[h['normalized_sha256']].append(r)
    for group in exact.values():
        if len({r['positive'] for r in group})>1:
            for r in group:excluded[r['id']]='conflicting_labels_identical_normalized_content'
        else:
            for r in sorted(group,key=lambda r:r['id'])[1:]:excluded[r['id']]='redundant_normalized_exact_copy'
    devgroups=defaultdict(list);fargroups=defaultdict(list)
    for r in rows:
        if r['id'] in excluded or r['collection'].startswith('test_'):continue
        (fargroups if r['domain']=='FAR' else devgroups)[uf.find(r['id'])].append(r)
    rng=random.Random(c['seed'])
    dev_assignment=grouped_assign(devgroups,{'train':.8,'val':.2},rng,len)
    far_assignment=grouped_assign(fargroups,{'train':.7,'val':.15,'test':.15},rng,
                                  lambda g:len({r['patient'] for r in g}))
    splits={k:[] for k in ['train','val','test']}
    for r in rows:
        if r['id'] in excluded:continue
        if r['collection'].startswith('test_'):split='test'
        elif r['domain']=='FAR':split=far_assignment[uf.find(r['id'])]
        else:split=dev_assignment[uf.find(r['id'])]
        splits[split].append(dict(r,split=split,duplicate_component=uf.find(r['id'])))
    assigned={r['id']:split for split,rs in splits.items() for r in rs}
    unresolved=[]
    for link in audit['links']:
        a,b=link['a'],link['b']
        if a in assigned and b in assigned and assigned[a]!=assigned[b]:unresolved.append(link)
    if unresolved:raise RuntimeError(f'Unresolved cross-split duplicate candidates: {len(unresolved)}')
    for r in rows:
        if r['patient'] and r['id'] in assigned:
            if len({assigned[x['id']] for x in rows if x['patient']==r['patient'] and x['id'] in assigned})!=1:
                raise RuntimeError('Patient crosses splits')
    for split,rs in splits.items():
        if not any(r['positive'] for r in rs) or not any(not r['positive'] for r in rs):
            raise RuntimeError(f'No positive/normal representation in {split}')
        atomic_json(RESEARCH/'splits'/f'{split}.json',{'seed':c['seed'],'rows':rs},immutable=True)
    trainpos=[r['foreground_fraction'] for r in splits['train'] if r['positive']]
    stats={'small_polyp_fraction_max':float(np.quantile(trainpos,1/3)),
           'medium_polyp_fraction_max':float(np.quantile(trainpos,2/3)),
           'positive_area_median':float(np.median(trainpos)),'definition':'Training positives, original-grid foreground fraction tertiles',
           'threshold':c['threshold'],'normal_fpr_limit':c['normal_fpr_limit'],
           'selection':'Feasible epochs first; maximize macro positive Dice. If none feasible: minimize ungated normal FPR, then maximize Dice.',
           'selection_grid':[350,350],'normal_frame_positive':'Any predicted foreground pixel at threshold 0.5',
           'seed':c['seed'],'config_fingerprint':fingerprint(scientific_config(c))}
    atomic_json(RESEARCH/'splits/protocol.json',stats,immutable=True)
    report={'split_counts':{k:dict(Counter(r['collection'] for r in rs)) for k,rs in splits.items()},
            'FAR_patients':{k:len({r['patient'] for r in rs if r['patient']}) for k,rs in splits.items()},
            'FAR_positive_patients':{k:len({r['patient'] for r in rs if r['patient'] and r['positive']}) for k,rs in splits.items()},
            'excluded_counts':dict(Counter(excluded.values())),
            'unresolved_detected_cross_split_links':len(unresolved),
            'policy':'Conservative pHash-candidate components kept together; development components linked to benchmark test images excluded; no test mask pixels used.',
            'limitations':'Heuristic candidate grouping does not prove the absence of undetected near-duplicates; FAR proportions are approximate due to component grouping.'}
    atomic_json(RESEARCH/'splits/leakage_report.json',report,immutable=True)
    csv_write(RESEARCH/'splits/exclusions.csv',[{'id':k,'reason':v} for k,v in sorted(excluded.items())])
    print(report,flush=True)
    return report


if __name__=='__main__':
    a=args('Finalize immutable development and patient-grouped FAR splits');run(config(a.config,a.data_root))

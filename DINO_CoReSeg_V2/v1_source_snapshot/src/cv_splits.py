from collections import Counter
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from .common import UnionFind

def canonical_groups(rows):
    uf=UnionFind()
    for row in rows:
        group='component/'+row['duplicate_component'];uf.find(group)
        if row['patient']:uf.union(group,'patient/'+row['patient'])
    return np.array([uf.find('component/'+r['duplicate_component']) for r in rows])


def partition_check(rows,partitions,groups):
    identifiers=[set(part) for part in partitions]
    if set.union(*identifiers)!=set(range(len(rows))):raise ValueError('CV partition coverage is incomplete')
    for i,a in enumerate(identifiers):
        if not a:raise ValueError('Empty CV partition')
        if {rows[k]['positive'] for k in a}!={True,False}:raise ValueError('Each CV partition needs positive and healthy images')
        for b in identifiers[i+1:]:
            if a&b or set(groups[list(a)])&set(groups[list(b)]):raise ValueError('CV patient/duplicate group leakage')


def fold_definitions(rows,folds=5,inner_folds=5,seed=42):
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('Duplicate CV sample IDs')
    groups=canonical_groups(rows)
    strata=np.array([f"{r['source']}|{int(r['positive'])}" for r in rows])
    outer=StratifiedGroupKFold(n_splits=folds,shuffle=True,random_state=seed)
    result=[];heldout_seen=[]
    for number,(development,heldout) in enumerate(outer.split(np.zeros(len(rows)),strata,groups),1):
        inner=StratifiedGroupKFold(n_splits=inner_folds,shuffle=True,random_state=seed+number-1)
        train_sub,val_sub=next(inner.split(np.zeros(len(development)),strata[development],groups[development]))
        train=development[train_sub];val=development[val_sub]
        partition_check(rows,[train,val,heldout],groups)
        def ids(indices):return sorted(rows[int(i)]['id'] for i in indices)
        training=[rows[int(i)] for i in train]
        positive_areas=[r['foreground_fraction'] for r in training if r['positive']]
        result.append({'fold':number,'seed':seed+number-1,'train_ids':ids(train),
                       'inner_val_ids':ids(val),'heldout_ids':ids(heldout),
                       'positive_area_median':float(np.median(positive_areas)),
                       'counts':{'train':len(train),'inner_val':len(val),'heldout':len(heldout)},
                       'FAR_patients':{name:len({rows[int(i)]['patient'] for i in part if rows[int(i)]['patient']})
                                       for name,part in [('train',train),('inner_val',val),('heldout',heldout)]}})
        heldout_seen.extend(ids(heldout))
    if Counter(heldout_seen)!=Counter(r['id'] for r in rows):raise ValueError('Every CV sample must be held out exactly once')
    return result


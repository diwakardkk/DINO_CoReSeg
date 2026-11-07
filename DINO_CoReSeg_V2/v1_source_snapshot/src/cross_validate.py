from collections import Counter
from pathlib import Path
import numpy as np
from .common import (RESEARCH,load_split,atomic_json,atomic_copy,read_json,scientific_config,
                     fingerprint,csv_write,STOP)
from .cv_splits import fold_definitions
from .prototypes import build
from .train import train_model
from .evaluate import evaluate
from .cv_plots import run as plot_cv


def protocol_for(rows):
    areas=[r['foreground_fraction'] for r in rows if r['positive']]
    return {'positive_area_median':float(np.median(areas)),
            'small_polyp_fraction_max':float(np.quantile(areas,1/3)),
            'medium_polyp_fraction_max':float(np.quantile(areas,2/3)),
            'size_definition':'Training-only original-grid mask-area tertiles'}


def prepare(c):
    cv=c['cross_validation'];base=RESEARCH/'outputs'/(c['experiment']+'_cv5')
    rows=sorted(load_split('train')+load_split('val'),key=lambda r:r['id']);reserved=load_split('test')
    if {r['id'] for r in rows}&{r['id'] for r in reserved}:raise ValueError('Final test sample entered CV')
    if {r['patient'] for r in rows if r['patient']}&{r['patient'] for r in reserved if r['patient']}:raise ValueError('Final test patient entered CV')
    if {r['duplicate_component'] for r in rows}&{r['duplicate_component'] for r in reserved}:raise ValueError('Final test duplicate component entered CV')
    folds=fold_definitions(rows,cv['folds'],cv['inner_folds'],c['seed'])
    atomic_json(base/'development_pool.json',{'rows':rows},immutable=True)
    for fold in folds:atomic_json(base/f"splits/fold_{fold['fold']:02d}.json",fold,immutable=True)
    atomic_json(base/'protocol.json',{'config':scientific_config(c),'folds':folds,
                'ROC':'Outer held-out native-resolution pixels; quantized streaming score histogram',
                'curves':'Training and inner-validation monitors; predictive L/(1+L), macro pixel accuracy',
                'test_policy':'Original FAR/benchmark final tests excluded from CV'},immutable=True)
    return base,rows,folds


def run(c):
    base,rows,folds=prepare(c);byid={r['id']:r for r in rows};metrics=[]
    for fold in folds:
        STOP.check();directory=base/f"fold_{fold['fold']:02d}";directory.mkdir(parents=True,exist_ok=True)
        train=[byid[k] for k in fold['train_ids']];val=[byid[k] for k in fold['inner_val_ids']];heldout=[byid[k] for k in fold['heldout_ids']]
        p=build(c,train,fold['train_ids'],directory/'prototypes.pt',fold['seed']);protocol=protocol_for(train)
        def update_plots(run_directory):
            import csv
            with (run_directory/'training_history.csv').open() as f:history=list(csv.DictReader(f))
            epoch=int(history[-1]['epoch'])
            if epoch==1 or epoch%c['artifacts']['plot_every_epochs']==0 or epoch==c['max_epochs']:
                plot_cv(base,cv_folds(c),c['cross_validation']['roc_grid_points'])
        train_model(c,train,val,p,directory,protocol,fold['seed'],on_epoch=update_plots)
        evaluate(c,heldout,directory,protocol,partition_mode='outer')
        roc=directory/'metrics/outer_heldout_roc.npz';atomic_copy(roc,directory/'heldout_roc.npz')
        result=read_json(directory/'evaluation_status.json')
        with np.load(roc,allow_pickle=False) as z:auc=float(z['auc'])
        summary=read_json(directory/'FROZEN.json')
        atomic_json(directory/'status.json',{'completed':True,'epochs':summary['completed_epochs'],
                    'selected_epoch':summary['selected_epoch'],'pixel_auc':auc})
        metrics.append({'fold':fold['fold'],'pixel_auc':auc,'epochs':summary['completed_epochs'],
                        'selected_epoch':summary['selected_epoch']})
        import csv
        with (directory/'metrics/final_metrics.csv').open() as f:fold_scores=list(csv.DictReader(f))
        for row in fold_scores:
            names=['dice','iou','precision','recall','specificity','f2','boundary_f1','boundary_iou','hd95','assd','mae','pixel_accuracy'] if row['target']=='positive' else ['fp_frame','foreground_fraction','fp_components','frame_specificity']
            for name in names:
                if row.get(name) not in ['',None]:metrics[-1][row['target']+'_'+name]=float(row[name])
        csv_write(base/'fold_metrics.csv',metrics)
        atomic_json(base/'status.json',{'completed_folds':[r['fold'] for r in metrics],'required_folds':5,'completed':len(metrics)==5})
        plot_cv(base,cv_folds(c),c['cross_validation']['roc_grid_points'])
    aucs=[r['pixel_auc'] for r in metrics]
    aggregate={'completed':len(metrics)==5,'auc_mean':float(np.mean(aucs)),
               'auc_std':float(np.std(aucs,ddof=1)),'note':'Descriptive fold variability; ROC uses native pixels.'}
    for name in sorted(set().union(*(row.keys() for row in metrics))-{'fold','epochs','selected_epoch'}):
        values=[row[name] for row in metrics if name in row]
        aggregate[name]={'mean':float(np.mean(values)),'std':float(np.std(values,ddof=1))}
    from .cv_metrics import PixelROC
    pooled=PixelROC(c['evaluation']['roc_bins'])
    for fold in folds:
        with np.load(base/f"fold_{fold['fold']:02d}/heldout_roc.npz",allow_pickle=False) as z:
            pooled.positive+=z['positive_hist'];pooled.negative+=z['negative_hist']
    pooled.save(base/'pooled_heldout_roc.npz');aggregate['pooled_heldout_pixel_auc']=pooled.curve()[2]
    atomic_json(base/'cv_summary.json',aggregate)
    return base


def cv_folds(c):return c['cross_validation']['folds']

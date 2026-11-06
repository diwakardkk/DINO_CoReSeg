from collections import defaultdict
from pathlib import Path
import csv
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from .common import (RESEARCH,output,read_json,pair_arrays,atomic_json,csv_write)
from .figure_utils import save,architecture,segmentation_figure
from .cv_plots import draw_history


def history_plots(directory):
    path=directory/'training_history.csv'
    if not path.exists():return
    with path.open() as f:rows=list(csv.DictReader(f))
    if not rows:return
    dest=directory/'figures';dest.mkdir(parents=True,exist_ok=True);epochs=[int(r['epoch']) for r in rows]
    for name,metric,label in [('01_train_validation_loss','loss_0_1','Normalized predictive loss: L / (1 + L)'),
                              ('02_train_validation_accuracy','accuracy','Macro pixel accuracy (350 × 350)'),
                              ('validation_dice_iou','positive_dice','Positive Dice')]:
        fig,ax=plt.subplots(figsize=(7,4.4))
        for split,legend,color in [('train','Training','#136fbc'),('val','Validation','#d65b24')]:
            ax.plot(epochs,[float(r[f'{split}_{metric}']) for r in rows],label=legend,color=color)
        if name=='validation_dice_iou':ax.plot(epochs,[float(r['val_positive_iou']) for r in rows],label='Validation IoU',linestyle='--')
        ax.set(xlabel='Epoch',ylabel=label,ylim=(0,1));ax.legend(frameon=False);ax.spines[['top','right']].set_visible(False)
        fig.tight_layout();save(fig,dest/name)
    fig,ax=plt.subplots(figsize=(7,4.4))
    for key in ['loss','positive_seg','normal_bce','contrast','auxiliary']:
        ax.plot(epochs,[float(r['optimization_'+key]) for r in rows],label=key)
    ax.set(xlabel='Epoch',ylabel='Raw optimization loss components');ax.legend(frameon=False,fontsize=8)
    fig.tight_layout();save(fig,dest/'optimization_loss_components')


def result_plots(c,directory):
    directory=Path(directory);history_plots(directory)
    path=directory/'metrics/per_image.csv'
    if not path.exists():return
    with path.open() as f:records=list(csv.DictReader(f))
    dest=directory/'figures';dest.mkdir(parents=True,exist_ok=True)
    pos=[r for r in records if r['positive']=='True'];neg=[r for r in records if r['positive']=='False'];n=c['evaluation']['qualitative_examples']
    choices={'successful':sorted(pos,key=lambda r:-float(r['dice']))[:n],
             'lowest_dice':sorted(pos,key=lambda r:float(r['dice']))[:n],
             'small_polyp':sorted([r for r in pos if r.get('size_bin')=='small'],key=lambda r:r['id'])[:n],
             'normal_false_positive':sorted([r for r in neg if float(r['foreground_fraction'])>0],key=lambda r:-float(r['foreground_fraction']))[:n],
             'normal_examples':sorted(neg,key=lambda r:r['id'])[:n]}
    examples=[]
    for label,selected in choices.items():
        for record in selected:
            stem=directory/'predictions'/record['partition']/record['prediction_key']
            metadata=read_json(stem.with_suffix('.json'));row=metadata['row'];image,truth=pair_arrays(c,row)
            prediction=np.asarray(Image.open(stem.with_suffix('.png')))>0
            segmentation_figure(np.asarray(image),truth,prediction,f"{label} | {row['id']}",dest/f"{label}_{record['prediction_key'][:12]}",derived=not row['positive'])
            rgb=np.asarray(image);error=rgb.copy()
            for mask,color in [(prediction&truth,[0,180,60]),(prediction&~truth,[240,30,30]),(~prediction&truth,[30,90,240])]:
                error[mask]=(.5*error[mask]+.5*np.array(color)).astype(np.uint8)
            fig,axes=plt.subplots(1,4,figsize=(14.4,4.2))
            for ax,(array,title) in zip(axes,[(rgb,'Original'),(truth,'Ground truth'),(prediction,'Prediction'),(error,'TP green / FP red / FN blue')]):
                if array.ndim==2:ax.imshow(array,cmap='gray',vmin=0,vmax=1)
                else:ax.imshow(array)
                ax.axis('off');ax.set_title(title,fontsize=9)
            fig.suptitle(row['id'],fontsize=9);fig.tight_layout();save(fig,dest/f"error_{label}_{record['prediction_key'][:12]}")
            with np.load(stem.with_suffix('.npz'),allow_pickle=False) as z:counter=z['counter_evidence']
            fig,axes=plt.subplots(1,2,figsize=(7,3.6));axes[0].imshow(rgb);heat=axes[1].imshow(counter,cmap='magma',vmin=0,vmax=1)
            axes[0].set_title('Original');axes[1].set_title('Normal-tissue dissimilarity')
            for ax in axes:ax.axis('off')
            fig.colorbar(heat,ax=axes[1],fraction=.04);save(fig,dest/f"counter_{label}_{record['prediction_key'][:12]}")
            examples.append({'label':label,'id':row['id'],'checkpoint_sha256':metadata['checkpoint_sha256']})
    atomic_json(dest/'qualitative_selection.json',{'records':examples,'selection':'Highest/lowest Dice, small cases by ID, largest actual normal FP areas, and normal examples by ID; illustrative extremes, not population estimates.'})
    with (directory/'metrics/final_metrics.csv').open() as f:summary=list(csv.DictReader(f))
    positive=[r for r in summary if r['target']=='positive']
    if positive:
        fig,ax=plt.subplots(figsize=(8,4));ax.bar([r['partition'] for r in positive],[float(r['dice']) for r in positive],color='#297b87')
        ax.set(ylabel='Macro positive Dice',ylim=(0,1));ax.tick_params(axis='x',labelrotation=20);fig.tight_layout();save(fig,dest/'performance_by_dataset')
    fig,ax=plt.subplots(figsize=(6.5,5))
    for file in sorted((directory/'metrics').glob('*_roc.npz')):
        with np.load(file,allow_pickle=False) as z:ax.plot(z['fpr'],z['tpr'],label=f"{file.stem[:-4]}: AUC {float(z['auc']):.4f}")
    ax.plot([0,1],[0,1],':',color='#888888');ax.set(xlabel='False-positive pixel rate',ylabel='True-positive pixel rate',xlim=(0,1),ylim=(0,1))
    ax.legend(frameon=False,fontsize=8);fig.tight_layout();save(fig,dest/'03_roc_auc')
    with (directory/'metrics/lesion_size_metrics.csv').open() as f:sizes=list(csv.DictReader(f))
    fig,ax=plt.subplots(figsize=(8,4))
    for size in ['small','medium','large']:
        group=[r for r in sizes if r['size_bin']==size]
        if group:ax.plot([r['partition'] for r in group],[float(r['dice']) for r in group],marker='o',label=size)
    ax.set(ylabel='Macro positive Dice',ylim=(0,1));ax.legend(frameon=False);ax.tick_params(axis='x',labelrotation=20)
    fig.tight_layout();save(fig,dest/'lesion_size_performance')
    runtime=directory/'metrics/end_to_end_runtime.json'
    if runtime.exists():
        stats=read_json(runtime)['means'];fig,ax=plt.subplots(figsize=(6,4))
        ax.bar(['Backbone','Decoder','End-to-end'],[stats[k]*1000 for k in ['backbone_seconds','decoder_seconds','end_to_end_seconds']],color='#297b87')
        ax.set(ylabel='Batch-one milliseconds/image');fig.tight_layout();save(fig,dest/'runtime_summary')


def run(c):
    base=output(c);(base/'figures').mkdir(parents=True,exist_ok=True);architecture(base/'figures')
    for directory in [base/'holdout']+sorted((RESEARCH/'outputs'/(c['experiment']+'_cv5')).glob('fold_*')):
        if directory.is_dir():result_plots(c,directory)

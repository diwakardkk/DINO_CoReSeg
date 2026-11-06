"""Three requested CV figures; never generate curves without real saved data."""
import csv
import os
from pathlib import Path
import numpy as np
import tempfile
os.environ.setdefault('MPLCONFIGDIR',os.path.join(tempfile.gettempdir(),f'coreseg-mpl-{os.getuid()}'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import atomic_json,read_json


def save(fig,path):
    for ext in ['png','pdf','svg']:
        fig.savefig(path.with_suffix('.'+ext),dpi=300,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def draw_history(base,folds,metric,ylabel,filename):
    histories=[]
    for fold in range(1,folds+1):
        path=base/f'fold_{fold:02d}/training_history.csv'
        if path.exists():
            with path.open() as f:rows=list(csv.DictReader(f))
            if rows:histories.append((fold,rows))
    if not histories:return False
    fig,ax=plt.subplots(figsize=(7.4,4.6),facecolor='white');ax.set_facecolor('white')
    colors={'train':'#136fbc','val':'#d65b24'}
    for split in ['train','val']:
        for _,rows in histories:
            ax.plot([int(r['epoch']) for r in rows],[float(r[f'{split}_{metric}']) for r in rows],
                    color=colors[split],alpha=.22,linewidth=.9,linestyle='-' if split=='train' else '--')
        if len(histories)==folds:
            common=min(len(rows) for _,rows in histories)
            values=np.array([[float(r[f'{split}_{metric}']) for r in rows[:common]] for _,rows in histories])
            x=np.arange(1,common+1);mean=values.mean(0);std=values.std(0,ddof=1)
            ax.plot(x,mean,color=colors[split],linewidth=2,label=('Training' if split=='train' else 'Inner validation')+' mean',
                    linestyle='-' if split=='train' else '--')
            # Physical plot limits apply only to the display band. Measurements
            # are never clipped or normalized per fold/epoch.
            ax.fill_between(x,np.maximum(0,mean-std),np.minimum(1,mean+std),color=colors[split],alpha=.1)
        else:
            ax.plot([],[],color=colors[split],label=('Training' if split=='train' else 'Inner validation')+' (available folds)',
                    linestyle='-' if split=='train' else '--')
    ax.set(xlabel='Epoch',ylabel=ylabel,ylim=(0,1))
    complete=sum((base/f'fold_{fold:02d}/status.json').exists() and
                 read_json(base/f'fold_{fold:02d}/status.json').get('completed',False) for fold in range(1,folds+1))
    ax.set_title(f'{folds}-fold CV | {complete}/{folds} complete | {len(histories)}/{folds} with history',fontsize=10)
    ax.legend(frameon=False);ax.spines[['top','right']].set_visible(False)
    fig.text(.5,.01,'Thin lines: folds. Mean ±1 SD uses epochs available in all five folds; no extrapolation.',ha='center',fontsize=7)
    fig.tight_layout(rect=(0,.04,1,1));save(fig,base/'figures'/filename)
    return True


def run(base,folds=5,grid_points=1001):
    base=Path(base);(base/'figures').mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'figure.facecolor':'white','axes.facecolor':'white'})
    made=[]
    if draw_history(base,folds,'loss_0_1','Normalized segmentation loss: L / (1 + L)','01_train_validation_loss'):made.append('loss')
    if draw_history(base,folds,'accuracy','Macro image pixel accuracy','02_train_validation_accuracy'):made.append('accuracy')
    from .paper_cv_plots import generate
    finished=all((base/f'fold_{fold:02d}/status.json').exists() and
                 read_json(base/f'fold_{fold:02d}/status.json').get('completed',False)
                 for fold in range(1,folds+1))
    if generate(base,folds,dpi=600 if finished else 300):made.append('combined_fold_loss_accuracy')
    completed=[]
    for fold in range(1,folds+1):
        path=base/f'fold_{fold:02d}/heldout_roc.npz';status=base/f'fold_{fold:02d}/status.json'
        if path.exists() and status.exists() and read_json(status).get('completed',False):completed.append((fold,path))
    if completed:
        fig,ax=plt.subplots(figsize=(6.6,5.3));grid=np.linspace(0,1,grid_points);interpolated=[];aucs=[];bins=set()
        for fold,path in completed:
            with np.load(path,allow_pickle=False) as z:
                fpr=z['fpr'];tpr=z['tpr'];auc=float(z['auc']);bins.add(int(z['bins']))
            interp=np.interp(grid,fpr,tpr);interp[0]=0;interp[-1]=1
            ax.plot(grid,interp,linewidth=1.1,label=f'Fold {fold}: AUC {auc:.4f}')
            interpolated.append(interp);aucs.append(auc)
        if len(completed)==folds:
            mean=np.mean(interpolated,axis=0)
            ax.plot(grid,mean,color='black',linewidth=2,label=f'Mean fold AUC {np.mean(aucs):.4f} ± {np.std(aucs,ddof=1):.4f}')
        ax.plot([0,1],[0,1],color='#888888',linestyle=':',linewidth=1,label='Chance')
        ax.set(xlim=(0,1),ylim=(0,1),xlabel='False-positive pixel rate',ylabel='True-positive pixel rate')
        ax.set_title(f'Held-out pixel ROC | {len(completed)}/{folds} completed outer folds',fontsize=10)
        ax.spines[['top','right']].set_visible(False);ax.legend(frameon=False,fontsize=8,loc='lower right')
        bin_label=', '.join(f'{b:,}' for b in sorted(bins))
        fig.text(.5,.01,f'All native-resolution held-out pixels; {bin_label} score bins. Inner-validation checkpoint selection.',ha='center',fontsize=7)
        fig.tight_layout(rect=(0,.035,1,1));save(fig,base/'figures/03_roc_auc');made.append('roc_auc')
    atomic_json(base/'plot_status.json',{'plots_created':made,'completed_roc_folds':[f for f,_ in completed],
               'required_folds':folds,'full_cross_validation_complete':len(completed)==folds,
               'note':'Absent plots await actual measurements; no placeholder or synthetic curves are generated.'})
    return made

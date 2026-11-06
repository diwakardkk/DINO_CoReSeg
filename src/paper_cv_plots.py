"""Paper-style fold curves from saved histories; no CUDA or training imports.

Usage: python -m src.paper_cv_plots --results-dir outputs/EXPERIMENT_cv5
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import tempfile

os.environ.setdefault('MPLCONFIGDIR',os.path.join(tempfile.gettempdir(),f'coreseg-mpl-{os.getuid()}'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIELDS=('train_loss_0_1','val_loss_0_1','train_accuracy','val_accuracy')


def read_histories(base,folds=5):
    histories=[]
    for fold in range(1,folds+1):
        path=Path(base)/f'fold_{fold:02d}/training_history.csv'
        if not path.exists():continue
        with path.open(newline='') as stream:
            reader=csv.DictReader(stream)
            if not reader.fieldnames:continue
            missing={'epoch',*FIELDS}-set(reader.fieldnames)
            if missing:raise ValueError(f'{path}: missing fields {sorted(missing)}')
            rows=[]
            for raw in reader:
                epoch=int(raw['epoch'])
                if epoch<1 or (rows and epoch<=rows[-1]['epoch']):
                    raise ValueError(f'{path}: epochs must be positive and strictly increasing')
                row={'epoch':epoch}
                for key in FIELDS:
                    value=float(raw[key])
                    if not math.isfinite(value) or not 0<=value<=1:
                        raise ValueError(f'{path}: {key} must be a finite fraction in [0, 1]')
                    row[key]=value
                rows.append(row)
        if rows:histories.append((fold,rows))
    return histories


def draw(histories,scale='percent',max_epochs=None):
    if not histories:raise ValueError('No measured fold histories available')
    if scale not in ['percent','unit']:raise ValueError('Scale must be percent or unit')
    multiplier=100 if scale=='percent' else 1
    observed=max(row['epoch'] for _,rows in histories for row in rows)
    if max_epochs is not None and max_epochs<observed:
        raise ValueError('--max-epochs would hide measured epochs')
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,
                         'svg.fonttype':'none','figure.facecolor':'white','axes.facecolor':'white'}):
        fig,axes=plt.subplots(1,2,figsize=(14,5.6))
        colors=plt.get_cmap('tab10').colors
        for fold,rows in histories:
            epochs=[row['epoch'] for row in rows]
            for split,style,offset in [('train','-',0),('val','--',1)]:
                color=colors[(2*(fold-1)+offset)%len(colors)]
                label='Train' if split=='train' else 'Val'
                for ax,metric,kind in [(axes[0],'loss_0_1','Loss'),(axes[1],'accuracy','Accuracy')]:
                    ax.plot(epochs,[multiplier*row[f'{split}_{metric}'] for row in rows],
                            color=color,linestyle=style,linewidth=1.2,
                            label=f'Fold {fold} {label} {kind}')
        for ax in axes:
            ax.set_xlim(0,max(1,max_epochs or observed))
            ax.set_ylim(0,multiplier)
            ax.set_xlabel('Epoch',fontsize=15)
            ax.tick_params(labelsize=10)
            ax.grid(True,color='#bdb7b7',alpha=.65,linewidth=.8)
            ax.set_axisbelow(True)
            for spine in ax.spines.values():spine.set_color('#777777')
        axes[0].set_ylabel('Normalized loss (%)' if scale=='percent' else 'Normalized loss',fontsize=15)
        axes[1].set_ylabel('Accuracy (%)' if scale=='percent' else 'Accuracy',fontsize=15)
        axes[0].legend(loc='upper right',ncol=2,fontsize=8.5,framealpha=.9,
                       columnspacing=1.3,handlelength=2.6)
        axes[1].legend(loc='lower center',ncol=2,fontsize=8.5,framealpha=.9,
                       columnspacing=1.3,handlelength=2.6)
        axes[0].text(.5,-.25,'(a) Training and Validation Loss',transform=axes[0].transAxes,
                     ha='center',va='top',fontsize=14)
        axes[1].text(.5,-.25,'(b) Training and Validation Accuracy',transform=axes[1].transAxes,
                     ha='center',va='top',fontsize=14)
        fig.subplots_adjust(left=.065,right=.99,bottom=.25,top=.97,wspace=.24)
        return fig


def atomic_text(path,text):
    with tempfile.NamedTemporaryFile('w',dir=path.parent,delete=False) as stream:
        temporary=Path(stream.name);stream.write(text);stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def generate(base,folds=5,scale='both',max_epochs=None,dpi=600):
    base=Path(base).resolve()
    if folds<1:raise ValueError('folds must be positive')
    if dpi<72:raise ValueError('dpi must be at least 72')
    if scale not in ['percent','unit','both']:raise ValueError('Unknown scale')
    histories=read_histories(base,folds)
    if not histories:return []
    dest=base/'figures';dest.mkdir(parents=True,exist_ok=True)
    exports=[]
    for current in (['percent','unit'] if scale=='both' else [scale]):
        name='04_cv_loss_accuracy_percent' if current=='percent' else '05_cv_loss_accuracy_0_1'
        fig=draw(histories,current,max_epochs)
        try:
            for extension in ['png','pdf','svg','tiff']:
                path=dest/(name+'.'+extension)
                with tempfile.NamedTemporaryFile(dir=dest,suffix='.'+extension,delete=False) as temporary:
                    temporary_path=Path(temporary.name)
                try:
                    kwargs={'pil_kwargs':{'compression':'tiff_lzw'}} if extension=='tiff' else {}
                    with plt.rc_context({'pdf.fonttype':42,'svg.fonttype':'none'}):
                        fig.savefig(temporary_path,dpi=dpi,bbox_inches='tight',facecolor='white',**kwargs)
                    os.replace(temporary_path,path)
                finally:temporary_path.unlink(missing_ok=True)
                exports.append(path.name)
        finally:plt.close(fig)
    caption=(
        f'Training and inner-validation curves for grouped {folds}-fold cross-validation. '
        'Each fold has separately labelled training (solid) and validation (dashed) curves. '
        'The loss is the category-balanced final-output segmentation monitor L '
        '(binary cross-entropy plus positive-image soft Dice loss), displayed as L/(1+L), '
        'or 100 × L/(1+L) in the percentage figure. This is a bounded display transform, '
        'not a percentage error or the composite optimization loss. Accuracy is the macro '
        'mean of per-image pixel accuracy at the configured fixed threshold on the 350 × 350 monitoring grid; '
        'percentage accuracy equals 100 times this fraction. Validation denotes the inner '
        'validation set, not the outer-held-out evaluation set. Curves show actual completed '
        'epochs without smoothing, interpolation, extrapolation or per-fold rescaling. '
        'Missing folds and unfinished epochs are not fabricated.'
    )
    metadata={'folds_expected':folds,'folds_available':[fold for fold,_ in histories],
              'epochs_available':{str(fold):[row['epoch'] for row in rows] for fold,rows in histories},
              'input_files':[str((base/f'fold_{fold:02d}/training_history.csv').relative_to(base)) for fold,_ in histories],
              'scale':scale,'dpi':dpi,'exports':exports,'caption':caption}
    atomic_text(dest/'cv_loss_accuracy_caption.txt',caption+'\n')
    atomic_text(dest/'cv_loss_accuracy_manifest.json',json.dumps(metadata,indent=2)+'\n')
    return exports


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir',type=Path,required=True)
    parser.add_argument('--folds',type=int,default=5)
    parser.add_argument('--scale',choices=['percent','unit','both'],default='both')
    parser.add_argument('--max-epochs',type=int,default=None)
    parser.add_argument('--dpi',type=int,default=600)
    args=parser.parse_args()
    try:exports=generate(args.results_dir,args.folds,args.scale,args.max_epochs,args.dpi)
    except (ValueError,OSError) as exc:parser.exit(2,str(exc)+'\n')
    if not exports:parser.exit(2,'No completed epoch histories found; no plot generated.\n')
    print(f'Saved {len(exports)} figure files in {args.results_dir.resolve()/"figures"}')


if __name__=='__main__':main()

"""Regenerate paper figures from CSVs/artifacts; never imports PyTorch/CUDA."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import numpy as np
from PIL import Image
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'.mplcache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT=Path(__file__).resolve().parents[1]


def read_csv(path):
    with Path(path).open() as stream:return list(csv.DictReader(stream))


def save(fig,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with plt.rc_context({'pdf.fonttype':42,'svg.fonttype':'none'}):
        for extension in ['pdf','svg','png']:fig.savefig(path.with_suffix('.'+extension),dpi=300,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def values(rows,split,metric):
    aliases={'loss':[f'{split}_segmentation_loss',f'{"train" if split=="training" else "val"}_raw_segmentation_loss'],
             'accuracy':[f'{split}_positive_pixel_accuracy',f'{"train" if split=="training" else "val"}_positive_accuracy'],
             'dice':[f'{split}_positive_dice',f'{"train" if split=="training" else "val"}_positive_dice']}
    result={}
    for row in rows:
        for key in aliases[metric]:
            if row.get(key) not in ['',None]:
                value=float(row[key])
                if not np.isfinite(value):break
                result[int(row['epoch'])]=value;break
    return result


def history_paths(base):
    for relative in ['logs/training_history.csv','training_history.csv']:
        if (base/relative).exists():return [(None,base/relative)]
    paths=[]
    for folder in sorted(base.glob('fold_*')):
        for relative in ['logs/training_history.csv','training_history.csv']:
            if (folder/relative).exists():paths.append((folder.name,folder/relative));break
    return paths


def training_figures(base,dest):
    histories=[(label,read_csv(path)) for label,path in history_paths(base)]
    histories=[pair for pair in histories if pair[1]]
    if not histories:return []
    cv=histories[0][0] is not None;colors={'training':'#0072B2','validation':'#D55E00'};made=[]
    for metric,filename,label in [('accuracy','training_curves','Positive-image pixel accuracy (%)'),
                                   ('dice','training_curves_dice','Positive-image Dice')]:
        with plt.rc_context({'font.family':'DejaVu Sans','font.size':8,'figure.facecolor':'white','axes.facecolor':'white'}):
            fig,axes=plt.subplots(1,2,figsize=(7.2,3.05));observed=[];legend=[];labels=[]
            for index,(measure,ylabel) in enumerate([('loss','Raw segmentation loss'),(metric,label)]):
                ax=axes[index]
                for split,style,name in [('training','-','Training'),('validation','--','Inner validation' if cv else 'Validation')]:
                    series=[values(rows,split,measure) for _,rows in histories];series=[row for row in series if row]
                    if not series:continue
                    multiplier=100 if measure=='accuracy' else 1
                    for curve in series:
                        epochs=sorted(curve);observed+=epochs
                        line,=ax.plot(epochs,[multiplier*curve[e] for e in epochs],style,color=colors[split],
                                      alpha=.16 if cv else 1,linewidth=.6 if cv else 1.4)
                    if cv:
                        common=sorted(set.intersection(*(set(curve) for curve in series)))
                        if common:
                            array=np.array([[multiplier*curve[e] for e in common] for curve in series])
                            mean=array.mean(0);sd=array.std(0,ddof=1) if len(array)>1 else np.zeros(len(common))
                            line,=ax.plot(common,mean,style,color=colors[split],linewidth=1.5)
                            ax.fill_between(common,mean-sd,mean+sd,color=colors[split],alpha=.07,linewidth=0)
                    if index==0:legend.append(line);labels.append(name+(' mean' if cv else ''))
                ax.set(xlabel='Epoch',ylabel=ylabel)
                if measure=='accuracy':ax.set_ylim(0,100)
                elif measure=='dice':ax.set_ylim(0,1)
                else:ax.set_ylim(bottom=0)
                ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15,linewidth=.5)
                ax.text(.5,-.30,'(a) Raw segmentation loss' if index==0 else ('(b) Positive-image pixel accuracy' if metric=='accuracy' else '(b) Positive-image Dice'),
                        transform=ax.transAxes,ha='center',va='top',fontsize=8)
            if not observed:plt.close(fig);continue
            for ax in axes:ax.set_xlim(0,max(observed))
            fig.legend(legend,labels,loc='upper center',bbox_to_anchor=(.5,1.025),ncol=2,frameon=False,fontsize=8)
            fig.subplots_adjust(left=.09,right=.985,bottom=.28,top=.86,wspace=.32);save(fig,dest/filename);made.append(filename)
    (dest/'training_curves_caption.txt').write_text(
        ('Grouped CV: thin curves are individual available folds. Bold means and SD bands use only epoch numbers shared by every available fold. '
         f'{len(histories)} fold histories were available; missing folds are not generated. ' if cv else 'Single holdout experiment; these are not five-fold curves. ')+
        'Loss is the actual raw final-output segmentation monitor, not a percentage and not L/(1+L). '
        'Accuracy is positive-image pixel accuracy multiplied by 100. Dice uses [0,1]. '
        'The monitor grid and threshold are defined by the experiment configuration. Validation is development validation. '
        'No missing epochs are filled; no smoothing/extrapolation is applied. Raw loss y-limits include early training values.\n')
    return made


def aligned_images(data_root,row,prediction):
    with Image.open(Path(data_root)/row['image']) as stream:rgb=np.asarray(stream.convert('RGB'))
    if row['mask']:
        with Image.open(Path(data_root)/row['mask']) as stream:truth=np.asarray(stream.convert('L'))>=128
    else:truth=np.zeros(rgb.shape[:2],bool)
    with Image.open(prediction) as stream:pred=np.asarray(stream)>0
    if truth.shape!=rgb.shape[:2] or pred.shape!=truth.shape:raise ValueError('Original/GT/prediction coordinate mismatch; no visual stretching allowed')
    return rgb,truth,pred


def qualitative(base,dest,data_root):
    metrics=base/'evaluation/metrics/per_image.csv'
    if not metrics.exists():metrics=base/'metrics/per_image.csv'
    frozen=base/'checkpoints/FROZEN.json'
    if not frozen.exists():frozen=base/'FROZEN.json'
    if not metrics.exists() or not frozen.exists():return False
    marker=json.loads(frozen.read_text());records=read_csv(metrics)
    quantiles={'Kvasir':.5,'CVC-ClinicDB':.5,'CVC-ColonDB':.5,'ETIS-LaribPolypDB':.25,'FAR':.5}
    config_path=base/'config/experiment_config.json'
    if config_path.exists():quantiles.update(json.loads(config_path.read_text()).get('evaluation',{}).get('qualitative_quantiles',{}))
    choices=[]
    for source,row_label,q in [('Kvasir','Kvasir',.5),('CVC-ClinicDB','ClinicDB',.5),('CVC-ColonDB','ColonDB',.5),('ETIS-LaribPolypDB','ETIS',.25),('FAR','FAR polyp',.5)]:
        q=quantiles[source]
        group=sorted((r for r in records if r['partition']==source and r['positive']=='True'),key=lambda r:(float(r['dice']),r['id']))
        if group:choices.append((row_label,group[int(q*(len(group)-1))],f'Predeclared Dice quantile {q:g}'))
    normals=sorted((r for r in records if r['positive']=='False'),key=lambda r:r['id'])
    if normals:choices.append(('FAR normal',normals[(len(normals)-1)//2],'Predeclared median identifier; derived empty ground truth'))
    if not choices:return False
    images=[];selection=[]
    for label,record,reason in choices:
        path=base/'predictions'/record['partition']/record['prediction_key']
        metadata=json.loads(path.with_suffix('.json').read_text())
        if metadata['checkpoint_sha256']!=marker['checkpoint_sha256']:raise ValueError('Qualitative prediction checkpoint differs from frozen model')
        rgb,truth,pred=aligned_images(data_root,metadata['row'],path.with_suffix('.png'))
        probability=path.with_suffix('.npz')
        if probability.exists():
            with np.load(probability,allow_pickle=False) as z:prob=z['probability']
            threshold=metadata.get('threshold',marker['threshold'])
            if prob.shape!=truth.shape or not np.array_equal(prob>=threshold,pred):raise ValueError('Selected probability map disagrees with saved mask')
        error=rgb.copy()
        for mask,color in [(pred&truth,[0,175,85]),(pred&~truth,[215,40,35]),(~pred&truth,[25,100,225])]:
            error[mask]=(.45*error[mask]+.55*np.array(color)).astype(np.uint8)
        images.append((label,rgb,truth,pred,error))
        selection.append({'dataset':record['partition'],'image_id':record['id'],'checkpoint_hash':marker['checkpoint_sha256'],
                          'Dice':record.get('dice') or None,'IoU':record.get('iou') or None,'selection_reason':reason})
    ratios=[item[1].shape[0]/item[1].shape[1] for item in images]
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':8,'figure.facecolor':'white','axes.facecolor':'white'}):
        fig,axes=plt.subplots(len(images),4,figsize=(7.2,1.64*sum(ratios)+.55),squeeze=False,gridspec_kw={'height_ratios':ratios})
        for index,(label,rgb,truth,pred,error) in enumerate(images):
            for column,array in enumerate([rgb,truth,pred,error]):
                ax=axes[index,column]
                if array.ndim==2:ax.imshow(array,cmap='gray',vmin=0,vmax=1,interpolation='nearest')
                else:ax.imshow(array)
                ax.axis('off')
                if index==0:ax.set_title(['Original image','Ground truth','Predicted mask','TP / FP / FN overlay'][column],fontsize=8,pad=6)
            axes[index,0].text(-.04,.5,label,transform=axes[index,0].transAxes,rotation=90,va='center',ha='right',fontsize=8)
        fig.legend(handles=[Patch(color='#00af55',label='TP'),Patch(color='#d72823',label='FP'),Patch(color='#1964e1',label='FN')],
                   loc='lower center',ncol=3,frameon=False,fontsize=8,bbox_to_anchor=(.5,-.005))
        fig.subplots_adjust(left=.08,right=.995,top=.96,bottom=.05,wspace=.035,hspace=.08)
        save(fig,dest/'qualitative_segmentation')
    with (dest/'qualitative_selection.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(selection[0]));writer.writeheader();writer.writerows(selection)
    (dest/'qualitative_caption.txt').write_text('Native-resolution qualitative outputs of the frozen selected checkpoint. Positive cases use predeclared per-dataset Dice quantiles recorded in the selection CSV; the default ETIS quantile is the lower quartile to include a difficult case. Healthy GT is a derived empty mask. Green/red/blue indicate TP/FP/FN relative to GT. No mask filtering, geometric stretching or failure concealment is applied. Blank normal Dice/IoU are intentional.\n')
    return True


def generate(base,dest,data_root=None):
    base=Path(base).resolve();dest=Path(dest).resolve()
    if not dest.is_relative_to(ROOT):raise ValueError('New reports must remain inside the independent V2 folder')
    dest.mkdir(parents=True,exist_ok=True);made=training_figures(base,dest)
    if data_root and qualitative(base,dest,data_root):made.append('qualitative_segmentation')
    (dest/'plot_manifest.json').write_text(json.dumps({'source':str(base),'figures_created':made,'raw_artifacts_only':True},indent=2)+'\n')
    return made


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--experiment',required=True)
    parser.add_argument('--data-root',type=Path);parser.add_argument('--v1-root',type=Path)
    args=parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+',args.experiment):parser.error('Invalid experiment name')
    root=(args.v1_root.resolve() if args.v1_root else ROOT)/'outputs'/args.experiment
    base=root/'holdout' if (root/'holdout').is_dir() else root
    dest=ROOT/'outputs/v1_corrected_reports'/args.experiment/'figures' if args.v1_root else root/'figures'
    data=args.data_root or os.environ.get('CORESEG_DATA_ROOT')
    if not data and (root/'config/resolved_paths.json').exists():data=json.loads((root/'config/resolved_paths.json').read_text())['root']
    print(generate(base,dest,data))


if __name__=='__main__':main()

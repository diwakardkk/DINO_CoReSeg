import os
import tempfile
os.environ.setdefault('MPLCONFIGDIR',os.path.join(tempfile.gettempdir(),f'coreseg-mpl-{os.getuid()}'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

plt.rcParams.update({'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white','font.family':'DejaVu Sans','font.size':9})

def save(fig,path):
    fig.savefig(path.with_suffix('.png'),dpi=300,bbox_inches='tight',facecolor='white')
    fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight',facecolor='white')
    fig.savefig(path.with_suffix('.svg'),bbox_inches='tight',facecolor='white');plt.close(fig)


def architecture(dest):
    fig,ax=plt.subplots(figsize=(13,5));ax.set(xlim=(0,13),ylim=(0,5));ax.axis('off')
    boxes=[(.2,3.2,2,1,'RGB 350 × 350\nFrozen DINOv2-S/14'),(3,3.2,2.1,1,'Spatial tokens\n384 × 25 × 25'),
           (5.9,3.2,2.1,1,'Projection + initial head\n64 × 25 × 25'),(9,3.2,2.9,1,'CEF: signed residual\nInitial + tanh(γ) × residual'),
           (3,1,2.5,1,'8 normal prototypes\nTraining background only'),(6.1,1,2.3,1,'Counter-evidence\nNormal dissimilarity'),
           (9,.5,2.9,1.6,'RPR at 100 × 100\nFused + DINO + RGB CNN\nResidual correction\nFinal logits 350 × 350')]
    for x,y,w,h,t in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.08',facecolor='white',edgecolor='#263d4f',linewidth=1.2))
        ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=9)
    for a,b in [((2.3,3.7),(2.85,3.7)),((5.2,3.7),(5.75,3.7)),((8.1,3.7),(8.85,3.7)),
                ((4,3.1),(4,2.15)),((5.6,1.5),(5.95,1.5)),((7.3,2.1),(10.3,3.05)),((10.5,3.05),(10.5,2.2))]:
        ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':'#263d4f','lw':1.1})
    ax.text(.3,.5,'Proposed implementation; no baseline or ablation results.\nDissimilarity is an auxiliary cue, not proof of a lesion.',fontsize=9)
    save(fig,dest/'architecture')


def csv_rows(path):
    with path.open() as f:return list(csv.DictReader(f))


def draw_segmentation_row(axes,rgb,truth,pred,derived=False,titles=True):
    """Localize only from model predictions; ground truth is a separate panel."""
    overlay=rgb.copy()
    overlay[pred]=(.65*rgb[pred]+.35*np.array([0,220,90])).astype(np.uint8)
    panels=[(rgb,'Original'),(truth,'Ground truth'+(' (derived empty)' if derived else '')),
            (pred,'Predicted mask'),(overlay,'Predicted localization')]
    for ax,(image,title) in zip(axes,panels):
        if image.ndim==2:ax.imshow(image,cmap='gray',vmin=0,vmax=1,interpolation='nearest')
        else:ax.imshow(image)
        if titles:ax.set_title(title,fontsize=10)
        ax.axis('off')
    ys,xs=np.where(pred)
    if len(xs):
        # A single enclosing box over ALL predicted foreground. No component
        # filtering or ground-truth-derived localization is applied.
        x0,y0,x1,y1=int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())
        axes[3].add_patch(Rectangle((x0-.5,y0-.5),x1-x0+1,y1-y0+1,
                                    fill=False,edgecolor='#ffd54f',linewidth=1.2))
        if not pred.all():axes[3].contour(pred.astype(float),levels=[.5],colors=['#ffd54f'],linewidths=.7)
    else:
        axes[3].text(.5,.04,'No foreground predicted',transform=axes[3].transAxes,
                     ha='center',fontsize=8,color='white',bbox=dict(facecolor='black',alpha=.7,pad=2))


def segmentation_figure(rgb,truth,pred,title,path,derived=False):
    fig,axes=plt.subplots(1,4,figsize=(14.4,4.2))
    draw_segmentation_row(axes,rgb,truth,pred,derived)
    fig.suptitle(title,fontsize=10)
    fig.text(.5,.015,'Green: predicted foreground. Yellow: predicted contour and enclosing box. Threshold: 0.5.',
             ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.04,1,.94));save(fig,path)



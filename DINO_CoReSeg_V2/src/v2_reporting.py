"""Selected-checkpoint reporting and a single explicit experiment report."""
from pathlib import Path
import csv
from .common import atomic_json,csv_write,read_json,file_hash
from .v2_config import output


def selected_record(state,frozen):
    selected=int(frozen['selected_epoch'])
    if state['epoch']+1!=selected:raise ValueError('Checkpoint epoch disagrees with frozen selection')
    rows=[row for row in state['history'] if int(row['epoch'])==selected]
    if len(rows)!=1:raise ValueError('Selected epoch is not uniquely present in saved history')
    return rows[0]


def corrected_v1_report(c):
    import torch
    folder=Path(c['v1_root'])/'outputs/dino_coreseg_gpu_100/holdout'
    frozen=read_json(folder/'FROZEN.json');checkpoint=folder/'best.pt'
    if file_hash(checkpoint)!=frozen['checkpoint_sha256']:raise ValueError('V1 selected checkpoint changed')
    state=torch.load(checkpoint,map_location='cpu',weights_only=False);row=selected_record(state,frozen)
    validation=float(row['val_positive_dice'])
    with (folder/'metrics/final_metrics.csv').open() as f:summary=list(csv.DictReader(f))
    dest=output(c)/'reports/v1_reporting_review';dest.mkdir(parents=True,exist_ok=True)
    csv_write(dest/'domain_gaps_selected_epoch.csv',[{'partition':r['partition'],'selected_epoch':frozen['selected_epoch'],
        'selected_validation_dice':validation,'native_test_dice':float(r['dice']),
        'validation_minus_test':validation-float(r['dice']),
        'interpretation':'Descriptive cross-grid mixed-source validation gap; not a causal domain-shift estimate'}
        for r in summary if r['target']=='positive'])
    atomic_json(dest/'review.json',{'original_V1_files_modified':False,'checkpoint_sha256':file_hash(checkpoint),
                'selected_epoch':frozen['selected_epoch'],'matched_validation_record':row,
                'note':'V1 best.pt may already end at its selected epoch. This explicit lookup does not assert that the historical report was wrong.'})


def early_stopping_analysis(history,patience,limit):
    best=(-1,-float('inf'),-float('inf'));selected=None;bad=0;stop=None
    for row in history:
        dice=float(row['validation_positive_dice']);fpr=float(row['validation_normal_fpr'])
        key=(1,dice,-fpr) if fpr<=limit else (0,-fpr,dice)
        if key>best:best=key;selected=int(row['epoch']);bad=0
        else:bad+=1
        if bad>=patience:stop=int(row['epoch']);break
    return {'counterfactual_only':True,'patience':patience,'would_stop_epoch':stop,
            'selected_epoch_at_stop_or_last_observation':selected,'observed_epochs':len(history),
            'interpretation':'Replays the existing validation-only selection rule; does not change actual training or infer unobserved future epochs.'}


def report(c):
    dest=output(c);marker=dest/'checkpoints/FROZEN.json';evaluation=dest/'evaluation/evaluation_status.json'
    lines=['# DINO-CoReSeg V2 experiment', '',f"Experiment: `{c['experiment']}`",'',
           f"Input {c['image_size']} × {c['image_size']}; patch {c['patch_size']}; grid {c['grid_size']} × {c['grid_size']}; "
           f"RGB refinement {c['rgb_size']} × {c['rgb_size']}; variant `{c['variant']}`.", '',
           'V1 is read-only. Dataset usage and detected-leakage verification are recorded separately. '
           'Official test collections and FAR test patients are excluded from development selection.', '',
           'Training logs retain raw loss components, monitor metrics and explicit selection metadata. '
           f"The configured {c['selection_size']}-grid selection monitor, threshold and FPR constraint follow the frozen V1 protocol. "
           'Native final evaluation is separate.','']
    history_path=dest/'logs/training_history.csv'
    if history_path.exists():
        with history_path.open() as stream:history=list(csv.DictReader(stream))
        atomic_json(dest/'reports/early_stopping_analysis.json',early_stopping_analysis(history,c['patience'],c['normal_fpr_limit']))
    if marker.exists():
        frozen=read_json(marker)
        lines += [f"Training complete: {frozen['completed_epochs']} epochs. Selected epoch: {frozen['selected_epoch']}. "
                  f"Constraint satisfied: {frozen['FPR_constraint_satisfied']}.",'']
    else:lines+=['Training is not complete. No performance improvement is claimed.','']
    if evaluation.exists():
        lines += ['Evaluation status: `'+str(read_json(evaluation))+'`.','',
                  'Per-dataset summaries must accompany any pooled statistic. Native HD95/ASSD are in native pixels, '
                  'with diagonal penalties for empty predictions. Valid-surface-only summaries and standardized-grid '
                  'surface metrics are provided separately. Pixel recall and connected-component lesion detection '
                  'recall are different quantities.','']
    lines += ['Decoder-only timing is not end-to-end inference. Live backbone/decoder/end-to-end timings are '
              'reported separately when measured. A missing measurement is JSON null / an empty CSV field, not zero.', '',
              'Figures can be regenerated using `python -m src.plot_paper --experiment '+c['experiment']+'`.',
              'Optional cleanup is dry-run only. No V1 files or active checkpoints are deleted.','']
    path=dest/'reports/EXPERIMENT_REPORT.md';path.parent.mkdir(parents=True,exist_ok=True);path.write_text('\n'.join(lines))

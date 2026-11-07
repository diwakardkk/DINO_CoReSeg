"""Read-only V1 capture and verification, written exclusively inside V2."""
from pathlib import Path
import csv
import shutil
import torch
from .common import RESEARCH,read_json,atomic_json,file_hash

V1_DIRS=['outputs/dino_coreseg_gpu_100','outputs/dino_coreseg_gpu_100_cv5','splits','manifests']


def inventory(root):
    root=Path(root);files=[]
    for relative in V1_DIRS:
        folder=root/relative
        if not folder.is_dir():raise FileNotFoundError(f'Required frozen V1 directory missing: {folder}')
        files.extend(p for p in folder.rglob('*') if p.is_file())
    for folder in ['src','configs']:
        files.extend(p for p in (root/folder).rglob('*') if p.is_file() and p.suffix in ['.py','.yaml'])
    for relative in ['cache/backbone_identity.json','vendor/dinov2/SOURCE_COMMIT.txt','vendor/dinov2_vits14_pretrain.pth']:
        if (root/relative).is_file():files.append(root/relative)
    files.extend(p for p in (root/'vendor/dinov2').rglob('*.py') if p.is_file())
    return {str(p.relative_to(root)):file_hash(p) for p in sorted(files)}


def verify(root,guard):
    current=inventory(root)
    if current!=guard['files']:raise RuntimeError('V1 preservation check failed: files were changed, added or removed')
    return True


def capture(c):
    from .v2_config import output
    root=Path(c['v1_root']);dest=output(c)/'reports/v1_preservation';path=dest/'manifest.json'
    if root.resolve()==Path(c['research']).resolve():raise ValueError('V1 and V2 directories must differ')
    if path.exists():
        guard=read_json(path);verify(root,guard);return guard
    holdout=root/'outputs/dino_coreseg_gpu_100/holdout'
    frozen=read_json(holdout/'FROZEN.json')
    if file_hash(holdout/'best.pt')!=frozen['checkpoint_sha256']:raise ValueError('V1 frozen checkpoint hash mismatch')
    evaluation=read_json(holdout/'evaluation_status.json')
    if not evaluation['completed']:raise ValueError('V1 final evaluation is incomplete')
    if evaluation['checkpoint_sha256']!=frozen['checkpoint_sha256']:raise ValueError('V1 final evaluation uses another checkpoint')
    cv=root/'outputs/dino_coreseg_gpu_100_cv5'
    if not read_json(cv/'status.json')['completed']:raise ValueError('V1 CV is incomplete')
    for fold in range(1,6):
        directory=cv/f'fold_{fold:02d}';marker=read_json(directory/'FROZEN.json')
        if file_hash(directory/'best.pt')!=marker['checkpoint_sha256']:raise ValueError('V1 CV checkpoint changed')
    state=torch.load(holdout/'best.pt',map_location='cpu',weights_only=False)
    implementation=state.get('identity',{}).get('implementation',{})
    if not implementation:raise ValueError('V1 checkpoint has no saved implementation identity')
    comparisons={name:(root/'src'/name).is_file() and file_hash(root/'src'/name)==expected for name,expected in implementation.items()}
    atomic_json(dest/'source_review.json',{'checkpoint_implementation_identity':implementation,'source_matches_checkpoint':comparisons})
    if not all(comparisons.values()):raise ValueError('V1 scientific source differs from its saved checkpoint identity; inspect the V2 source_review.json before a controlled comparison')
    from .v2_reporting import selected_record
    record=selected_record(state,read_json(holdout/'FROZEN.json'))
    with (holdout/'training_history.csv').open() as stream:history=list(csv.DictReader(stream))
    if len(history)!=frozen['completed_epochs']:raise ValueError('V1 completed epoch count differs from history CSV')
    matched=[row for row in history if int(row['epoch'])==frozen['selected_epoch']]
    if len(matched)!=1 or abs(float(matched[0]['val_positive_dice'])-record['val_positive_dice'])>1e-8:raise ValueError('V1 history CSV does not match the frozen selected checkpoint')
    with (holdout/'metrics/final_metrics.csv').open() as stream:metrics=list(csv.DictReader(stream))
    if sum(int(row['images']) for row in metrics)!=evaluation['images']:raise ValueError('V1 metric image counts differ from evaluation completion')
    guard={'v1_root':str(root),'git_directory_present':(root/'.git').is_dir(),
           'source_is_identified_by':'SHA-256, not Git alone','files':inventory(root),
           'selected_epoch':frozen['selected_epoch'],'selected_validation_record':record,
           'training_epochs_in_csv':len(history),'final_evaluation_status':evaluation,'final_test_metrics':metrics,
           'checkpoint_implementation_identity':implementation,'source_matches_checkpoint':comparisons,
           'scope':'All V1 scientific output, split and manifest files plus Python/YAML source; cache contents are not copied or altered.'}
    for p in (root/'src').glob('*.py'):
        target=dest/'source'/p.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    atomic_json(path,guard,immutable=True)
    return guard

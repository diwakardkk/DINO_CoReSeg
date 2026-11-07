"""List V2-only optional cleanup candidates; this module never deletes files."""
import json
from .v2_config import output
from .common import atomic_json


def dry_run(c):
    root=output(c);candidates=[]
    for path in root.rglob('*.tmp'):
        candidates.append({'path':str(path.relative_to(root)),'bytes':path.stat().st_size,'reason':'Temporary artifact; verify no writer is active before considering deletion'})
    for path in (root/'checkpoints').glob('epoch_*.pt'):
        candidates.append({'path':str(path.relative_to(root)),'bytes':path.stat().st_size,'reason':'Optional retained intermediate checkpoint; preserve selected best and resumable last'})
    result={'dry_run':True,'deletion_implemented':False,'approval_required_before_any_deletion':True,'scope':str(root),
            'candidates':candidates,'protected':['checkpoints/best.pt','checkpoints/last.pt','checkpoints/FROZEN.json','all V1 files','all shared datasets and caches']}
    atomic_json(root/'reports/cleanup_dry_run.json',result);return result

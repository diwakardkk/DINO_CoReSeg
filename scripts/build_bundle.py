"""Build a portable code ZIP, excluding machine-specific run artifacts."""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED

root=Path(__file__).resolve().parents[1]
excluded={'.venv','.git','__pycache__','.pytest_cache','outputs','cache','manifests','splits','.mplcache','dataset_audit'}
excluded_files={'BUNDLE_SHA256.json','local_preflight.txt','requirements.server.lock.txt','dinov2_vits14_pretrain.pth'}
def included(path):
    relative=path.relative_to(root)
    return path.is_file() and not any(p in excluded for p in relative.parts) and path.name not in excluded_files and path.suffix not in ['.pyc','.tmp']
files=sorted(p for p in root.rglob('*') if included(p))
hashes={}
for path in files:
    digest=hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda:f.read(8*1024*1024),b''):digest.update(data)
    hashes[str(path.relative_to(root))]=digest.hexdigest()
(root/'BUNDLE_SHA256.json').write_text(json.dumps(hashes,indent=2,sort_keys=True))
destination=root.parent/(root.name+'.zip')
with ZipFile(destination,'w',compression=ZIP_DEFLATED,compresslevel=6) as z:
    for path in files+[root/'BUNDLE_SHA256.json']:z.write(path,arcname=str(Path(root.name)/path.relative_to(root)))
print(f'{destination}\n{len(files)+1} files; {destination.stat().st_size/2**20:.2f} MiB')

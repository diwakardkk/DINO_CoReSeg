import hashlib
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'BUNDLE_SHA256.json').read_text())
for relative,expected in manifest.items():
    path=root/relative;h=hashlib.sha256()
    if not path.is_file():raise SystemExit(f'Missing bundled file: {relative}')
    with path.open('rb') as f:
        for data in iter(lambda:f.read(8*1024*1024),b''):h.update(data)
    if h.hexdigest()!=expected:raise SystemExit(f'Bundle file differs from packaged version: {relative}')
print(f'Verified {len(manifest)} bundled files. Dataset images are external and audited by the pipeline.')

"""Verify extracted portable V2 code against its packaged SHA-256 manifest."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if __name__ == '__main__':
    manifest = json.loads((ROOT / 'BUNDLE_SHA256.json').read_text())
    required = {'run_v2.sh', 'install_v2.sh', 'README_V2.md', 'requirements.txt',
                'configs/v2_resolution_448.yaml', 'configs/v2_boundary_refinement.yaml',
                'configs/v2_multilayer_dino.yaml', 'src/v2_pipeline.py',
                'src/plot_paper.py', 'tests/test_v2.py'}
    if not required.issubset(manifest):
        raise SystemExit(f'Missing required bundle entries: {required - manifest.keys()}')
    for relative, expected in manifest.items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise SystemExit(f'Invalid or missing bundled file: {relative}')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise SystemExit(f'Bundle file differs from packaged version: {relative}')
    print(f'Verified {len(manifest)} V2 source files. Datasets, weights and runs are separate.')

"""Build an independent V2 code bundle; datasets and run artifacts stay external."""
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {'.venv', '.git', '__pycache__', '.pytest_cache', 'outputs',
            'cache_v2', '.mplcache', '.cache', '.agents', '.codex'}
EXCLUDED_FILES = {'BUNDLE_SHA256.json', 'requirements.server.lock.txt', '.DS_Store'}


def included(path):
    relative = path.relative_to(ROOT)
    return (path.is_file() and not any(part in EXCLUDED for part in relative.parts)
            and path.name not in EXCLUDED_FILES
            and not path.name.startswith('.env')
            and path.suffix not in {'.pyc', '.tmp', '.pth', '.pt', '.zip', '.pem', '.key'})


if __name__ == '__main__':
    files = sorted(path for path in ROOT.rglob('*') if included(path))
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in files}
    manifest = ROOT / 'BUNDLE_SHA256.json'
    manifest.write_text(json.dumps(hashes, indent=2, sort_keys=True) + '\n')
    destination = ROOT / 'DINO_CoReSeg_V2.zip'
    with ZipFile(destination, 'w', ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files + [manifest]:
            archive.write(path, arcname=str(Path(ROOT.name) / path.relative_to(ROOT)))
    with ZipFile(destination) as archive:
        bad = archive.testzip()
        if bad:
            raise RuntimeError(f'Corrupt ZIP member: {bad}')
    print(f'{destination}\n{len(files) + 1} files; {destination.stat().st_size / 2**20:.2f} MiB')

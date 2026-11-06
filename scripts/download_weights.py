"""Fetch official model weights once; keep them outside Git and verify SHA-256."""
import hashlib
import os
from pathlib import Path
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth'
SHA256 = 'b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9'


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    destination = ROOT / 'vendor/dinov2_vits14_pretrain.pth'
    if destination.exists():
        if digest(destination) != SHA256:
            raise SystemExit('Existing DINO weights have an unexpected SHA-256; inspect the file before replacing it.')
        print('Official DINO weights verified; download unnecessary.')
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix='.tmp', delete=False) as target:
            temporary = Path(target.name)
            with urllib.request.urlopen(URL, timeout=120) as source:
                while block := source.read(8 * 1024 * 1024):
                    target.write(block)
            target.flush()
            os.fsync(target.fileno())
        if digest(temporary) != SHA256:
            raise RuntimeError('Downloaded DINO weights failed SHA-256 verification.')
        os.replace(temporary, destination)
        print('Official DINO weights downloaded and verified.')
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


if __name__ == '__main__':
    main()

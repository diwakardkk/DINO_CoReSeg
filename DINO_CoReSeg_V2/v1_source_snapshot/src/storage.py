from pathlib import Path
import shutil
import uuid
from zipfile import ZipFile
from .common import RESEARCH,file_hash,read_json,atomic_json,STOP


def safe_extract(archive,destination,strip_prefix=None):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    with ZipFile(archive) as z:
        for info in z.infolist():
            STOP.check()
            path=Path(info.filename)
            if path.is_absolute() or '..' in path.parts:raise ValueError(f'Unsafe ZIP path: {path}')
            if ((info.external_attr>>16)&0o170000)==0o120000:raise ValueError('ZIP symbolic links are not allowed')
            if strip_prefix:
                try:path=path.relative_to(strip_prefix)
                except ValueError:continue
            if not path.parts:continue
            target=destination/path
            if info.is_dir():target.mkdir(parents=True,exist_ok=True)
            else:
                target.parent.mkdir(parents=True,exist_ok=True)
                with z.open(info) as source,target.open('wb') as out:shutil.copyfileobj(source,out)


def prepare_data(c):
    root=Path(c['root'])
    for rel in ['dataset-1/kvasir-seg.zip','dataset-1/kvasir-sessile.zip']:
        if not (root/rel).is_file():raise FileNotFoundError(f'Missing dataset archive: {root/rel}')
    for rel,archive,prefix in [('dataset-2/Dataset','dataset-2/Dataset.zip','Dataset'),
                               ('dataset-3/PolypDataset','dataset-3/PolypDataset.zip',None)]:
        destination=root/rel
        if destination.is_dir():continue
        if not (root/archive).is_file():raise FileNotFoundError(f'Need extracted {rel} or archive {archive}')
        staging=destination.parent/(destination.name+'.extracting-'+uuid.uuid4().hex)
        staging.mkdir(parents=True)
        try:safe_extract(root/archive,staging,prefix);staging.rename(destination)
        except BaseException:
            shutil.rmtree(staging,ignore_errors=True);raise
    # Establish an immutable local archive baseline on first execution. On resume,
    # reject changed archives. No dataset hashes are shipped in the repository.
    verified=[{'archive':rel,'sha256':file_hash(root/rel)} for rel in
              ['dataset-1/kvasir-seg.zip','dataset-1/kvasir-sessile.zip']]
    atomic_json(RESEARCH/'manifests/dataset1_archive_baseline.json',verified,immutable=True)
    atomic_json(RESEARCH/'manifests/storage_verification.json',{'dataset1':verified,'data_root':str(root)})

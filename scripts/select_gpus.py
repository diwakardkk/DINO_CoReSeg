"""Choose a stable homogeneous RTX 3090 mask before importing PyTorch."""
import csv
import fcntl
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def parse_inventory(text):
    return [dict(index=int(row[0]), uuid=row[1].strip(), name=row[2].strip(),
                 free_mib=int(row[3]), total_mib=int(row[4]), utilization=int(row[5]))
            for row in csv.reader(io.StringIO(text)) if row]


def choose_3090(gpus):
    selected = [gpu for gpu in gpus if gpu['name'] == 'NVIDIA GeForce RTX 3090']
    if not selected:
        raise RuntimeError('No RTX 3090 found. Set CUDA_VISIBLE_DEVICES to the GPU allocation you want to use.')
    # DataParallel gathers results on device zero; give that GPU most headroom.
    primary = max(selected, key=lambda gpu: (gpu['free_mib'], -gpu['index']))
    return [primary] + sorted((gpu for gpu in selected if gpu is not primary), key=lambda gpu: gpu['index'])


def main():
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        mask = os.environ['CUDA_VISIBLE_DEVICES'].strip()
        if not mask or mask == '-1':
            raise RuntimeError('CUDA_VISIBLE_DEVICES disables CUDA. Select at least one allocated GPU.')
        print('Using your existing CUDA_VISIBLE_DEVICES allocation unchanged.', file=sys.stderr)
        print(mask)
        return
    if any(os.environ.get(key) for key in ['SLURM_JOB_ID', 'PBS_JOBID', 'LSB_JOBID']):
        raise RuntimeError('A scheduler job was detected without CUDA_VISIBLE_DEVICES. Set the GPU mask assigned to your job.')
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name,memory.free,memory.total,utilization.gpu',
                             '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True)
    gpus = parse_inventory(result.stdout)
    directory = ROOT / 'outputs/launcher'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'rtx3090_selection.json'
    with (directory / '.gpu_selection.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            saved = json.loads(path.read_text())
            by_uuid = {gpu['uuid']: gpu for gpu in gpus}
            if not all(uuid in by_uuid and by_uuid[uuid]['name'] == 'NVIDIA GeForce RTX 3090' for uuid in saved['uuids']):
                raise RuntimeError('Saved RTX 3090 allocation changed. Preserve the GPU set/order when resuming.')
            selected = [by_uuid[uuid] for uuid in saved['uuids']]
        else:
            selected = choose_3090(gpus)
            with tempfile.NamedTemporaryFile('w', dir=directory, delete=False) as target:
                json.dump({'uuids': [gpu['uuid'] for gpu in selected], 'initial_inventory': selected}, target, indent=2)
                target.flush()
                os.fsync(target.fileno())
                temporary = target.name
            os.replace(temporary, path)
    print('Selected RTX 3090 physical indices (primary first): ' + ','.join(str(gpu['index']) for gpu in selected), file=sys.stderr)
    if any(gpu['utilization'] > 10 for gpu in selected):
        print('Selected GPUs already have workloads. Use only GPUs allocated to you; free memory does not mean a GPU is idle.', file=sys.stderr)
    print(','.join(gpu['uuid'] for gpu in selected))


if __name__ == '__main__':
    main()

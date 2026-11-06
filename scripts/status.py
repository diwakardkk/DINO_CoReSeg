import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
for path in sorted((root/'outputs').rglob('pipeline_status.json'))+sorted((root/'outputs').rglob('progress.json'))+sorted((root/'outputs').rglob('evaluation_progress.json')):
    try:print(str(path.relative_to(root))+': '+json.dumps(json.loads(path.read_text()),sort_keys=True))
    except (OSError,ValueError) as exc:print(f'{path}: {exc}')

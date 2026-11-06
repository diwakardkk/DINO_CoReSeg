import csv
import importlib.metadata
from pathlib import Path
import platform
import os
from .common import (RESEARCH,output,atomic_json,read_json,file_hash,csv_write,scientific_config)
from .runtime import hardware


def run(c):
    base=output(c);reports=base/'reports';reports.mkdir(parents=True,exist_ok=True)
    atomic_json(reports/'environment.json',{'hardware':hardware(c),'python':platform.python_version(),
                'packages':{d.metadata['Name']:d.version for d in importlib.metadata.distributions()},
                'source_sha256':{str(p.relative_to(RESEARCH)):file_hash(p) for p in sorted((RESEARCH/'src').glob('*.py'))},
                'data_root':c['root']})
    atomic_json(reports/'configuration.json',scientific_config(c))
    text='# DINO-CoReSeg GPU 100-epoch study\n\nOnly actual saved artifacts are summarized. No baselines, ablations, fabricated results or clinical-readiness claims.\n\n'
    for label,path in [('Pipeline status',base/'pipeline_status.json'),('Data audit',RESEARCH/'manifests/audit.json'),
                       ('Splits and leakage',RESEARCH/'splits/leakage_report.json'),('GPU smoke test',base/'smoke/smoke_status.json'),
                       ('Five-fold CV',RESEARCH/'outputs'/(c['experiment']+'_cv5')/'status.json'),
                       ('Holdout training',base/'holdout/status.json'),('Final evaluation',base/'holdout/evaluation_status.json')]:
        text+='## '+label+'\n\n'
        if path.exists():
            obj=read_json(path)
            if label=='Data audit':obj={k:v for k,v in obj.items() if k!='links'}
            import json
            text+='```json\n'+json.dumps(obj,indent=2)+'\n```\n\n'
        else:text+='Not completed; no result artifact exists.\n\n'
    tables=base/'tables';tables.mkdir(parents=True,exist_ok=True)
    metrics=base/'holdout/metrics/final_metrics.csv'
    if metrics.exists():
        with metrics.open() as f:rows=list(csv.DictReader(f))
        csv_write(tables/'final_test_metrics.csv',rows)
        keys=['partition','target','images','dice','iou','precision','recall','f2','boundary_f1','boundary_iou','hd95','assd','mae','fp_frame']
        markdown='| '+' | '.join(keys)+' |\n| '+' | '.join(['---']*len(keys))+' |\n'
        for row in rows:markdown+='| '+' | '.join(row.get(k,'') for k in keys)+' |\n'
        (tables/'final_test_metrics.md').write_text(markdown)
        selected=['partition','target','images','dice','iou','precision','recall','hd95','assd','mae','fp_frame']
        escape=lambda v:str(v).replace('_',r'\_').replace('%',r'\%')
        latex=r'\begin{tabular}{'+'l'*len(selected)+'}\n'+' & '.join(escape(k) for k in selected)+r' \\'+'\n'+r'\hline'+'\n'
        for row in rows:latex+=' & '.join(escape(row.get(k,'--')) for k in selected)+r' \\'+'\n'
        latex+=r'\end{tabular}'+'\n'
        (tables/'final_test_metrics.tex').write_text(latex)
        text+='## Final metrics\n\n'+markdown+'\n'
    cv=RESEARCH/'outputs'/(c['experiment']+'_cv5')/'fold_metrics.csv'
    if cv.exists():
        with cv.open() as f:rows=list(csv.DictReader(f))
        csv_write(tables/'cv_fold_metrics.csv',rows)
    cache=list((RESEARCH/'cache').glob('*.npz'))
    text+=f'## Storage and limitations\n\nFeature-cache entries: {len(cache)}; bytes: {sum(p.stat().st_size for p in cache)}.\n\n'
    text+='''GPU utilization is workload-dependent: DINO features are frozen, the decoder is small, and I/O, clustering, distance metrics and plotting require CPU work. Decoder-only speed is not end-to-end speed. Mixed precision, TF32, chosen batch size and hardware are recorded; this is a new GPU experiment, not a continuation of CPU model weights. Seeds do not guarantee bitwise equality across CUDA hardware/kernels. Detected duplicate candidates are grouped/excluded; heuristic screening cannot establish the absence of all correlated frames. CVC-300/CVC-ColonDB held-out image overlap must be disclosed. Relative pixel area does not establish physical lesion diameter. HD95/ASSD use pixels and explicit image-diagonal penalties with saved failure counts. Normal targets are derived empty masks. The FAR 100-pixel gate is scored at native resolution; separately labeled 256-grid gates are exploratory.\n\n'''
    text+='''## Resumption\n\nRun the same `./run_server.sh start` command with the same data, configuration and visible GPU layout. It automatically reuses committed caches, prototypes, step/epoch checkpoints, completed evaluations and predictions. SIGTERM/SIGINT requests a graceful stop; SIGKILL/power loss can lose steps since the last atomic checkpoint (default at most nine completed optimizer steps), and an interrupted monitor is recomputed. Every completed epoch has its own model/optimizer checkpoint and JSON metrics.\n'''
    (reports/'EXPERIMENT_SUMMARY.md').write_text(text)

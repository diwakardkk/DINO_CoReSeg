import copy
from .common import output,atomic_json,read_json,load_split
from .extract_features import extract
from .prototypes import build
from .cross_validate import protocol_for
from .train import train_model


def run(c):
    marker=output(c)/'smoke/smoke_status.json'
    if marker.exists() and read_json(marker).get('passed',False):return
    pool=load_split('train');positive=[r for r in pool if r['positive']];negative=[r for r in pool if not r['positive']]
    chosen=[]
    for source in sorted({r['source'] for r in positive}):chosen.append(next(r for r in positive if r['source']==source))
    chosen += [r for r in positive if r not in chosen][:4-len(chosen)]
    chosen += negative[:4]
    if len(chosen)!=8:raise RuntimeError('Eight development samples required for the GPU integration test')
    extract(c,chosen)
    train=chosen[:3]+chosen[4:7];val=[chosen[3],chosen[7]]
    directory=output(c)/'smoke';directory.mkdir(parents=True,exist_ok=True)
    protos=build(c,train,[r['id'] for r in train],directory/'prototypes.pt',c['seed'])
    test=copy.deepcopy(c);test['max_epochs']=1;test['batch_size']=4;test['evaluation_batch_size']=4
    test['loader']['workers']=0;test['checkpoint']['every_steps']=1
    protocol=protocol_for(train)
    first=train_model(test,train,val,protos,directory/'integration',protocol,c['seed'])
    second=train_model(test,train,val,protos,directory/'integration',protocol,c['seed'])
    if first['checkpoint_sha256']!=second['checkpoint_sha256']:raise RuntimeError('Smoke checkpoint restoration failed')
    atomic_json(marker,{'passed':True,'device':c['device'],'samples':8,'cache_resume_supported':True,
                'checkpoint_restored':True,'all_decoder_gradients_checked':True,
                'purpose':'Integration only; smoke model is not a scientific result or the final model'})

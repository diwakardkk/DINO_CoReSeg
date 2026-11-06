"""Local CPU logic tests. These do not claim that CUDA execution was validated."""
import copy
import json
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import pytest
import torch
import yaml

from src.common import RESEARCH,STOP,StopRequested,atomic_json,atomic_torch,read_json,scientific_config
from src.model_coreseg import DINOCoReSeg
from src.losses import objective
from src.metrics import positive_metrics,normal_metrics
from src.cv_metrics import PixelROC,Monitor,bounded_loss
from src.runtime import parallelize,unwrap,amp_dtype
from src.storage import safe_extract
from src.cv_splits import fold_definitions
from src.train import BalancedBatches,train_model


@pytest.fixture(autouse=True)
def reset_stop():
    STOP.requested=False
    yield
    STOP.requested=False


@pytest.fixture
def cpu_config():
    c=yaml.safe_load((RESEARCH/'configs/gpu_100.yaml').read_text())
    c.update(device='cpu',root='/unit-fixture',research=str(RESEARCH),batch_size=4,evaluation_batch_size=4,max_epochs=2)
    c['gpu']['amp']=False;c['loader']['workers']=0;c['threads']=2;c['checkpoint']['every_steps']=1
    torch.set_num_threads(2)
    return c


def test_default_full_hundred_epochs():
    c=yaml.safe_load((RESEARCH/'configs/gpu_100.yaml').read_text())
    assert c['max_epochs']==100 and not c['early_stopping']
    assert c['cross_validation']['enabled'] and c['cross_validation']['folds']==5


def test_portable_configuration_identity(cpu_config):
    second=copy.deepcopy(cpu_config);second['root']='/a/new/server';second['research']='/different/package';second['loader']['workers']=8
    assert scientific_config(cpu_config)==scientific_config(second)


def test_no_cuda_amp_on_cpu(cpu_config):
    assert amp_dtype(cpu_config) is None
    raw=DINOCoReSeg(torch.randn(8,384));wrapped=parallelize(raw,cpu_config)
    assert unwrap(wrapped) is raw


def test_model_full_shapes_and_gradients(cpu_config):
    model=DINOCoReSeg(torch.randn(8,384));out=model(torch.randn(2,384,25,25),torch.rand(2,3,100,100))
    truth=torch.zeros(2,1,350,350);truth[0,:,120:170,110:160]=1
    loss,parts=objective(out,truth,torch.tensor([True,False]),cpu_config,.05);loss.backward()
    assert out['final'].shape==(2,1,350,350)
    assert out['initial'].shape==out['fused'].shape==out['counter_evidence'].shape==(2,1,25,25)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(torch.isfinite(v) for v in parts.values())


def test_one_bit_mask_conversion():
    from PIL import Image
    mask=np.zeros((10,10),bool);mask[2:4,3:6]=True
    assert np.array_equal(np.asarray(Image.fromarray(mask).convert('L'))>=128,mask)


def test_atomic_checkpoint_replacement(tmp_path):
    path=tmp_path/'last.pt';atomic_torch(path,{'step':1});atomic_torch(path,{'step':2})
    assert torch.load(path,weights_only=True)['step']==2


def test_atomic_manifest_immutability(tmp_path):
    path=tmp_path/'split.json';atomic_json(path,{'a':1},immutable=True)
    with pytest.raises(RuntimeError):atomic_json(path,{'a':2},immutable=True)


def test_zip_slip_rejected(tmp_path):
    path=tmp_path/'bad.zip'
    with ZipFile(path,'w') as z:z.writestr('../outside.txt','bad')
    with pytest.raises(ValueError,match='Unsafe'):safe_extract(path,tmp_path/'safe')
    assert not (tmp_path/'outside.txt').exists()


def test_zip_root_prefix_extraction(tmp_path):
    path=tmp_path/'data.zip'
    with ZipFile(path,'w') as z:z.writestr('Dataset/Patient1/a.txt','a')
    safe_extract(path,tmp_path/'out','Dataset')
    assert (tmp_path/'out/Patient1/a.txt').read_text()=='a'


def test_empty_prediction_failure_is_retained():
    truth=np.zeros((20,30),bool);truth[4:8,5:10]=1
    metrics=positive_metrics(np.zeros_like(truth,float),truth)
    assert metrics['dice']==0 and metrics['surface_failure']==1
    assert metrics['hd95']==pytest.approx(np.hypot(20,30))


def test_normal_gates_and_nan_rejection():
    probability=np.zeros((30,30));probability[2:12,2:12]=1
    m=normal_metrics(probability,gates=[1,100,101])
    assert m['fp_frame']==1 and m['fp_frame_gate_100']==1 and m['fp_frame_gate_101']==0
    with pytest.raises(ValueError):normal_metrics(np.full((3,3),np.nan))


def test_quantized_roc_and_loss_display():
    roc=PixelROC(65536);roc.update(np.array([.1,.8,.2,.9]),np.array([0,1,0,1]))
    assert roc.curve()[2]==1 and bounded_loss(3)==.75


def test_monitor_never_applies_dice_to_normals(cpu_config):
    mask=torch.zeros(2,1,350,350);mask[0,:,100:150,100:150]=1
    monitor=Monitor(cpu_config,.05);monitor.update(torch.zeros_like(mask),mask,torch.tensor([True,False]))
    summary=monitor.summary()
    assert summary['normal_loss']==pytest.approx(np.log(2),rel=1e-6)
    assert 0<=summary['accuracy']<=1 and 0<=summary['loss_0_1']<1


def test_resumable_balanced_sampler():
    rows=[{'positive':i<8,'source':'A' if i%2 else 'B'} for i in range(16)]
    full=list(BalancedBatches(rows,4,42,3,0));tail=list(BalancedBatches(rows,4,42,3,2))
    assert tail==full[2:]
    assert all(sum(rows[i]['positive'] for i in batch)==2 for batch in full)


def test_nested_folds_keep_patients_and_duplicates_together():
    rows=[]
    for patient in range(60):
        for positive in [True,False]:
            rows.append({'id':f'{patient}/{positive}','patient':f'P{patient}','duplicate_component':f'group{patient//2}',
                         'positive':positive,'source':'FAR','foreground_fraction':.05 if positive else 0.})
    folds=fold_definitions(rows,5,5,42);lookup={r['id']:r for r in rows}
    assert sorted(k for f in folds for k in f['heldout_ids'])==sorted(lookup)
    for fold in folds:
        parts=[{lookup[k]['duplicate_component'] for k in fold[field]} for field in ['train_ids','inner_val_ids','heldout_ids']]
        assert not(parts[0]&parts[1] or parts[0]&parts[2] or parts[1]&parts[2])


def _unit_caches(tmp_path,monkeypatch):
    import src.extract_features as features
    import src.train as training
    monkeypatch.setattr(features,'RESEARCH',tmp_path);monkeypatch.setattr(training,'RESEARCH',tmp_path)
    monkeypatch.setattr(training,'implementation_identity',lambda:{'fixture':'unit-test-only'})
    identity={'name':'synthetic-unit-fixture','parameters':0}
    atomic_json(tmp_path/'cache/backbone_identity.json',identity)
    rng=np.random.default_rng(5);rows=[]
    for i in range(12):
        split='train' if i<8 else 'val';positive=i%2==0
        row={'id':f'unit/{i}','split':split,'positive':positive,'source':'fixture','patient':f'P{i}',
             'domain':'FAR','collection':'unit','mask_kind':'supplied' if positive else 'derived_empty',
             'image':f'images/{i}.png','mask':f'masks/{i}.png' if positive else None,'archive':None}
        mask=np.zeros((1,350,350),np.uint8)
        if positive:mask[:,120:160,130:170]=1
        metadata={'row':row,'model':identity}
        with features.cache_path(row).open('wb') as f:
            np.savez(f,features=rng.standard_normal((384,25,25)).astype(np.float16),
                     rgb=rng.integers(0,256,(3,100,100),dtype=np.uint8),mask=mask,metadata=np.array(json.dumps(metadata)))
        rows.append(row)
    return rows[:8],rows[8:]


def test_real_engine_step_resume_matches_uninterrupted_training(tmp_path,monkeypatch,cpu_config):
    import src.train as training
    train,val=_unit_caches(tmp_path,monkeypatch)
    torch.manual_seed(7);prototypes=torch.randn(8,384);protocol={'positive_area_median':.05}
    full=tmp_path/'full';train_model(cpu_config,train,val,prototypes,full,protocol,42)
    original=training.append_jsonl
    def interrupt_after_first_step(path,obj):
        original(path,obj)
        if Path(path).name=='steps.jsonl' and obj['global_step']==1:STOP.requested=True
    monkeypatch.setattr(training,'append_jsonl',interrupt_after_first_step)
    interrupted=tmp_path/'interrupted'
    with pytest.raises(StopRequested):train_model(cpu_config,train,val,prototypes,interrupted,protocol,42)
    partial=torch.load(interrupted/'last.pt',map_location='cpu',weights_only=False)
    assert partial['next_step']==1 and not partial['epoch_complete']
    STOP.requested=False;monkeypatch.setattr(training,'append_jsonl',original)
    train_model(cpu_config,train,val,prototypes,interrupted,protocol,42)
    a=torch.load(full/'last.pt',map_location='cpu',weights_only=False)
    b=torch.load(interrupted/'last.pt',map_location='cpu',weights_only=False)
    assert a['global_step']==b['global_step']==4
    assert all(torch.equal(a['model'][k],b['model'][k]) for k in a['model'])
    assert len(b['history'])==2 and b['epoch_complete']
    assert (interrupted/'checkpoints/epoch_001.pt').exists() and (interrupted/'checkpoints/epoch_002.pt').exists()
    assert len(list((interrupted/'epochs').glob('*.json')))==2


def test_native_evaluation_and_report_are_resumable(tmp_path,monkeypatch,cpu_config):
    from PIL import Image
    import src.common as common
    import src.report as reporting
    import src.evaluate as evaluation
    from src.evaluate import evaluate
    train,val=_unit_caches(tmp_path,monkeypatch)
    monkeypatch.setattr(common,'RESEARCH',tmp_path);monkeypatch.setattr(reporting,'RESEARCH',tmp_path);monkeypatch.setattr(evaluation,'RESEARCH',tmp_path)
    c=copy.deepcopy(cpu_config);c['root']=str(tmp_path);c['experiment']='unit-artifacts';c['evaluation']['metric_workers']=2
    directory=tmp_path/'outputs/unit-artifacts/holdout'
    for i,row in enumerate(val):
        image=np.full((40,50,3),60+i*20,np.uint8)
        (tmp_path/'images').mkdir(exist_ok=True);Image.fromarray(image).save(tmp_path/row['image'])
        if row['positive']:
            (tmp_path/'masks').mkdir(exist_ok=True);mask=np.zeros((40,50),bool);mask[12:22,15:25]=1
            Image.fromarray(mask).save(tmp_path/row['mask'])
    torch.manual_seed(17);p=torch.randn(8,384)
    protocol={'positive_area_median':.05,'small_polyp_fraction_max':.03,'medium_polyp_fraction_max':.1}
    train_model(c,train,val,p,directory,protocol,42)
    status=evaluate(c,val,directory,protocol)
    assert status['completed'] and status['images']==4
    files=list((directory/'predictions/FAR').glob('*.json'));assert len(files)==4
    for path in files:
        with np.load(path.with_suffix('.npz')) as z:prob=z['probability']
        mask=np.asarray(Image.open(path.with_suffix('.png')))
        assert mask.shape==prob.shape==(40,50)
        assert np.array_equal(mask>0,prob>=.5)
    assert evaluate(c,val,directory,protocol)==status
    reporting.run(c)
    table=tmp_path/'outputs/unit-artifacts/tables/final_test_metrics.tex'
    assert table.read_text().startswith(r'\begin{tabular}')
    assert (directory/'metrics/surface_failures.csv').exists()

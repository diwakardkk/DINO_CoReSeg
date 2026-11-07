"""CPU fixtures and real offline DINO shape tests; not experiment results."""
import copy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import os
import shutil
import numpy as np
from PIL import Image
import pytest
import torch
from src.common import RESEARCH,STOP,StopRequested,atomic_json,file_hash
from src.v2_config import load,output,cache_root
from src.v2_model import DINOCoReSegV2
from src.v2_losses import occupancy,background_eligible,boundary_loss,objective
from src.v2_metrics import detection
from src.v2_data import verify_disjoint,used_counts
from src.v2_features import SpatialDinoV2,Prepared,CachedDataset,cache_path,load_backbone
from src.v2_prototypes import build
from src.v2_train import train
from src.v2_reporting import selected_record


@pytest.fixture(autouse=True)
def reset_stop():
    STOP.requested=False
    yield
    STOP.requested=False


def configuration(name='v2_resolution_448'):
    c=load(RESEARCH/'configs'/f'{name}.yaml',allow_cpu=True);c['device']='cpu';c['gpu']['amp']=False;c['loader']['workers']=0
    torch.set_num_threads(2)
    return c


@pytest.mark.parametrize('name',['v2_resolution_448','v2_rgb_path','v2_boundary_refinement','v2_boundary_loss','v2_multilayer_dino'])
def test_full_448_model_shapes_and_finite_gradients(name):
    c=configuration(name);model=DINOCoReSegV2(c,torch.randn(8,384))
    layers=len(c['intermediate_layers']) if c['variant']=='multilayer' else 1
    features=torch.randn(2,layers,384,32,32);rgb=torch.rand(2,3,224,224)
    target=torch.zeros(2,1,448,448);target[0,:,200:215,195:220]=1
    out=model(features,rgb);loss,parts=objective(out,target,torch.tensor([True,False]),c,.05);loss.backward()
    assert out['final'].shape==(2,1,448,448)
    assert out['initial'].shape==out['fused'].shape==out['counter_evidence'].shape==(2,1,32,32)
    assert out['refined'].shape==(2,1,224,224)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(torch.isfinite(value) for value in parts.values())


def test_resolution_version_preserves_v1_parameterization_and_native_behavior():
    c=configuration();c.update(image_size=350,grid_size=25,rgb_size=100)
    spec=importlib.util.spec_from_file_location('snapshot_model',RESEARCH/'v1_source_snapshot/src/model_coreseg.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    proto=torch.randn(8,384);old=module.DINOCoReSeg(proto);new=DINOCoReSegV2(c,proto)
    assert sum(p.numel() for p in new.parameters())==85287
    new.load_state_dict(old.state_dict(),strict=True)
    features=torch.randn(1,384,25,25);rgb=torch.rand(1,3,100,100)
    with torch.inference_mode():a=old(features,rgb);b=new(features,rgb)
    assert all(torch.equal(a[key],b[key]) for key in a)


@pytest.mark.parametrize('kind',['healthy','small','patch_boundary','large','thin'])
def test_exact_patch_occupancy_and_background_exclusion(kind):
    c=configuration();mask=torch.zeros(1,1,448,448)
    if kind=='small':mask[:,:,101:104,202:205]=1
    if kind=='patch_boundary':mask[:,:,13:16,13:16]=1
    if kind=='large':mask[:,:,30:400,40:390]=1
    if kind=='thin':mask[:,:,40:400,111:112]=1
    expected=mask.unfold(2,14,14).unfold(3,14,14).mean((-1,-2))
    actual=occupancy(mask,14,32)
    assert actual.shape==(1,1,32,32) and torch.equal(actual,expected)
    eligible=background_eligible(mask,c)
    assert not eligible[actual>c['background_occupancy']].any()
    if kind=='healthy':assert eligible.all()
    else:assert not eligible.all()


def test_token_positions_are_not_transposed():
    c=configuration()
    class Tokens(torch.nn.Module):
        def forward_features(self,x):return {'x_norm_patchtokens':torch.arange(1024)[None,:,None].expand(len(x),1024,384).float()}
    out=SpatialDinoV2(Tokens(),c)(torch.zeros(1,3,448,448))
    assert out[0,0,0,3,7]==3*32+7


@pytest.mark.parametrize('positive',[False,True])
def test_boundary_loss_is_stable_for_empty_targets(positive):
    c=configuration();logits=torch.zeros(2,1,56,56,requires_grad=True);target=torch.zeros_like(logits)
    loss=boundary_loss(logits,target,torch.tensor([positive,False]),c);loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    if not positive:assert loss==0


def test_lesion_detection_is_distinct_from_pixel_recall_and_empty_is_defined():
    truth=np.zeros((20,30),bool);truth[2:5,3:6]=1;truth[13:17,20:24]=1
    pred=np.zeros_like(truth,float);pred[2:5,3:6]=1
    assert detection(pred,truth)['lesion_detection_recall']==.5
    assert detection(np.zeros_like(pred),truth)['lesion_detection_recall']==0
    assert detection(pred,np.zeros_like(truth))['lesion_detection_recall'] is None


def row(sid,patient='',collection='train_base',positive=True):
    return {'id':sid,'patient':patient,'collection':collection,'domain':'FAR' if patient else 'benchmark',
            'duplicate_component':sid,'image_sha256':sid,'positive':positive,'mask':sid+'.png' if positive else None,
            'mask_kind':'supplied' if positive else 'derived_empty'}


def test_dataset3_usage_comes_from_actual_membership_and_tests_are_reserved():
    splits={'train':[row('a')],'val':[row('b')],'test':[row('c',collection='test_Kvasir')]}
    assert used_counts(splits,lambda r:r['collection']=='train_base')=={'train':1,'val':1,'test':0}
    verify_disjoint(splits)
    splits['train'][0]['collection']='test_Kvasir'
    with pytest.raises(ValueError,match='test sample'):verify_disjoint(splits)


def test_far_patient_disjoint_and_duplicate_exclusion():
    splits={'train':[row('a','P1','far')],'val':[row('b','P2','far')],'test':[row('c',collection='test_Kvasir')]}
    verify_disjoint(splits)
    bad=copy.deepcopy(splits);bad['val'][0]['patient']='P1'
    with pytest.raises(ValueError,match='patient leakage'):verify_disjoint(bad)
    bad=copy.deepcopy(splits);bad['val'][0]['duplicate_component']='a'
    with pytest.raises(ValueError,match='component leakage'):verify_disjoint(bad)
    hashes=[{'id':'a','normalized_sha256':'same'},{'id':'c','normalized_sha256':'same'}]
    with pytest.raises(ValueError,match='image leakage'):verify_disjoint(splits,hashes)


def test_augmented_material_cannot_enter_v2():
    splits={'train':[row('a',collection='NewTRimageDA')],'val':[row('b')],'test':[row('c',collection='test_Kvasir')]}
    with pytest.raises(ValueError,match='Augmented'):verify_disjoint(splits)


def unit_caches(tmp_path):
    c=configuration();c.update(research=str(tmp_path/'v2'),v1_root=str(tmp_path/'v1'),root=str(tmp_path/'data'),
                               image_size=56,rgb_size=28,selection_size=56,grid_size=4,batch_size=2,evaluation_batch_size=2,
                               max_epochs=2,lr_schedule='cosine',gradient_accumulation=2)
    c['checkpoint']['every_steps']=1
    folder=cache_root(c);folder.mkdir(parents=True);identity={'fixture':'unit-only','parameters':0}
    atomic_json(folder/'identity.json',identity);rows=[];rng=np.random.default_rng(8)
    for i in range(12):
        root=Path(c['root']);(root/'images').mkdir(parents=True,exist_ok=True);(root/'masks').mkdir(exist_ok=True)
        image=root/'images'/f'{i}.png';Image.fromarray(np.full((20,30,3),90+i,np.uint8)).save(image)
        positive=i%2==0;mask=root/'masks'/f'{i}.png';truth=np.zeros((20,30),bool)
        if positive:truth[6:12,10:18]=1;Image.fromarray(truth).save(mask)
        r={'id':f'fixture/{i}','image':f'images/{i}.png','mask':f'masks/{i}.png' if positive else None,
           'image_sha256':file_hash(image),'mask_sha256':file_hash(mask) if positive else None,'patient':f'P{i}',
           'source':'FAR','domain':'FAR','collection':'far','duplicate_component':f'P{i}',
           'mask_kind':'supplied' if positive else 'derived_empty','positive':positive,'archive':None,
           'foreground_fraction':float(truth.mean()),'split':'train' if i<8 else 'val'}
        item=Prepared(c,[r])[0]
        with cache_path(c,r).open('wb') as f:
            np.savez(f,features=rng.standard_normal((1,384,4,4)).astype(np.float16),
                     rgb=item['rgb'].numpy(),mask=item['mask'].numpy(),selection_mask=item['selection_mask'].numpy(),
                     metadata=np.array(json.dumps({'identity':identity,'row':r})))
        rows.append(r)
    return c,rows[:8],rows[8:]


def test_prototype_training_identity_and_normalized_vectors(tmp_path):
    c,training,validation=unit_caches(tmp_path);path=output(c)/'checkpoints/prototypes.pt'
    p=build(c,training,[r['id'] for r in training],path)
    assert p.shape==(8,384) and torch.allclose(p.norm(dim=1),torch.ones(8),atol=1e-5)
    assert torch.equal(p,build(c,training,[r['id'] for r in training],path))
    with pytest.raises(ValueError,match='membership'):build(c,training,[r['id'] for r in training+validation],path)
    other=copy.deepcopy(c);other['grid_size']=8
    with pytest.raises(ValueError,match='reuse refused'):build(other,training,[r['id'] for r in training],path)


def test_exact_optimizer_boundary_resume_preserves_v1(tmp_path,monkeypatch):
    import src.v2_train as engine
    c,training,validation=unit_caches(tmp_path);torch.manual_seed(9);proto=torch.nn.functional.normalize(torch.randn(8,384),dim=1)
    protocol={'positive_area_median':.08,'small_polyp_fraction_max':.1,'medium_polyp_fraction_max':.2}
    v1=Path(c['v1_root']);v1.mkdir();sentinel=v1/'FROZEN.json';sentinel.write_text('V1 untouched');before=file_hash(sentinel)
    full=output(c)/'full';train(c,training,validation,proto,protocol,full)
    original=engine.atomic_json
    def stop_after_step(path,record,**kwargs):
        original(path,record,**kwargs)
        if Path(path).name=='progress.json' and record.get('global_step')==1:STOP.requested=True
    monkeypatch.setattr(engine,'atomic_json',stop_after_step);interrupted=output(c)/'interrupted'
    with pytest.raises(StopRequested):train(c,training,validation,proto,protocol,interrupted)
    checkpoint=torch.load(interrupted/'checkpoints/last.pt',weights_only=False)
    assert checkpoint['next_microbatch']==2 and not checkpoint['epoch_complete']
    STOP.requested=False;monkeypatch.setattr(engine,'atomic_json',original)
    train(c,training,validation,proto,protocol,interrupted)
    a=torch.load(full/'checkpoints/last.pt',weights_only=False);b=torch.load(interrupted/'checkpoints/last.pt',weights_only=False)
    assert a['global_step']==b['global_step'] and all(torch.equal(a['model'][key],b['model'][key]) for key in a['model'])
    assert a['scheduler']==b['scheduler'] and file_hash(sentinel)==before
    assert len(list((interrupted/'checkpoints').glob('*.pt')))==2
    history=list(csv.DictReader((interrupted/'logs/training_history.csv').open()))
    assert len(history)==2 and 'training_boundary_loss' in history[0] and 'validation_small_polyp_recall' in history[0]
    required={'epoch','learning_rate','epoch_seconds','GPU_peak_memory','checkpoint_selection_score','checkpoint_constraint_satisfied'}
    for split in ['training','validation']:
        required.update(split+'_'+key for key in ['total_loss','segmentation_loss','positive_loss','negative_loss',
                'positive_dice','positive_iou','positive_recall','small_polyp_dice','small_polyp_recall',
                'normal_fpr','positive_pixel_accuracy','empty_prediction_rate','lesion_detection_recall'])
    required.update('training_'+key for key in ['contrastive_loss','auxiliary_loss','boundary_loss'])
    assert required<=set(history[0]) and history[0]['GPU_peak_memory']==''
    wrong=copy.deepcopy(c);wrong['learning_rate']*=2
    with pytest.raises(ValueError,match='resume refused'):train(wrong,training,validation,proto,protocol,interrupted)


def test_selected_epoch_report_uses_matching_record_not_last():
    state={'epoch':59,'history':[{'epoch':60,'val_positive_dice':.83},{'epoch':100,'val_positive_dice':.77}]}
    assert selected_record(state,{'selected_epoch':60})['val_positive_dice']==.83
    with pytest.raises(ValueError):selected_record(state,{'selected_epoch':100})


def test_corrected_v1_report_leaves_historical_csv_intact(tmp_path):
    from src.v2_reporting import corrected_v1_report
    c=configuration();c.update(research=str(tmp_path/'v2'),v1_root=str(tmp_path/'v1'))
    folder=Path(c['v1_root'])/'outputs/dino_coreseg_gpu_100/holdout';(folder/'metrics').mkdir(parents=True)
    checkpoint=folder/'best.pt'
    torch.save({'epoch':59,'history':[{'epoch':60,'val_positive_dice':.83},{'epoch':100,'val_positive_dice':.7}]},checkpoint)
    atomic_json(folder/'FROZEN.json',{'selected_epoch':60,'checkpoint_sha256':file_hash(checkpoint)})
    (folder/'metrics/final_metrics.csv').write_text('partition,target,dice\nKvasir,positive,0.6\n')
    old=folder/'metrics/domain_gaps.csv';old.write_text('historical artifact');before=file_hash(old)
    corrected_v1_report(c)
    path=output(c)/'reports/v1_reporting_review/domain_gaps_selected_epoch.csv'
    record=list(csv.DictReader(path.open()))[0]
    assert float(record['selected_validation_dice'])==.83 and file_hash(old)==before


def test_early_stopping_analysis_does_not_invent_future_epochs():
    from src.v2_reporting import early_stopping_analysis
    history=[{'epoch':i,'validation_positive_dice':.8,'validation_normal_fpr':.05} for i in range(1,7)]
    result=early_stopping_analysis(history,3,.1)
    assert result['would_stop_epoch']==4 and result['selected_epoch_at_stop_or_last_observation']==1
    assert early_stopping_analysis(history[:2],3,.1)['would_stop_epoch'] is None


def test_cache_namespaces_share_only_identical_preparation():
    first=configuration();second=configuration('v2_boundary_loss');third=configuration('v2_multilayer_dino')
    assert cache_root(first)==cache_root(second)
    assert cache_root(first)!=cache_root(third)


def test_native_evaluation_resume_probabilities_and_error_alignment(tmp_path):
    from src.v2_evaluate import evaluate
    from src.plot_paper import aligned_images
    c,training,validation=unit_caches(tmp_path);c['gradient_accumulation']=1;c['evaluation']['bootstrap_replicates']=10
    proto=torch.nn.functional.normalize(torch.randn(8,384),dim=1)
    protocol={'positive_area_median':.08,'small_polyp_fraction_max':.1,'medium_polyp_fraction_max':.2}
    train(c,training,validation,proto,protocol);status=evaluate(c,validation,protocol)
    assert status['completed'] and status['images']==4 and evaluate(c,validation,protocol)==status
    folder=output(c)/'predictions/FAR'
    for path in folder.glob('*.json'):
        metadata=json.loads(path.read_text());rgb,truth,pred=aligned_images(c['root'],metadata['row'],path.with_suffix('.png'))
        assert rgb.shape[:2]==truth.shape==pred.shape==(20,30)
        if path.with_suffix('.npz').exists():
            with np.load(path.with_suffix('.npz')) as z:assert np.array_equal(z['probability']>=.5,pred)
    assert (output(c)/'evaluation/metrics/valid_surface_only.csv').exists()


def test_interrupted_evaluation_resumes_without_double_counting(tmp_path,monkeypatch):
    import src.v2_evaluate as evaluation
    c,training,validation=unit_caches(tmp_path);c['evaluation']['bootstrap_replicates']=10
    proto=torch.nn.functional.normalize(torch.randn(8,384),dim=1)
    protocol={'positive_area_median':.08,'small_polyp_fraction_max':.1,'medium_polyp_fraction_max':.2}
    train(c,training,validation,proto,protocol)
    original=evaluation.atomic_torch
    def interrupt(path,state):
        original(path,state)
        if Path(path).name=='resume.pt' and state['next_batch']==1:STOP.requested=True
    monkeypatch.setattr(evaluation,'atomic_torch',interrupt)
    with pytest.raises(StopRequested):evaluation.evaluate(c,validation,protocol)
    partial=torch.load(output(c)/'evaluation/resume.pt',weights_only=False)
    assert len(partial['records'])==2
    STOP.requested=False;monkeypatch.setattr(evaluation,'atomic_torch',original)
    result=evaluation.evaluate(c,validation,protocol)
    records=list(csv.DictReader((output(c)/'evaluation/metrics/per_image.csv').open()))
    assert result['images']==4 and len(records)==len({r['id'] for r in records})==4


def test_cv_reuses_exact_frozen_v1_membership(tmp_path,monkeypatch):
    import src.v2_pipeline as pipeline
    import src.plot_paper as plotting
    from src.cv_splits import fold_definitions
    from src.common import csv_write
    c=configuration();c.update(research=str(tmp_path/'v2'),v1_root=str(tmp_path/'v1'),seed=123)
    rows=[]
    for patient in range(60):
        for positive in [True,False]:
            rows.append({'id':f'{patient}/{positive}','patient':f'P{patient}','duplicate_component':f'P{patient}',
                         'positive':positive,'source':'FAR','foreground_fraction':.05 if positive else 0.})
    rows=sorted(rows,key=lambda row:row['id']);folds=fold_definitions(rows,5,5,42)
    root=Path(c['v1_root'])/'outputs/dino_coreseg_gpu_100_cv5/splits'
    for fold in folds:atomic_json(root/f"fold_{fold['fold']:02d}.json",fold)
    captured=[]
    monkeypatch.setattr(pipeline,'build',lambda *a,**kw:torch.randn(8,384))
    def training(config,train_rows,val_rows,prototypes,protocol,directory):
        captured.append((sorted(r['id'] for r in train_rows),sorted(r['id'] for r in val_rows)))
    monkeypatch.setattr(pipeline,'train',training)
    def evaluate(config,heldout,protocol,directory,outer):
        path=directory/'evaluation/metrics';path.mkdir(parents=True,exist_ok=True)
        csv_write(path/'final_metrics.csv',[{'partition':'outer_heldout','target':'positive','dice':.5,'iou':.4}])
        np.savez(path/'outer_heldout_roc.npz',auc=.5,positive_hist=np.ones(65536,np.int64),negative_hist=np.ones(65536,np.int64))
    monkeypatch.setattr(pipeline,'evaluate',evaluate);monkeypatch.setattr(plotting,'generate',lambda *a,**kw:[])
    pipeline.cv(c,{'train':rows[:60],'val':rows[60:]})
    assert captured==[(fold['train_ids'],fold['inner_val_ids']) for fold in folds]


def test_plotting_csv_only_with_unequal_folds(tmp_path,monkeypatch):
    import src.plot_paper as plotting
    monkeypatch.setattr(plotting,'ROOT',tmp_path)
    for fold,count in [(1,3),(2,2)]:
        path=tmp_path/f'experiment/fold_{fold:02d}/logs/training_history.csv';path.parent.mkdir(parents=True)
        with path.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=['epoch','training_segmentation_loss','validation_segmentation_loss','training_positive_dice','validation_positive_dice','training_positive_pixel_accuracy','validation_positive_pixel_accuracy']);writer.writeheader()
            for epoch in range(1,count+1):writer.writerow({'epoch':epoch,'training_segmentation_loss':3/epoch,'validation_segmentation_loss':4/epoch,
                'training_positive_dice':.5,'validation_positive_dice':.4,'training_positive_pixel_accuracy':.7,'validation_positive_pixel_accuracy':.6})
    inspected=[];original=plotting.save
    def inspect(fig,path):
        assert fig.axes[0].get_ylim()[1]>1
        assert fig.axes[0].get_xlim()==fig.axes[1].get_xlim()==(0,3)
        means=[line for line in fig.axes[0].lines if line.get_linewidth()==1.5]
        assert all(list(line.get_xdata())==[1,2] for line in means)
        inspected.append(True);original(fig,path)
    monkeypatch.setattr(plotting,'save',inspect)
    made=plotting.generate(tmp_path/'experiment',tmp_path/'figures')
    assert made==['training_curves','training_curves_dice']
    assert (tmp_path/'figures/training_curves.pdf').exists()
    result=subprocess.run([sys.executable,'-c','import src.plot_paper,sys; assert "torch" not in sys.modules'],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert len(inspected)==2


def test_qualitative_shape_mismatch_is_rejected(tmp_path):
    from src.plot_paper import aligned_images
    Image.fromarray(np.zeros((10,20,3),np.uint8)).save(tmp_path/'image.png')
    Image.fromarray(np.zeros((8,8),np.uint8)).save(tmp_path/'prediction.png')
    with pytest.raises(ValueError,match='coordinate mismatch'):
        aligned_images(tmp_path,{'image':'image.png','mask':None},tmp_path/'prediction.png')


def test_qualitative_six_rows_are_reproducible_from_saved_masks(tmp_path,monkeypatch):
    import src.plot_paper as plotting
    from src.common import csv_write
    monkeypatch.setattr(plotting,'ROOT',tmp_path)
    base=tmp_path/'experiment';dest=tmp_path/'figures';data=tmp_path/'data';data.mkdir()
    atomic_json(base/'checkpoints/FROZEN.json',{'checkpoint_sha256':'unit-fixture-only','threshold':.5})
    records=[]
    for index,partition in enumerate(['Kvasir','CVC-ClinicDB','CVC-ColonDB','ETIS-LaribPolypDB','FAR','FAR']):
        positive=index<5;shape=(60+index*4,80+index*3);rgb=np.zeros((*shape,3),np.uint8);rgb[:,:,0]=100+index*15;rgb[:,:,1]=65
        truth=np.zeros(shape,bool)
        if positive:truth[15:35,20:45]=True
        pred=np.roll(truth,3,axis=1)
        if partition=='ETIS-LaribPolypDB':pred[:]=False
        Image.fromarray(rgb).save(data/f'{index}.png');Image.fromarray(truth).save(data/f'{index}_mask.png')
        key=f'unit_{index}';path=base/'predictions'/partition/key;path.parent.mkdir(parents=True,exist_ok=True)
        Image.fromarray(pred.astype(np.uint8)*255).save(path.with_suffix('.png'))
        atomic_json(path.with_suffix('.json'),{'checkpoint_sha256':'unit-fixture-only','threshold':.5,
            'row':{'image':f'{index}.png','mask':f'{index}_mask.png' if positive else None}})
        records.append({'id':key,'partition':partition,'positive':positive,'prediction_key':key,
                        'dice':0. if partition=='ETIS-LaribPolypDB' else .8 if positive else None,'iou':.7 if positive else None})
    csv_write(base/'evaluation/metrics/per_image.csv',records)
    assert plotting.generate(base,dest,data)==['qualitative_segmentation']
    selection=list(csv.DictReader((dest/'qualitative_selection.csv').open()))
    assert len(selection)==6 and selection[-1]['Dice']==''
    assert any(row['dataset']=='ETIS-LaribPolypDB' and float(row['Dice'])==0 for row in selection)
    assert (dest/'qualitative_segmentation.pdf').exists()


def test_test_lesion_disappearing_on_standard_grid_is_not_discarded(tmp_path):
    c=configuration();c.update(root=str(tmp_path),image_size=56,selection_size=14)
    truth=np.zeros((448,448),bool);truth[1,1]=True
    Image.fromarray(np.zeros((448,448,3),np.uint8)).save(tmp_path/'image.png');Image.fromarray(truth).save(tmp_path/'mask.png')
    r={'id':'tiny','image':'image.png','mask':'mask.png','image_sha256':file_hash(tmp_path/'image.png'),
       'mask_sha256':file_hash(tmp_path/'mask.png'),'archive':None,'positive':True,'split':'test'}
    assert not Prepared(c,[r])[0]['mask'].any()
    r['split']='train'
    with pytest.raises(ValueError,match='development lesion disappeared'):Prepared(c,[r])[0]


def test_memory_probe_checks_secondary_gpu_pressure(tmp_path,monkeypatch):
    import src.v2_features as features
    c=configuration();c.update(device='cuda',image_size=56,research=str(tmp_path/'v2'),v1_root=str(tmp_path/'v1'))
    c['gpu']['all_visible_devices']=True;c['gpu']['extraction_cap_per_gpu']=2;c['gpu']['amp']=False
    attempt={'value':0}
    monkeypatch.setattr(torch.cuda,'device_count',lambda:2)
    monkeypatch.setattr(torch.cuda,'empty_cache',lambda:None)
    def reset(index):
        if index==0:attempt['value']+=1
    monkeypatch.setattr(torch.cuda,'reset_peak_memory_stats',reset)
    monkeypatch.setattr(torch.cuda,'mem_get_info',lambda index:((100 if index==1 and attempt['value']==1 else 1000),2000))
    monkeypatch.setattr(torch.cuda,'memory_allocated',lambda index:0)
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda index:300)
    monkeypatch.setattr(features,'device',lambda c:torch.device('cpu'))
    monkeypatch.setattr(features,'synchronize',lambda c:None)
    monkeypatch.setattr(features,'rng_state',lambda:{})
    monkeypatch.setattr(features,'restore_rng',lambda state:None)
    assert features.tune(c,torch.nn.Identity(),'extraction')==2
    record=json.loads((output(c)/'reports/memory_probe_extraction.json').read_text())
    assert not record['probes'][0]['safe'] and record['probes'][1]['safe']


def test_launcher_status_needs_no_gpu_mask_and_start_requires_one(tmp_path):
    shutil.copy2(RESEARCH/'run_v2.sh',tmp_path/'run_v2.sh');(tmp_path/'configs').mkdir()
    shutil.copy2(RESEARCH/'configs/v2_resolution_448.yaml',tmp_path/'configs/v2_resolution_448.yaml')
    env=dict(os.environ,CORESEG_V2_PYTHON=sys.executable);env.pop('CUDA_VISIBLE_DEVICES',None)
    result=subprocess.run(['bash',str(tmp_path/'run_v2.sh'),'status'],env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert 'No live V2' in result.stdout
    result=subprocess.run(['bash',str(tmp_path/'run_v2.sh'),'start'],env=env,capture_output=True,text=True)
    assert result.returncode!=0 and 'allocated GPU' in result.stderr
    assert not (tmp_path/'outputs/dino_coreseg_v2_448/logs/pipeline.pid').exists()


def test_v1_guard_detects_changes_without_deleting(tmp_path):
    from src.v2_preserve import inventory,verify,V1_DIRS
    for name in V1_DIRS:(tmp_path/name).mkdir(parents=True)
    target=tmp_path/V1_DIRS[0]/'sentinel.txt';target.write_text('original')
    guard={'files':inventory(tmp_path)};assert verify(tmp_path,guard)
    target.write_text('changed')
    with pytest.raises(RuntimeError,match='preservation check'):verify(tmp_path,guard)
    assert target.exists()


def test_capture_reads_actual_v1_history_metrics_and_source_identity(tmp_path):
    from src.v2_preserve import capture,verify,V1_DIRS
    c=configuration();c.update(research=str(tmp_path/'v2'),v1_root=str(tmp_path/'v1'))
    root=Path(c['v1_root'])
    for name in V1_DIRS:(root/name).mkdir(parents=True)
    (root/'src').mkdir();source=root/'src/model_coreseg.py';source.write_text('# unit fixture source')
    holdout=root/'outputs/dino_coreseg_gpu_100/holdout';(holdout/'metrics').mkdir(parents=True)
    history=[{'epoch':i,'val_positive_dice':.83 if i==60 else .7,'val_normal_fpr':.08 if i==60 else .2} for i in range(1,101)]
    checkpoint=holdout/'best.pt';torch.save({'epoch':59,'history':history[:60],
        'identity':{'implementation':{'model_coreseg.py':file_hash(source)}}},checkpoint)
    atomic_json(holdout/'FROZEN.json',{'selected_epoch':60,'completed_epochs':100,'checkpoint_sha256':file_hash(checkpoint)})
    atomic_json(holdout/'evaluation_status.json',{'completed':True,'images':2,'checkpoint_sha256':file_hash(checkpoint),'partitions':['FAR']})
    from src.common import csv_write
    csv_write(holdout/'training_history.csv',history)
    csv_write(holdout/'metrics/final_metrics.csv',[{'partition':'FAR','target':'positive','images':1,'dice':.6},{'partition':'FAR','target':'normal','images':1}])
    cv=root/'outputs/dino_coreseg_gpu_100_cv5';atomic_json(cv/'status.json',{'completed':True})
    for fold in range(1,6):
        folder=cv/f'fold_{fold:02d}';folder.mkdir();(folder/'best.pt').write_bytes(b'unit-fixture-checkpoint')
        atomic_json(folder/'FROZEN.json',{'checkpoint_sha256':file_hash(folder/'best.pt')})
    guard=capture(c)
    assert guard['training_epochs_in_csv']==100 and guard['selected_epoch']==60
    assert guard['selected_validation_record']['val_positive_dice']==.83
    assert guard['source_matches_checkpoint']=={'model_coreseg.py':True}
    assert not guard['git_directory_present'] and verify(root,guard)


def test_real_official_frozen_dino_448_and_multilayer_shapes():
    c=configuration();model,identity=load_backbone(c)
    assert identity['parameters']==22056576 and not any(p.requires_grad for p in model.parameters())
    with torch.inference_mode():features=model(torch.zeros(1,3,448,448))
    assert features.shape==(1,1,384,32,32)
    raw=model.model.backbone if hasattr(model,'model') else model.backbone
    c['variant']='multilayer';c['intermediate_layers']=[2,5,8,11]
    with torch.inference_mode():multi=SpatialDinoV2(raw,c)(torch.zeros(1,3,448,448))
    assert multi.shape==(1,4,384,32,32)

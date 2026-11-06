"""Plot-control tests use explicit temporary fixtures, never research results."""
import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest
from src.paper_cv_plots import FIELDS,read_histories,draw,generate,plt


def history(base,fold,epochs=(1,2,3)):
    path=base/f'fold_{fold:02d}/training_history.csv';path.parent.mkdir(parents=True)
    rows=[{'epoch':epoch,'train_loss_0_1':.4/epoch,'val_loss_0_1':.45/epoch,
           'train_accuracy':.7+epoch*.03,'val_accuracy':.65+epoch*.04} for epoch in epochs]
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['epoch',*FIELDS]);writer.writeheader();writer.writerows(rows)
    return path,rows


def test_five_folds_have_ten_exact_unmodified_curves(tmp_path):
    for fold in range(1,6):history(tmp_path,fold)
    histories=read_histories(tmp_path)
    fig=draw(histories,'percent',100)
    try:
        assert len(fig.axes)==2
        for ax,metric in zip(fig.axes,['loss_0_1','accuracy']):
            assert len(ax.lines)==10
            assert ax.get_ylim()==(0,100) and ax.get_xlim()==(0,100)
            for index,line in enumerate(ax.lines):
                fold=index//2;split='train' if index%2==0 else 'val';rows=histories[fold][1]
                assert list(line.get_xdata())==[1,2,3]
                assert list(line.get_ydata())==[100*row[f'{split}_{metric}'] for row in rows]
                assert line.get_linestyle()==('-' if split=='train' else '--')
                assert f'Fold {fold+1}' in line.get_label()
        fig.savefig(tmp_path/'layout_fixture.png',dpi=100,bbox_inches='tight')
    finally:plt.close(fig)


def test_missing_folds_and_unequal_epochs_are_not_fabricated(tmp_path):
    history(tmp_path,1,(1,2,4));history(tmp_path,3,(1,2))
    fig=draw(read_histories(tmp_path),'unit')
    try:
        assert len(fig.axes[0].lines)==4
        assert list(fig.axes[0].lines[0].get_xdata())==[1,2,4]
        assert list(fig.axes[0].lines[2].get_xdata())==[1,2]
        assert fig.axes[0].get_ylim()==(0,1)
    finally:plt.close(fig)


def test_cannot_hide_recorded_epochs(tmp_path):
    history(tmp_path,1)
    with pytest.raises(ValueError,match='hide measured epochs'):draw(read_histories(tmp_path),max_epochs=2)


@pytest.mark.parametrize('bad_value',['nan','inf','1.5','-0.1'])
def test_invalid_fraction_is_rejected(tmp_path,bad_value):
    path,_=history(tmp_path,1)
    text=path.read_text();path.write_text(text.replace('0.4,',bad_value+',',1))
    with pytest.raises(ValueError,match='finite fraction'):read_histories(tmp_path)


def test_empty_results_do_not_produce_placeholder_figures(tmp_path):
    assert generate(tmp_path)==[]
    assert not (tmp_path/'figures').exists()


def test_all_exports_and_provenance_preserve_source_csv(tmp_path):
    path,_=history(tmp_path,1);original=path.read_bytes()
    exports=generate(tmp_path,scale='both',dpi=72)
    assert len(exports)==8
    assert all((tmp_path/'figures'/name).stat().st_size>0 for name in exports)
    assert path.read_bytes()==original
    metadata=json.loads((tmp_path/'figures/cv_loss_accuracy_manifest.json').read_text())
    assert metadata['folds_available']==[1] and metadata['folds_expected']==5
    assert metadata['epochs_available']=={'1':[1,2,3]}
    assert 'L/(1+L)' in metadata['caption']


def test_command_runs_without_torch_or_cuda(tmp_path):
    history(tmp_path,1)
    result=subprocess.run([sys.executable,'-m','src.paper_cv_plots','--results-dir',str(tmp_path),
                           '--scale','percent','--dpi','72'],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert 'Saved 4 figure files' in result.stdout

import copy
import json
import sys
from pathlib import Path

import pytest

from test_experiment import setup
from test_multiscene_runtime import records, configure, perfect
from medical_kv.config import ROOT, load_config
from medical_kv.data import dataset_identity, load_split
from medical_kv.io import read_checkpoint
from medical_kv.training import train_soft

sys.path.insert(0, str(ROOT / 'scripts'))
from build_reduced_multiscene_dataset import EXCLUDED, reduce_record
from multiscene_common import make_qa
import run_multiscene_experiment as pipeline


def test_reduction_preserves_input_identity_and_remaining_supervision():
    original = records()[1]
    for field in EXCLUDED:
        original['fields'][field] = '123456789'
        original['field_annotations'][field] = {'origin': 'native', 'evidence_span': [0, 9]}
    make_qa(original)
    original['native_target_count'] = len(original['fields'])
    before = copy.deepcopy(original)
    reduced = reduce_record(original, 'reduced')
    assert original == before
    assert reduced['private_prefix'] == original['private_prefix']
    assert reduced['group_id'] == original['group_id']
    assert reduced['record_id'] == original['record_id']
    for key in ('fields', 'field_annotations'):
        assert reduced[key] == {k: v for k, v in original[key].items() if k not in EXCLUDED}
    for key in ('qa', 'training_qa', 'evaluation_qa'):
        assert reduced[key] == [q for q in original[key] if q['field'] not in EXCLUDED]


@pytest.mark.parametrize('scene,counts', [('medical', (4000,800,4000)),
                                       ('finance', (3565,713,3565)), ('hr', (2000,400,2000))])
def test_scene_configs_and_complete_reduced_data(scene, counts):
    cfg = load_config(f'configs/experiment_v6_{scene}_no_negatives.json')
    evaluation = load_config(f'configs/evaluation_v2_{scene}.json')
    assert cfg['scene'] == evaluation['scene'] == scene
    assert cfg['dataset_directory'] == evaluation['dataset_directory'] == 'data/dataset_multiscene_v2_reduced'
    assert cfg['input_cache_directory'] == 'data/input_cache_multiscene_v2_clean'
    assert cfg['epochs'] == 100 and cfg['batch_records'] == 8 and cfg['prompt_length'] == 32
    assert cfg['early_stopping'] is False and cfg['diagnosis_contrastive']['enabled'] is False
    # Full scene token caches are prepared on the server. These local checks
    # inspect data only; small cache/model equivalence has a separate fixture.
    cfg.pop('input_cache_directory')
    identity = dataset_identity(cfg)
    assert identity['selection'] == {'scene': scene, 'record_form': None}
    for split, count in zip(('train','validation','test'), counts):
        selected = load_split(cfg, split)
        assert len(selected) == count
        assert {r['scene'] for r in selected} == {scene}
        for record in selected:
            assert not set(record['fields']) & EXCLUDED
            assert not set(record['field_annotations']) & EXCLUDED
            assert all(q['field'] not in EXCLUDED for key in ('qa','training_qa','evaluation_qa') for q in record[key])


@pytest.mark.parametrize('scene', ['medical', 'finance', 'hr'])
def test_scene_pipeline_uses_separate_matching_train_and_test(tmp_path, monkeypatch, scene):
    train_cfg, test_cfg, run, test_out, log = pipeline.experiment_paths(tmp_path, scene)
    train_cfg.parent.mkdir()
    cfg = load_config(f'configs/experiment_v6_{scene}_no_negatives.json')
    train_cfg.write_text(json.dumps(cfg))
    test_cfg.write_text(json.dumps(dict(cfg, output_directory=str(test_out))))
    calls = []
    def stage(command, **kw):
        calls.append(command)
        if len(calls) == 1:
            run.mkdir(parents=True)
            (run / 'best.pt').write_bytes(b'fixture')
            (run / 'run.json').write_text(json.dumps({'config': cfg,'run_signature':'fixture'}))
            (run / 'run_summary.json').write_text(json.dumps({'status':'complete','partial_dataset':False,
                'stop_reason':'epochs_completed','state':{'epoch':100},'run_signature':'fixture'}))
    monkeypatch.setattr(pipeline, 'run_child', stage)
    pipeline.run_experiment(tmp_path, '0,2', scene=scene, epoch_validation=False)
    assert len(calls) == 2
    assert calls[0][calls[0].index('--config')+1] == str(train_cfg)
    assert '--no-epoch-validation' in calls[0]
    assert calls[1][calls[1].index('--config')+1] == str(test_cfg)
    assert calls[1][calls[1].index('--checkpoint')+1] == str(run/'best.pt')
    assert calls[1][calls[1].index('--output-directory')+1] == str(test_out)
    assert log.exists()
    test_config = json.loads(test_cfg.read_text())
    test_config['scene'] = 'finance' if scene != 'finance' else 'hr'
    test_cfg.write_text(json.dumps(test_config))
    with pytest.raises(ValueError, match='same scene'):
        pipeline.run_experiment(tmp_path, '0,2', scene=scene)


def test_three_separate_tiny_prompts_train_on_own_scene(setup, tmp_path, monkeypatch):
    backend, cfg, _ = setup
    configure(cfg)
    cfg.update(epochs=2, comparison_epochs=1, early_stopping=False)
    monkeypatch.setattr(backend, 'generate_many', perfect)
    checkpoints = []
    for record in records():
        reduced = reduce_record(record, 'fixture_reduced')
        scene = record['scene']
        cfg.update(scene=scene, output_directory=str(tmp_path / scene))
        identity = {'schema':'multiscene_privacy_v1', 'selection': {'scene':scene,'record_form':None}}
        result = train_soft(backend, cfg, [reduced], [reduced], identity, {'development_only':True}, epoch_validation=False)
        assert result['state']['epoch'] == 2
        saved = read_checkpoint(tmp_path / scene / 'final.pt')
        assert saved['run_info']['dataset']['selection']['scene'] == scene
        assert not set(saved['state']['field_exposures']) & EXCLUDED
        assert len(list((tmp_path / scene / 'validation_predictions').glob('*/predictions.csv'))) == 1
        checkpoints.append(saved['run_signature'])
    assert len(set(checkpoints)) == 3


def test_scene_recovery_forwards_scene_and_keeps_outputs_separate(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import training_control as command
    import medical_kv.control as control
    root = tmp_path / 'project'
    run = root / 'runs/soft_finance_v2_seed17_100epochs'
    run.mkdir(parents=True)
    (run / 'run.json').write_text(json.dumps({'method':'soft','config':{
        'scene':'finance','question_protocol':'multiscene_privacy_v1'}}))
    # An older joint test must not block the independent finance experiment.
    (root / 'runs/test_multiscene_v1_best').mkdir()
    monkeypatch.setattr(command, 'ROOT', root)
    monkeypatch.setattr(command, 'os', SimpleNamespace(name='posix'))
    monkeypatch.setattr(command, 'training_status', lambda *a: {'running':False,'checkpoint_exists':True,'status':'paused'})
    monkeypatch.setattr(control, 'process_start', lambda pid: 'fake-start')
    calls = []
    monkeypatch.setattr(command.subprocess, 'Popen', lambda args, **kw: calls.append(args) or SimpleNamespace(pid=123))
    result = command.recover(run, ['0','2'], epoch_validation=False)
    assert result['gpus'] == [0,2]
    assert Path(result['log']).parent == root/'logs'
    assert calls[0][calls[0].index('--scene')+1] == 'finance'
    assert '--no-epoch-validation' in calls[0]

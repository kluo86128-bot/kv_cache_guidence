import copy

import pytest
import torch

from test_experiment import setup
from medical_kv import training
from medical_kv.io import read_checkpoint, write_json


def test_pause_resume_skip_and_inherit_keep_optimizer_and_final_selection(setup, tmp_path, monkeypatch):
    backend, cfg, record = setup
    cfg.update(epochs=4, early_stopping=False, comparison_epochs=2)
    validation_epochs = []
    def evaluate(*a, **kw):
        epoch = kw['metadata']['epoch']
        validation_epochs.append(epoch)
        return {'macro_em': 1.0 / epoch, 'macro_nll': float(epoch)}, []
    monkeypatch.setattr(training, 'evaluate_records', evaluate)
    class Curve:
        def __init__(self, *a, **kw): pass
        def update_training(self, *a): pass
        def update_validation(self, *a): pass
        def save(self): pass
    monkeypatch.setattr(training, 'LossCurve', Curve)
    subsets = {'development_only': True}
    reference = tmp_path / 'reference'
    cfg['output_directory'] = str(reference)
    training.train_soft(backend, cfg, [record], [record], {'toy': True}, subsets)
    expected = read_checkpoint(reference / 'latest.pt')
    assert validation_epochs == [1, 2, 3, 4]
    validation_epochs.clear()
    out = tmp_path / 'resumed'
    cfg['output_directory'] = str(out)
    save = training.save_checkpoint
    pauses = {2, 3}
    def request(path, saved):
        save(path, saved)
        if path.name == 'latest.pt' and saved['state']['step'] in pauses:
            pauses.remove(saved['state']['step'])
            write_json(out / 'pause_request.json', {'request_id': 'test'})
    monkeypatch.setattr(training, 'save_checkpoint', request)
    first = training.train_soft(backend, cfg, [record], [record], {'toy': True}, subsets)
    assert first['stop_reason'] == 'paused' and validation_epochs == [1]
    prior_best = (out / 'best.pt').read_bytes()
    second = training.train_soft(backend, cfg, [record], [record], {'toy': True}, subsets,
                                 resume=out / 'latest.pt', epoch_validation=False)
    assert second['stop_reason'] == 'paused' and validation_epochs == [1]
    assert (out / 'last_first2.pt').exists()
    assert 'metrics' not in second['checkpoint_results']['checkpoints']['last_first2']
    final = training.train_soft(backend, cfg, [record], [record], {'toy': True}, subsets,
                                resume=out / 'latest.pt')
    assert final['status'] == 'complete' and validation_epochs == [1, 4]
    actual = read_checkpoint(out / 'latest.pt')
    assert actual['run_info']['validation_policy']['every_epoch'] is False
    assert actual['state']['step'] == expected['state']['step'] == 4
    assert actual['scheduler'] == expected['scheduler']
    torch.testing.assert_close(actual['prompt'], expected['prompt'], rtol=0, atol=0)
    for key in ('exp_avg', 'exp_avg_sq', 'step'):
        torch.testing.assert_close(actual['optimizer']['state'][0][key], expected['optimizer']['state'][0][key])
    assert (out / 'best.pt').read_bytes() == prior_best
    assert (out / 'final.pt').exists()


def test_disabled_validation_rejects_early_stopping(setup):
    backend, cfg, record = setup
    cfg['early_stopping'] = True
    with pytest.raises(ValueError, match='early_stopping=false'):
        training.train_soft(backend, cfg, [record], [record], {'toy': True},
                            {'development_only': True}, epoch_validation=False)


def test_recovery_option_reaches_training_but_not_test(tmp_path, monkeypatch):
    import training_control as control
    from test_multiscene_pipeline import prepare
    import run_multiscene_experiment as pipeline
    calls = []
    monkeypatch.setattr(control, 'recover', lambda root, gpus, **kw: calls.append(kw) or {})
    assert control.main(['recover', '--run-root', str(tmp_path), '--gpus', '0', '1',
                         '--no-epoch-validation']) == 0
    assert calls == [{'epoch_validation': False}]
    prepare(tmp_path)
    calls.clear()
    monkeypatch.setattr(pipeline, 'run_child', lambda command, **kw: calls.append(command))
    pipeline.run_experiment(tmp_path, '0,1', resume=True, epoch_validation=False)
    assert '--no-epoch-validation' in calls[0]
    assert '--no-epoch-validation' not in calls[1]

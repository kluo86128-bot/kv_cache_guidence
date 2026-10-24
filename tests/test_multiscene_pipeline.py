"""Verify automatic testing cannot run after failed or partial training."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_multiscene_experiment as pipeline


def prepare(root, *, status='complete', partial=False, epoch=100, reason='epochs_completed'):
    (root / 'configs').mkdir()
    (root / 'configs/experiment_v5_multiscene_no_negatives.json').write_text('{"epochs":100}')
    (root / 'configs/evaluation_multiscene_v1.json').write_text('{}')
    run = root / 'runs/soft_multiscene_v1_seed17_100epochs'
    run.mkdir(parents=True)
    (run / 'best.pt').write_bytes(b'fixture checkpoint')
    (run / 'run.json').write_text(json.dumps({'config': {'epochs': 100}, 'run_signature': 'same'}))
    (run / 'run_summary.json').write_text(json.dumps({'status': status, 'partial_dataset': partial,
        'stop_reason': reason, 'state': {'epoch': epoch}, 'run_signature': 'same'}))
    return run


def test_success_runs_test_after_training_on_same_gpus(tmp_path, monkeypatch):
    prepare(tmp_path)
    calls = []
    def stage(command, **kwargs):
        calls.append(command)
        if len(calls) == 2:
            assert command[command.index('--gpus') + 1] == '0,2'
            assert command[command.index('--split') + 1] == 'test'
            assert command[command.index('--checkpoint') + 1].endswith('best.pt')
            kwargs['stdout'].write('test output\n')
    monkeypatch.setattr(pipeline, 'run_child', stage)
    pipeline.run_experiment(tmp_path, '0,2')
    assert [cmd[4] for cmd in calls] == ['train-soft', 'evaluate']
    assert (tmp_path / 'logs/test_multiscene_v1_best.log').read_text() == 'test output\n'


@pytest.mark.parametrize('changes', [
    {'status': 'partial', 'epoch': 20, 'reason': 'time_limit'},
    {'partial': True}, {'epoch': 99}, {'reason': 'early_stopping'},
])
def test_incomplete_training_never_launches_test(tmp_path, monkeypatch, changes):
    prepare(tmp_path, **changes)
    calls = []
    monkeypatch.setattr(pipeline, 'run_child', lambda command, **kw: calls.append(command))
    with pytest.raises(RuntimeError, match='did not complete'):
        pipeline.run_experiment(tmp_path, '0,1')
    assert len(calls) == 1
    assert not (tmp_path / 'logs/test_multiscene_v1_best.log').exists()


@pytest.mark.parametrize('reason', ['paused', 'interrupted'])
def test_cooperative_pause_defers_automatic_test(tmp_path, monkeypatch, reason):
    prepare(tmp_path, status='partial', epoch=20, reason=reason)
    calls = []
    monkeypatch.setattr(pipeline, 'run_child', lambda command, **kw: calls.append(command))
    pipeline.run_experiment(tmp_path, '0,1')
    assert len(calls) == 1
    assert not (tmp_path / 'logs/test_multiscene_v1_best.log').exists()


def test_training_failure_does_not_test_stale_successful_run(tmp_path, monkeypatch):
    prepare(tmp_path)
    calls = []
    def fail(command, **kw):
        calls.append(command)
        raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(pipeline, 'run_child', fail)
    with pytest.raises(subprocess.CalledProcessError):
        pipeline.run_experiment(tmp_path, '0,1')
    assert len(calls) == 1


def test_resume_keeps_auto_test_and_test_failure_propagates(tmp_path, monkeypatch):
    prepare(tmp_path)
    calls = []
    def stage(command, **kwargs):
        calls.append(command)
        if len(calls) == 2:
            raise subprocess.CalledProcessError(2, command)
    monkeypatch.setattr(pipeline, 'run_child', stage)
    with pytest.raises(subprocess.CalledProcessError):
        pipeline.run_experiment(tmp_path, '0,1', resume=True)
    assert calls[0][calls[0].index('--resume') + 1].endswith('latest.pt')
    assert len(calls) == 2


def test_existing_test_output_is_not_overwritten(tmp_path, monkeypatch):
    prepare(tmp_path)
    (tmp_path / 'runs/test_multiscene_v1_best').mkdir()
    def forbidden(*a, **kw):
        pytest.fail('Existing test output should be checked before starting training')
    monkeypatch.setattr(pipeline, 'run_child', forbidden)
    with pytest.raises(FileExistsError):
        pipeline.run_experiment(tmp_path, '0,1')

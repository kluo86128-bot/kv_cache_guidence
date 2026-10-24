"""Checkpoint/topology continuation and cooperative pause contracts."""
import copy
import json
from pathlib import Path
import socket

import pytest
import torch
from filelock import FileLock

from test_experiment import setup
from medical_kv import control, resume, training
from medical_kv.config import ROOT, fingerprint
from medical_kv.io import code_identity, read_checkpoint, save_checkpoint


def legacy_checkpoint(contract, *, world_size=1):
    registry = json.loads((ROOT / 'configs/resume_compatibility_v1.json').read_text())
    old_code = {key: values[0] for key, values in registry['legacy_sources'][0].items()}
    origin = dict(contract, code=old_code,
                  subsets=dict(contract['subsets'], parallel_world_size=world_size))
    signature = fingerprint(origin)
    return {'method': 'soft', 'run_signature': signature, 'run_info': {
        'method': 'soft', 'run_signature': signature, 'config': origin['config'],
        'dataset': origin['dataset'], 'subsets': origin['subsets'], 'backend': origin['backend'],
        'code_sha256': old_code}}


def contract():
    return resume.training_contract({'model': 'toy'}, {'batch_records': 8}, {'sha': 'data'},
                                    {'train_ids': ['a'], 'parallel_world_size': 4}, code_identity())


def test_reviewed_legacy_migration_keeps_origin_and_rejects_changes():
    target = contract()
    old = legacy_checkpoint(target)
    origin, migrated = resume.validate_resume(old, target)
    assert migrated and fingerprint(origin) == old['run_signature']
    assert 'parallel_world_size' not in target['subsets']
    for key in ('config', 'dataset', 'subsets', 'backend'):
        changed = copy.deepcopy(target)
        changed[key]['unauthorized_change'] = True
        with pytest.raises(ValueError, match='identity mismatch'):
            resume.validate_resume(old, changed)
    changed = copy.deepcopy(old)
    changed['run_info']['code_sha256']['backend.py'] = 'unknown-version'
    old_origin = dict(origin, code=changed['run_info']['code_sha256'])
    changed['run_signature'] = changed['run_info']['run_signature'] = fingerprint(old_origin)
    with pytest.raises(ValueError, match='reviewed migration'):
        resume.validate_resume(changed, target)


def test_future_checkpoint_checks_code_and_signature():
    target = contract()
    signature = fingerprint(target)
    saved = {'method': 'soft', 'resume_format': 2, 'run_signature': signature,
             'run_info': {'method': 'soft', 'run_signature': signature,
                          'origin_identity': target, 'resume_contract': target}}
    assert resume.validate_resume(saved, target) == (target, False)
    changed = copy.deepcopy(target)
    changed['code']['training.py'] = 'unreviewed'
    with pytest.raises(ValueError, match='reviewed migration'):
        resume.validate_resume(saved, changed)
    saved['run_signature'] = 'corrupt'
    with pytest.raises(ValueError, match='origin signature'):
        resume.validate_resume(saved, target)


class SimulatedRanks:
    """CPU reference: execute every rank's share before the global Adam step."""
    world_size = 1
    rank = 0
    primary = True

    def __init__(self, device):
        self.device = device

    def barrier(self):
        pass

    def objects(self, value):
        return [copy.deepcopy(value) for _ in range(self.world_size)]

    def broadcast(self, value):
        return value

    def sum_(self, value):
        return value

    def maximum(self, value):
        return value

    def any(self, value):
        return bool(value)

    def metadata(self):
        return {'world_size': self.world_size, 'visible_gpu_ids': 'simulated'}


def test_pause_migrate_one_to_four_then_two_keeps_progress_and_optimizer(setup, tmp_path, monkeypatch):
    backend, cfg, record = setup
    cfg.update(epochs=2, batch_records=2, early_stopping=False)
    rows = [dict(copy.deepcopy(record), record_id=f'row_{i}') for i in range(5)]
    for i, row in enumerate(rows):
        row['fields']['diagnosis'] = f'Cold {i}'
        row['private_prefix'] = row['private_prefix'].replace('Cold', f'Cold {i}')
        next(q for q in row['qa'] if q['field'] == 'diagnosis')['answer'] = f'Cold {i}'
    subsets = {'development_only': True}
    monkeypatch.setattr(training, 'Runtime', SimulatedRanks)
    monkeypatch.setattr(training, 'evaluate_records', lambda *a, **kw: ({'macro_em': .5, 'macro_nll': 1.}, []))
    class Curve:
        def __init__(self, *a, **kw): pass
        def update_training(self, *a): pass
        def update_validation(self, *a): pass
        def save(self): pass
    monkeypatch.setattr(training, 'LossCurve', Curve)
    backward = training.local_batch_backward
    batches = []
    def all_ranks(backend, prompt, records, indices, rank, world_size, **kwargs):
        batches.append(list(indices))
        losses = []
        for worker in range(world_size):
            losses += backward(backend, prompt, records, indices, worker, world_size, **kwargs)
        return losses
    monkeypatch.setattr(training, 'local_batch_backward', all_ranks)
    reference = tmp_path / 'reference'
    cfg['output_directory'] = str(reference)
    SimulatedRanks.world_size = 1
    training.train_soft(backend, cfg, rows, [record], {'toy': True}, subsets)
    expected = read_checkpoint(reference / 'latest.pt')
    expected_batches = list(batches)
    batches.clear()
    actual_dir = tmp_path / 'resumed'
    cfg['output_directory'] = str(actual_dir)
    pause_steps = {1, 3}
    save = training.save_checkpoint
    def save_and_request(path, value):
        save(path, value)
        step = value['state']['step']
        if Path(path).name == 'latest.pt' and step in pause_steps:
            pause_steps.remove(step)
            control.write_json(actual_dir / 'pause_request.json', {'request_id': f'pause_{step}'})
    monkeypatch.setattr(training, 'save_checkpoint', save_and_request)
    first = training.train_soft(backend, cfg, rows, [record], {'toy': True}, subsets)
    assert first['stop_reason'] == 'paused'
    assert first['state']['cursor'] == 2
    # Convert the first checkpoint to the exact old schema to test real Adam,
    # scheduler and prompt restoration through the reviewed legacy migration.
    saved = read_checkpoint(actual_dir / 'latest.pt')
    old = legacy_checkpoint(saved['run_info']['resume_contract'])
    saved.pop('resume_format')
    saved['run_signature'] = old['run_signature']
    saved['run_info'].update(old['run_info'])
    for name in ('origin_identity', 'resume_contract', 'resume_history'):
        saved['run_info'].pop(name, None)
    save_checkpoint(actual_dir / 'latest.pt', saved)
    original_signature = saved['run_signature']
    SimulatedRanks.world_size = 4
    second = training.train_soft(backend, cfg, rows, [record], {'toy': True}, subsets,
                                 resume=actual_dir / 'latest.pt')
    assert second['stop_reason'] == 'paused' and second['state']['step'] == 3
    assert second['state']['cursor'] == 5  # next recovery must validate, not repeat this epoch
    SimulatedRanks.world_size = 2
    final = training.train_soft(backend, cfg, rows, [record], {'toy': True}, subsets,
                                resume=actual_dir / 'latest.pt')
    actual = read_checkpoint(actual_dir / 'latest.pt')
    assert final['status'] == 'complete'
    assert final['run_signature'] == original_signature
    assert expected['state']['step'] == actual['state']['step'] == 6
    assert batches == expected_batches
    assert expected['state']['field_exposures'] == actual['state']['field_exposures']
    assert expected['scheduler'] == actual['scheduler']
    torch.testing.assert_close(expected['prompt'], actual['prompt'], rtol=1e-5, atol=1e-6)
    for key in ('step', 'exp_avg', 'exp_avg_sq'):
        torch.testing.assert_close(expected['optimizer']['state'][0][key],
                                   actual['optimizer']['state'][0][key], rtol=1e-5, atol=1e-6)
    assert [entry['to_parallelism']['world_size'] for entry in actual['run_info']['resume_history']] == [4, 2]
    best = read_checkpoint(actual_dir / 'best.pt')
    assert best['run_signature'] == final['run_signature']
    assert best['run_info']['parallelism']['world_size'] == 2


def test_rng_streams_restored_and_new_ranks_seeded(monkeypatch):
    restored, seeded = [], []
    monkeypatch.setattr(resume, 'restore_rng', restored.append)
    monkeypatch.setattr(resume, 'seed_all', seeded.append)
    saved = {'rng_by_rank': ['rank0'], 'state': {'step': 17}}
    runtime = SimulatedRanks('cpu')
    runtime.world_size = 4
    runtime.rank = 0
    resume.restore_training_rng(saved, runtime, 17)
    assert restored == ['rank0']
    runtime.rank = 3
    resume.restore_training_rng(saved, runtime, 17)
    resume.restore_training_rng(saved, runtime, 17)
    assert seeded[0] == seeded[1] and len(seeded) == 2


def test_best_selected_before_legacy_migration_remains_eligible(setup, tmp_path, monkeypatch):
    backend, cfg, record = setup
    cfg.update(epochs=2, early_stopping=False, output_directory=str(tmp_path / 'best_run'))
    out = Path(cfg['output_directory'])
    monkeypatch.setattr(training, 'evaluate_records', lambda *a, **kw: ({'macro_em': .5, 'macro_nll': 1.}, []))
    save = training.save_checkpoint
    requested = []
    def pause_after_first_validation(path, value):
        save(path, value)
        if Path(path).name == 'latest.pt' and value['state']['epoch'] == 1 and not requested:
            requested.append(True)
            control.write_json(out / 'pause_request.json', {'request_id': 'validated'})
    monkeypatch.setattr(training, 'save_checkpoint', pause_after_first_validation)
    training.train_soft(backend, cfg, [record], [record], {'toy': True}, {'development_only': False})
    for name in ('latest.pt', 'best.pt'):
        value = read_checkpoint(out / name)
        old = legacy_checkpoint(value['run_info']['resume_contract'])
        value.pop('resume_format')
        value['run_signature'] = old['run_signature']
        value['run_info'].update(old['run_info'])
        for key in ('origin_identity', 'resume_contract', 'resume_history'):
            value['run_info'].pop(key, None)
        save_checkpoint(out / name, value)
    before = (out / 'best.pt').read_bytes()
    result = training.train_soft(backend, cfg, [record], [record], {'toy': True},
                                 {'development_only': False}, resume=out / 'latest.pt')
    best = read_checkpoint(out / 'best.pt')
    assert (out / 'best.pt').read_bytes() == before  # no synthetic rewrite/reselection
    assert best['selected_by_validation']
    assert best['run_signature'] == result['run_signature']
    assert result['status'] == 'complete' and not result['partial_dataset']


def test_pause_failure_does_not_reuse_stale_paused_summary(tmp_path):
    control.write_json(tmp_path / 'run.json', {})
    control.write_json(tmp_path / 'run_summary.json', {'status': 'partial', 'stop_reason': 'paused'})
    control.write_json(tmp_path / 'training_control.json', {'status': 'failed'})
    (tmp_path / 'latest.pt').write_bytes(b'old checkpoint')
    assert control.training_status(tmp_path)['status'] == 'failed'
    with pytest.raises(RuntimeError, match='without an acknowledged safe pause'):
        control.wait_paused(tmp_path, timeout=.1)


def test_legacy_wait_acknowledges_completed_save(tmp_path):
    control.write_json(tmp_path / 'run.json', {})
    state = {'step': 17, 'epoch': 0, 'cursor': 136}
    control.write_json(tmp_path / 'run_summary.json', {'status': 'partial',
        'stop_reason': 'interrupted', 'state': state})
    control.write_json(tmp_path / 'pause_request.json', {'request_id': 'legacy',
        'mechanism': 'legacy_safe_stop_signal'})
    (tmp_path / 'latest.pt').write_bytes(b'committed checkpoint')
    assert control.wait_paused(tmp_path, timeout=.1)['status'] == 'paused'
    assert not (tmp_path / 'pause_request.json').exists()
    acknowledged = control.read_json(tmp_path / 'pause_state.json')
    assert acknowledged['request_id'] == 'legacy' and acknowledged['step'] == 17


def test_file_pause_acknowledges_after_checkpoint_and_releases_lock(tmp_path):
    control.write_json(tmp_path / 'run.json', {})
    controller = control.PauseController(tmp_path)
    controller.begin(resume=False, step=0, world_size=1)
    with FileLock(str(tmp_path / '.run.lock')):
        control.write_json(controller.request_path, {'request_id': 'one'})
        assert control.training_status(tmp_path)['status'] == 'pause_requested'
        (tmp_path / 'latest.pt').write_bytes(b'checkpoint')
        state = {'step': 3, 'epoch': 0, 'cursor': 24}
        control.write_json(tmp_path / 'run_summary.json', {'status': 'partial', 'stop_reason': 'paused', 'state': state})
        controller.finish('paused', state, reason='paused')
        assert control.training_status(tmp_path)['running']
    assert control.wait_paused(tmp_path, timeout=.1)['status'] == 'paused'
    assert control.read_json(controller.state_path)['request_id'] == 'one'
    assert not controller.requested()
    controller.begin(resume=True, step=3, world_size=4)
    assert control.read_json(controller.process_path)['world_size'] == 4


def test_pause_old_process_signals_only_verified_workers(tmp_path, monkeypatch):
    control.write_json(tmp_path / 'run.json', {})
    monkeypatch.setattr(control, 'lock_held', lambda path: True)
    monkeypatch.setattr(control.Path, 'is_dir', lambda self: True)
    monkeypatch.setattr(control, 'legacy_workers', lambda *a: [(123, 'started')])
    monkeypatch.setattr(control, 'process_start', lambda pid: 'started')
    signals = []
    monkeypatch.setattr(control.os, 'kill', lambda *args: signals.append(args))
    request = control.request_pause(tmp_path, ROOT)
    assert request['mechanism'] == 'legacy_safe_stop_signal'
    assert signals == [(123, control.signal.SIGTERM)]


def test_current_process_uses_request_file_without_signal(tmp_path, monkeypatch):
    control.write_json(tmp_path / 'run.json', {})
    control.write_json(tmp_path / 'training_control.json', {'status': 'running', 'pid': 123,
        'host': socket.gethostname(), 'start_ticks': 'started'})
    monkeypatch.setattr(control, 'lock_held', lambda path: True)
    monkeypatch.setattr(control, 'process_start', lambda pid: 'started')
    monkeypatch.setattr(control.os, 'kill', lambda *a: pytest.fail('File pause must not send a signal'))
    assert control.request_pause(tmp_path, ROOT)['mechanism'] == 'request_file'


def test_legacy_process_scan_excludes_other_runs_and_launchers(tmp_path):
    project, root = tmp_path / 'project', tmp_path / 'run'
    project.mkdir(); root.mkdir()
    proc = tmp_path / 'proc'; proc.mkdir()
    def worker(pid, args, target=project):
        directory = proc / str(pid); directory.mkdir()
        (directory / 'cmdline').write_bytes(('\0'.join(args) + '\0').encode())
        (directory / 'stat').write_text(f'{pid} (python) ' + ' '.join(['x'] * 19 + ['start']))
        # Windows tests avoid creating symlinks: cwd as a directory works with resolve.
        (directory / 'cwd').mkdir()
        return directory / 'cwd'
    args = ['python', '-m', 'medical_kv', 'train-soft', '--output-directory', str(root)]
    cwd = worker(123, args)
    assert control.legacy_workers(root, cwd, proc) == [(123, 'start')]
    worker(124, args + ['--gpus', '0'])
    worker(125, ['python', '-m', 'torch.distributed.run', '--module', 'medical_kv', *args[3:]])
    assert control.legacy_workers(root, cwd, proc) == [(123, 'start')]


def test_resume_command_accepts_space_or_comma_gpu_ids_and_refuses_live_run(tmp_path, monkeypatch):
    import training_control as command
    calls = []
    monkeypatch.setattr(command, 'recover', lambda root, gpus: calls.append(gpus) or {'status': 'launched'})
    assert command.main(['recover', '--run-root', str(tmp_path), '--gpus', '0', '2', '3']) == 0
    assert command.main(['resume', '--run-root', str(tmp_path), '--gpus', '0,2,3']) == 0
    assert calls == [['0', '2', '3'], ['0,2,3']]


def test_recovery_refuses_active_run_and_duplicate_gpu_ids(tmp_path, monkeypatch):
    import training_control as command
    control.write_json(tmp_path / 'run.json', {})
    monkeypatch.setattr(command, 'training_status', lambda path: {'running': True})
    with pytest.raises(RuntimeError, match='still holds the run lock'):
        command.recover(tmp_path, ['0', '2'])
    with pytest.raises(ValueError, match='duplicate'):
        command.recover(tmp_path, ['0', '0'])

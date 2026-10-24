import copy
import csv
import json

import pytest
import torch

from test_experiment import setup
from test_multiscene_runtime import configure, records, perfect
from test_training_control import contract
from medical_kv.cli import _main, validate_test_checkpoint
from medical_kv.config import ROOT, fingerprint
from medical_kv.evaluation import evaluate_records
from medical_kv.io import read_checkpoint, save_checkpoint
from medical_kv.resume import validate_resume


def paused():
    state = {'epoch': 24, 'cursor': 80, 'step': 28707}
    checkpoint = {'state': state, 'run_signature': 'run', 'selected_by_validation': False,
                  'run_info': {'partial_dataset': False}}
    summary = {'state': copy.deepcopy(state), 'run_signature': 'run', 'partial_dataset': False,
               'status': 'partial', 'stop_reason': 'paused'}
    return checkpoint, summary


def test_explicit_interim_mode_preserves_formal_guard():
    checkpoint, summary = paused()
    with pytest.raises(ValueError, match='Formal test'):
        validate_test_checkpoint(checkpoint, summary)
    validate_test_checkpoint(checkpoint, summary, interim=True)
    for change in ({'partial_dataset': True}, {'run_signature': 'other'},
                   {'stop_reason': 'runtime_limit'}, {'status': 'complete'}):
        with pytest.raises(ValueError):
            validate_test_checkpoint(checkpoint, dict(summary, **change), interim=True)
    stale = copy.deepcopy(checkpoint)
    stale['state']['step'] -= 1
    with pytest.raises(ValueError, match='latest paused'):
        validate_test_checkpoint(stale, summary, interim=True)
    checkpoint['selected_by_validation'] = True
    validate_test_checkpoint(checkpoint, dict(summary, status='complete'))


@pytest.mark.parametrize('extra', [['--limit', '10'], ['--development'], ['--tiny'],
                                  ['--record-form', 'dialogue']])
def test_interim_forbids_partial_test_before_loading_model(extra):
    with pytest.raises(SystemExit):
        _main(['evaluate', '--interim-test', '--split', 'test', '--condition', 'soft',
               '--checkpoint', 'unused.pt', *extra])


def test_snapshot_all_questions_csv_and_original_checkpoint_unchanged(setup, tmp_path, monkeypatch):
    backend, cfg, _ = setup
    configure(cfg)
    monkeypatch.setattr(backend, 'generate_many', perfect)
    source = tmp_path / 'latest.pt'
    saved = {'state': {'epoch': 24, 'cursor': 80, 'step': 28707}, 'prompt': torch.ones(3, 32)}
    save_checkpoint(source, saved)
    original = source.read_bytes()
    out = tmp_path / 'interim'
    meta = {'evaluation_stage': 'interim', 'partial_training': True, 'completed_epochs': 24,
            'records_seen_in_current_epoch': 80, 'step': 28707}
    summary, rows = evaluate_records(backend, records(), prompt_ids=backend.initial_ids(),
                                    output=out, metadata=meta, checkpoint_source=source)
    assert summary['status'] == 'complete' and summary['records'] == 3
    assert {r['record_id'] for r in rows} == {r['record_id'] for r in records()}
    assert source.read_bytes() == original == (out / 'checkpoint.pt').read_bytes()
    assert read_checkpoint(source)['state'] == saved['state']
    with (out / 'predictions.csv').open(encoding='utf-8-sig', newline='') as stream:
        csv_rows = list(csv.DictReader(stream))
    assert len(csv_rows) == len(rows)
    assert all(r['evaluation_stage'] == 'interim' and r['completed_epochs'] == '24' for r in csv_rows)
    assert json.loads((out / 'metadata.json').read_text()) == meta
    # A later training write does not change the archived evaluation checkpoint.
    save_checkpoint(source, dict(saved, state={'epoch': 26}))
    assert (out / 'checkpoint.pt').read_bytes() == original
    with pytest.raises(FileExistsError):
        evaluate_records(backend, records(), output=out, checkpoint_source=source)


@pytest.mark.parametrize('source_index', [0, -1])
def test_reviewed_pause_version_can_resume_after_interim_update(source_index):
    target = contract()
    registry = json.loads((ROOT / 'configs/resume_compatibility_v1.json').read_text())
    old = copy.deepcopy(target)
    old['code'] = {key: values[0] for key, values in registry['reviewed_v2_sources'][source_index].items()}
    signature = fingerprint(old)
    checkpoint = {'method': 'soft', 'resume_format': 2, 'run_signature': signature,
                  'run_info': {'method': 'soft', 'run_signature': signature,
                               'origin_identity': old, 'resume_contract': old}}
    assert validate_resume(checkpoint, target) == (old, True)
    changed = copy.deepcopy(target)
    changed['config']['batch_records'] = 9
    with pytest.raises(ValueError, match='config changed'):
        validate_resume(checkpoint, changed)
    changed = copy.deepcopy(target)
    changed['code']['training.py'] = 'unknown'
    with pytest.raises(ValueError, match='reviewed migration'):
        validate_resume(checkpoint, changed)


@pytest.mark.parametrize('scene', [None, 'medical', 'finance', 'hr'])
def test_cli_interim_locks_run_and_reads_all_test_records(setup, tmp_path, monkeypatch, scene):
    from filelock import FileLock, Timeout
    from medical_kv import cli
    from medical_kv.io import inference_identity
    backend, cfg, _ = setup
    configure(cfg)
    cfg['scene'] = scene
    dataset = {'schema': 'multiscene_privacy_v1', 'split_sha256': {'test': 'fixture'},
               'tokenizer_sha256': 'fixture', 'selection': {'scene': scene, 'record_form': None}}
    run = tmp_path / 'training'
    run.mkdir()
    checkpoint, summary = paused()
    checkpoint.update(method='soft', initial_ids=backend.initial_ids(),
                      prompt=backend.embedding(backend.tensor(backend.initial_ids()))[0].detach(),
                      inference_signature=inference_identity(backend))
    checkpoint['run_info'].update(config=cfg, dataset=dataset)
    source = run / 'latest.pt'
    save_checkpoint(source, checkpoint)
    (run / 'run_summary.json').write_text(json.dumps(summary))
    before = source.read_bytes()
    monkeypatch.setattr(cli, 'load_config', lambda *a: cfg)
    monkeypatch.setattr(cli, 'dataset_identity', lambda *a: dataset)
    monkeypatch.setattr(cli.Backend, 'load', lambda *a: backend)
    monkeypatch.setattr(backend, 'generate_many', perfect)
    seen = []
    def load(cfg, split, limit=None):
        assert limit is None
        seen.append(split)
        return [r for r in records() if cfg.get('scene') in (None, r['scene'])]
    monkeypatch.setattr(cli, 'load_split', load)
    real_read = cli.read_checkpoint
    def read(path):
        with pytest.raises(Timeout):
            with FileLock(str(run / '.run.lock'), timeout=0):
                pass
        return real_read(path)
    monkeypatch.setattr(cli, 'read_checkpoint', read)
    out = tmp_path / 'cli_interim'
    arguments = ['evaluate', '--interim-test', '--split', 'test', '--condition', 'soft',
                 '--checkpoint', str(source)] + (['--scene', scene] if scene else [])
    cli._main([*arguments, '--output-directory', str(out)])
    assert seen == ['test', 'train']
    assert source.read_bytes() == before == (out / 'checkpoint.pt').read_bytes()
    meta = json.loads((out / 'metadata.json').read_text())
    assert meta['evaluation_records'] == (1 if scene else 3) and meta['completed_epochs'] == 24
    assert meta['evaluation_scope'] == {'scene': scene, 'record_form': None}
    with FileLock(str(run / '.run.lock'), timeout=0):
        pass  # Evaluation released the lock; training can resume.
    monkeypatch.setattr(backend, 'context', lambda *a, **kw: pytest.fail('Completed records must not be regenerated'))
    recovered = tmp_path / 'resumed_evaluation'
    cli._main(['evaluate', '--resume-evaluation', str(out), '--output-directory', str(recovered)])
    restored_meta = json.loads((recovered / 'metadata.json').read_text())
    assert restored_meta['evaluation_resume']['remaining_records'] == 0
    assert restored_meta['evaluation_resume']['completed_records'] == (1 if scene else 3)
    assert source.read_bytes() == before == (recovered / 'checkpoint.pt').read_bytes()
    if scene:
        wrong = copy.deepcopy(dataset)
        wrong['selection']['scene'] = 'hr' if scene != 'hr' else 'medical'
        monkeypatch.setattr(cli, 'dataset_identity', lambda *a: wrong)
        with pytest.raises(ValueError, match='scene/form selection'):
            cli._main([*arguments, '--output-directory', str(tmp_path / 'wrong_scene')])

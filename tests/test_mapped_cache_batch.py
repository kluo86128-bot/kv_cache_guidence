"""Mixed-length cache batches and reviewed resume of scalar evaluation records."""
import copy
import json
from pathlib import Path

import pytest
import torch

import map_soft_u_cosine as mapping
import evaluate_mapped_u as evaluation
import mapped_cache_batch as batch
from medical_kv.config import fingerprint
from medical_kv.control import write_json
from medical_kv.tasks import questions_for
from test_experiment import setup
from test_multiscene_runtime import configure, records
from test_soft_u_cosine_mapping import tiny_pipeline, inline_launch


@pytest.mark.parametrize('questions_per_wave', [1, 2, 4])
def test_mixed_length_contexts_losses_and_generation_match_scalar(setup, questions_per_wave):
    backend, cfg, _ = setup
    configure(cfg)
    cfg['question_batch_size'] = questions_per_wave
    cfg['split_private_prefill'] = True
    cfg['max_new_tokens'] = 5
    sample = copy.deepcopy(records())
    sample[0]['private_prefix'] += '\n' + 'Extra private text. ' * 5
    sample[1]['private_prefix'] += '\nShort note.'
    ids = backend.initial_ids()
    with torch.no_grad():
        contexts = [backend.context(r, prompt_ids=ids) for r in sample]
        original = [tuple((k.clone(), v.clone()) for k, v in c.layers) for c in contexts]
        assert len({c.next_position for c in contexts}) == 3
        qs = [questions_for(r, cfg, evaluation=True) for r in sample]
        scalar_losses = [backend.answer_losses(c, q) for c, q in zip(contexts, qs)]
        scalar_answers = [backend.generate_many(c, q) for c, q in zip(contexts, qs)]
        actual_losses, actual_answers = batch.score_contexts(backend, contexts, qs)
    for expected, actual in zip(scalar_losses, actual_losses):
        torch.testing.assert_close(torch.tensor(actual), expected.cpu(), atol=2e-6, rtol=1e-5)
    assert actual_answers == scalar_answers
    for context, layers in zip(contexts, original):
        for (ak, av), (ek, ev) in zip(context.layers, layers):
            assert torch.equal(ak, ek) and torch.equal(av, ev)


def test_masks_positions_and_batch_rows_are_independent(setup):
    backend, _, record = setup
    other = copy.deepcopy(record)
    other['private_prefix'] += '\nExtra context.'
    with torch.no_grad():
        contexts = [backend.context(r, prompt_ids=backend.initial_ids()) for r in (record, other)]
    ids, lengths, positions, layers, mask, _, starts = batch.question_inputs(backend, contexts, [0, 1], [[9, 10], [11, 12, 13, 14]])
    assert lengths.tolist() == [2, 4]
    assert positions[0, -2:].tolist() == [contexts[0].next_position, contexts[0].next_position + 1]
    assert positions[1].tolist() == list(range(contexts[1].next_position, contexts[1].next_position + 4))
    physical = contexts[0].physical_length
    assert (mask[0, :, :, physical:physical + 2] < -1e10).all()
    for layer, (k, v) in enumerate(layers):
        assert torch.equal(k[0:1], contexts[0].layers[layer][0])
        assert torch.equal(v[1:2], contexts[1].layers[layer][1])


def test_independent_eos_and_output_limits(setup, monkeypatch):
    backend, cfg, record = setup
    cfg['max_new_tokens'] = 3
    with torch.no_grad():
        context = backend.context(record, prompt_ids=backend.initial_ids())
    observed = []
    def forward(**kwargs):
        observed.append(kwargs['position_ids'].clone())
        logits = torch.zeros(2, 1, backend.embedding.num_embeddings)
        logits[0, 0, backend.eos] = 10
        logits[1, 0, ord('X') + 3] = 10
        return logits, kwargs['layers']
    monkeypatch.setattr(backend, 'forward', forward)
    qs = [record['qa'][0], dict(record['qa'][0], question='A much longer question?')]
    outputs = batch.generate(backend, [context], [0, 0], qs)
    assert outputs[0][0] == '' and outputs[0][1]['stop_reason'] == 'eos'
    assert outputs[1][0] == 'XXX' and outputs[1][1]['truncated']
    assert len(observed) == 3


def test_worker_prefills_once_per_record_and_commits_tail_batch(setup, tmp_path, monkeypatch):
    backend, cfg, _ = setup
    configure(cfg)
    samples = records()
    directory = tmp_path / 'records'
    directory.mkdir()
    calls = []
    context = backend.context
    monkeypatch.setattr(backend, 'context', lambda r, **kw: calls.append(r['record_id']) or context(r, **kw))
    evaluation.evaluate_worker_records(backend, samples, [0, 1, 2], backend.initial_ids(), directory, 'fixture', tmp_path,
                                       batch_records_per_gpu=2)
    assert calls == [r['record_id'] for r in samples]
    saved = evaluation.committed_records(directory, samples, cfg, 'fixture')
    assert len(saved) == 3
    last = mapping.read_json(directory / '00000002.json')
    assert last['execution']['actual_batch_records'] == 1
    for i, record in enumerate(samples):
        _, expected = evaluation.evaluate_records(backend, [record], prompt_ids=backend.initial_ids(),
                                                  seen_birthdays=set(), seen_values={})
        for actual, reference in zip(saved[i], expected):
            assert actual.keys() == reference.keys()
            assert actual['prediction'] == reference['prediction']
            assert actual['em'] == reference['em']
            assert actual['nll'] == pytest.approx(reference['nll'], abs=2e-6)


def test_pause_during_prefill_does_not_commit_incomplete_batch(setup, tmp_path, monkeypatch):
    backend, cfg, _ = setup
    configure(cfg)
    directory = tmp_path / 'records'
    directory.mkdir()
    context = backend.context
    def paused(r, **kw):
        result = context(r, **kw)
        (tmp_path / 'pause.request').touch()
        return result
    monkeypatch.setattr(backend, 'context', paused)
    evaluation.evaluate_worker_records(backend, records(), [0, 1], backend.initial_ids(), directory, 's', tmp_path,
                                       batch_records_per_gpu=2)
    assert not list(directory.glob('*.json'))


def legacy_codes(current):
    policy = mapping.read_json(mapping.ROOT / 'configs/mapped_evaluation_compatibility_v1.json')
    immutable = {k: v for k, v in current.items() if k.startswith('medical_kv/')}
    return {**immutable, **policy['source_scripts']}


def test_reviewed_upgrade_preserves_old_artifacts_and_records(tiny_pipeline, monkeypatch):
    path, _, _ = tiny_pipeline
    cfg = mapping.load_settings(path)
    original_codes = mapping.code_hashes
    old = legacy_codes(original_codes())
    monkeypatch.setattr(mapping, 'code_hashes', lambda: old)
    monkeypatch.setattr(evaluation, 'code_hashes', lambda: old)
    inline_launch(monkeypatch)
    worker = evaluation.evaluate_worker_records
    def interrupt(backend, samples, assigned, ids, directory, signature, root, seen=None, **kwargs):
        worker(backend, samples, assigned[:1], ids, directory, signature, root, seen, batch_records_per_gpu=1)
        raise RuntimeError('simulated old worker pause')
    monkeypatch.setattr(evaluation, 'evaluate_worker_records', interrupt)
    assert mapping.run_pipeline(cfg)['status'] == 'failed'
    out = Path(cfg['output_directory']) / 'fixture'
    preserved_paths = [out / name for name in ('job.json', 'mapped_prompt.json', 'mapped_prompt.txt',
                       'mapping_complete.json', 'source_soft.pt', 'evaluation/metadata.json',
                       'evaluation/records/00000000.json')]
    original_bytes = {p: p.read_bytes() for p in preserved_paths}
    monkeypatch.setattr(mapping, 'code_hashes', original_codes)
    monkeypatch.setattr(evaluation, 'code_hashes', original_codes)
    monkeypatch.setattr(evaluation, 'evaluate_worker_records', worker)
    cfg['evaluation']['batch_records_per_gpu'] = 2
    assert mapping.run_pipeline(cfg, 'resume')['status'] == 'complete'
    assert {p: p.read_bytes() for p in preserved_paths} == original_bytes
    summary = mapping.read_json(out / 'evaluation/summary.json')
    assert summary['records'] == 3 and summary['execution_batches'] == [1, 2]
    assert summary['runtime_upgrades']


def test_unreviewed_changes_still_rejected():
    current = mapping.code_hashes()
    old = legacy_codes(current)
    assert mapping.compatible_execution_code(old, current)
    changed = dict(old, **{'medical_kv/backend.py': 'wrong'})
    assert not mapping.compatible_execution_code(changed, current)
    changed = dict(old, **{'evaluate_mapped_u.py': 'unreviewed'})
    assert not mapping.compatible_execution_code(changed, current)
    changed = dict(current, **{'mapped_cache_batch.py': 'modified'})
    assert not mapping.compatible_execution_code(old, changed)


def test_over_thousand_committed_records_reuse_with_three_new_shards(setup, tmp_path, monkeypatch):
    backend, cfg, record = setup
    directory = tmp_path / 'records'
    directory.mkdir()
    ids = backend.initial_ids()
    _, template = evaluation.evaluate_records(backend, [record], prompt_ids=ids,
                                               seen_birthdays=set(), seen_values={})
    samples = []
    for index in range(1009):
        item = copy.deepcopy(record)
        item['record_id'] = f'sample_{index}'
        samples.append(item)
        if index < 1003:
            rows = [dict(row, record_id=item['record_id']) for row in template]
            write_json(directory / f'{index:08d}.json', {'signature': 'preserved', 'index': index,
                       'rows_sha256': fingerprint(rows), 'rows': rows})
    original = {p.name: p.read_bytes() for p in directory.glob('*.json')}
    committed = evaluation.committed_records(directory, samples, cfg, 'preserved')
    assert len(committed) == 1003
    pending = [i for i in range(len(samples)) if i not in committed]
    observed = []
    context = backend.context
    monkeypatch.setattr(backend, 'context', lambda r, **kw: observed.append(r['record_id']) or context(r, **kw))
    for rank in range(3):
        evaluation.evaluate_worker_records(backend, samples, pending[rank::3], ids, directory,
            'preserved', tmp_path, batch_records_per_gpu=2)
    assert sorted(observed) == sorted(r['record_id'] for r in samples[1003:])
    assert all((directory / name).read_bytes() == value for name, value in original.items())
    assert len(evaluation.committed_records(directory, samples, cfg, 'preserved')) == 1009


def test_real_subprocess_cache_batch_pipeline(tiny_pipeline):
    import subprocess
    import sys
    path, _, _ = tiny_pipeline
    result = subprocess.run([sys.executable, str(mapping.ROOT / 'map_soft_u_cosine.py'), '--config', str(path),
                             '--batch-records-per-gpu', '2'], cwd=mapping.ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    cfg = mapping.load_settings(path)
    out = Path(cfg['output_directory']) / 'fixture'
    summary = mapping.read_json(out / 'evaluation/summary.json')
    assert summary['execution_batches'] == [2] and summary['records'] == 3

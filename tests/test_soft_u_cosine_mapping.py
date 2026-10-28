"""Projection math, durable progress and actual CPU tiny-Qwen integration."""
import copy
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from safetensors.torch import save_file

import map_soft_u_cosine as mapping
import evaluate_mapped_u as mapped_eval
from medical_kv.config import FIELDS, file_hash, fingerprint, load_config
from medical_kv.control import write_json
from medical_kv.data import dataset_identity
from medical_kv.io import inference_identity, save_checkpoint
from test_experiment import setup
from test_multiscene_runtime import configure, perfect, records as multiscene_records


class ArraySource:
    def __init__(self, values):
        self.values = values
        self.shape = list(values.shape)

    def chunks(self, size):
        for start in range(0, len(self.values), size):
            yield start, self.values[start:start + size]


def test_cosine_matches_dense_and_is_independent_of_vocab_partition():
    torch.manual_seed(81)
    values = torch.randn(29, 7)
    prompt = torch.randn(5, 7) * 8
    allowed = torch.tensor([1, 2, 7, 9, 12, 18, 21, 28])
    source = ArraySource(values)
    expected = torch.nn.functional.normalize(prompt) @ torch.nn.functional.normalize(values[allowed]).T
    for size in (1, 4, 100):
        ids, scores = mapping.cosine_nearest(prompt, source, allowed, size)
        assert ids == allowed[expected.argmax(dim=1)].tolist()
        torch.testing.assert_close(torch.tensor(scores), expected.max(dim=1).values)


def test_ties_masks_zero_and_nonfinite_embeddings():
    values = torch.tensor([[1., 0.], [10., 0.], [0., 1.], [0., 0.], [float('nan'), 1.]])
    ids, scores = mapping.cosine_nearest(torch.tensor([[2., 0.]]), ArraySource(values), torch.arange(5), 1)
    assert ids == [0] and scores == [1.]
    ids, _ = mapping.cosine_nearest(torch.tensor([[2., 0.]]), ArraySource(values), torch.tensor([1, 2]), 3)
    assert ids == [1]
    with pytest.raises(ValueError, match='zero/non-finite'):
        mapping.cosine_nearest(torch.zeros(1, 2), ArraySource(values), torch.arange(5), 2)
    assert mapping.cosine_nearest(torch.ones(1, 2), ArraySource(values), torch.arange(5), 2, stop=lambda: True) is None


def test_embedding_loader_reads_only_input_embedding_shard(tmp_path):
    model = tmp_path / 'model'
    model.mkdir()
    values = torch.randn(12, 4).bfloat16()
    save_file({'model.embed_tokens.weight': values}, model / 'embedding.safetensors')
    write_json(model / 'model.safetensors.index.json', {'weight_map': {
        'model.embed_tokens.weight': 'embedding.safetensors', 'model.layers.0.weight': 'DOES_NOT_EXIST.safetensors'}})
    write_json(model / 'config.json', {'hidden_size': 4})
    source = mapping.EmbeddingSource(model)
    assert source.shape == [12, 4]
    torch.testing.assert_close(torch.cat([b for _, b in source.chunks(5)]), values)
    assert len(source.identity()['embedding_sha256']) == 64


def test_permitted_tokens_keep_unicode_and_exclude_specials(setup):
    backend, _, _ = setup
    settings = {'exclude_special_tokens': True, 'allow_non_ascii': True}
    allowed = mapping.permitted_tokens(backend.tokenizer, backend.embedding.num_embeddings, settings)
    assert not set(allowed.tolist()).intersection(backend.tokenizer.all_special_ids)
    assert 35 in allowed  # ordinary whitespace is not silently removed


def test_unstable_projection_is_saved_but_not_reported_as_usable(tmp_path, setup):
    backend, _, _ = setup
    job = {'signature': 's', 'source_sha256': 'source', 'prompt_shape': [2, 32], 'contract': {'name': 'fixture'}}
    (tmp_path / 'positions').mkdir()
    # TinyTokenizer merges the characters " a" into token 130 on re-encoding.
    for index, token in enumerate([ord(' ') + 3, ord('a') + 3]):
        write_json(tmp_path / 'positions' / f'{index:06d}.json', {
            'job_signature': 's', 'position': index, 'token_id': token, 'cosine_similarity': .9})
    write_json(tmp_path / 'mapping_model.json', {})
    artifact = mapping.finish_mapping(tmp_path, job, backend.tokenizer)
    assert artifact['status'] == 'invalid_text' and not artifact['decode_encode_verified']
    assert mapping.verify_marker(tmp_path / 'mapping_complete.json')['usable_text'] is False
    assert (tmp_path / 'mapped_prompt.txt').read_bytes() == b' a'


def test_mapping_corrupt_progress_and_changed_complete_output_fail(tmp_path):
    (tmp_path / 'positions').mkdir()
    job = {'signature': 's', 'prompt_shape': [1, 2]}
    write_json(tmp_path / 'positions/000000.json', {'job_signature': 'wrong', 'position': 0,
                                                 'token_id': 1, 'cosine_similarity': 1})
    with pytest.raises(ValueError, match='Invalid mapping progress'):
        mapping.collect_positions(tmp_path, job)
    path = tmp_path / 'output.txt'
    path.write_text('original')
    mapping.marker(tmp_path / 'done.json', [path])
    path.write_text('changed')
    with pytest.raises(ValueError, match='Completed output changed'):
        mapping.verify_marker(tmp_path / 'done.json')


@pytest.fixture
def tiny_pipeline(tmp_path):
    """A real saved tiny model/tokenizer and synthetic DEV data, not formal scores."""
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
    from medical_kv.backend import Backend
    torch.set_num_threads(2)
    torch.manual_seed(91)
    model_dir = tmp_path / 'tiny_model'
    model_dir.mkdir()
    vocab = {'<unk>': 0, '<pad>': 1, '<eos>': 2}
    vocab.update({v: i + 3 for i, v in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))})
    engine = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token='<unk>'))
    engine.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    engine.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=engine, unk_token='<unk>',
                                       pad_token='<pad>', eos_token='<eos>')
    tokenizer.save_pretrained(model_dir)
    model = Qwen3ForCausalLM(Qwen3Config(vocab_size=len(tokenizer), hidden_size=32,
        intermediate_size=48, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=1, head_dim=16, eos_token_id=2, pad_token_id=1,
        max_position_embeddings=2048, tie_word_embeddings=False))
    model.save_pretrained(model_dir)
    data = tmp_path / 'data'
    data.mkdir()
    token_sha = file_hash(model_dir / 'tokenizer.json')
    for split in ('dev', 'train', 'validation', 'test'):
        rows = []
        for index in range(3):
            fields = dict(zip(FIELDS, ['Ax Bo', '2001-01-02', '3 Oak', 'Cold']))
            rows.append({'record_id': f'{split}_{index}', 'group_id': f'{split}_{index}',
                'name_format': 'natural_no_numeric_suffix_v1', 'split': split,
                'private_prefix': '\n'.join(f'{k}: {v}' for k, v in fields.items()),
                'fields': fields, 'source_name_fields': {'FIRST': 'Ax', 'MIDDLE': '', 'LAST': 'Bo'},
                'qa': [{'field': k, 'question': k + '?', 'answer': v} for k, v in fields.items()]})
        mapping.atomic_text(data / f'{split}.jsonl', ''.join(json.dumps(r) + '\n' for r in rows))
    write_json(data / 'manifest.json', {'schema_version': 'source_preserving_v3',
        'name_format': 'natural_no_numeric_suffix_v1', 'dataset_id': 'implementation-only',
        'target_fields': list(FIELDS), 'tokenizer': {'sha256': token_sha}})
    write_json(data / 'validation_report.json', {'status': 'PASS'})
    write_json(data / 'artifact_hashes.json', {p.name: file_hash(p) for p in data.iterdir()})
    cfg = load_config('configs/experiment_v1.json', {'device': 'cpu', 'dtype': 'float32',
        'model_name_or_path': str(model_dir), 'dataset_directory': str(data),
        'prompt_length': 3, 'max_new_tokens': 2, 'question_format': 'plain', 'public_text': 'Public.'})
    backend = Backend.load(cfg, token_sha)
    source = tmp_path / 'soft_run'
    source.mkdir()
    prompt = backend.embedding(backend.tensor(tokenizer.encode('XYZ', add_special_tokens=False)))[0].detach().clone()
    checkpoint = {'method': 'soft', 'prompt': prompt, 'run_signature': 'implementation-fixture',
        'inference_signature': inference_identity(backend), 'selected_by_validation': True,
        'state': {'epoch': 2, 'step': 3, 'cursor': 0},
        'run_info': {'config': cfg, 'backend': backend.identity,
                     'dataset': dataset_identity(cfg), 'partial_dataset': False}}
    save_checkpoint(source / 'best.pt', checkpoint)
    evaluation = tmp_path / 'evaluation.json'
    write_json(evaluation, cfg)
    configuration = tmp_path / 'mapping.json'
    write_json(configuration, {'output_directory': str(tmp_path / 'mapped'),
        'mapping': {'vocabulary_chunk_size': 71, 'position_chunk_size': 1},
        'evaluation': {'enabled': True, 'split': 'dev', 'gpus': [], 'batch_records_per_gpu': 1},
        'soft_prefixes': [{'name': 'fixture', 'directory': str(source), 'evaluation_config': str(evaluation)}]})
    return configuration, backend, checkpoint


def inline_launch(monkeypatch, ranks=1, after_worker=None):
    def launch(script, out, root, settings, devices, stage):
        for rank in range(ranks):
            if script == 'map_soft_u_cosine.py':
                mapping.mapping_worker(out, root, settings, rank, ranks, 'cpu')
            else:
                mapped_eval.evaluation_worker(out, root, settings, rank, ranks, 'cpu')
            if after_worker:
                after_worker(stage, rank, out, root)
    monkeypatch.setattr(mapping, 'launch_workers', launch)
    monkeypatch.setattr(mapped_eval, 'launch_workers', launch)


def test_mapping_resume_repartitions_immutable_pending_positions(tiny_pipeline, monkeypatch):
    path, _, checkpoint = tiny_pipeline
    cfg = mapping.load_settings(path)
    root = Path(cfg['output_directory'])
    root.mkdir()
    out, job = mapping.prepare_job(cfg, cfg['soft_prefixes'][0])
    from map_soft_u_cosine import EmbeddingSource, tokenizer_for
    src = EmbeddingSource(job['contract']['evaluation_config']['model_name_or_path'])
    tokenizer, canonical = tokenizer_for(str(src.directory))
    write_json(out / 'mapping_model.json', {**src.identity(), 'tokenizer_canonical_sha256': canonical})
    settings = dict(cfg['mapping'], pending_positions=[0, 1, 2])
    # First rank commits positions 0/2 before rank 1 starts: its slice stays [1].
    mapping.mapping_worker(out, root, settings, rank=0, world=2)
    assert sorted(mapping.collect_positions(out, job)) == [0, 2]
    (root / 'pause.request').touch()
    mapping.mapping_worker(out, root, settings, rank=1, world=2)
    (root / 'pause.request').unlink()
    # Resume on three logical workers: only one unfinished position exists.
    settings['pending_positions'] = [1]
    for rank in range(3):
        mapping.mapping_worker(out, root, settings, rank, 3)
    artifact = mapping.finish_mapping(out, job, tokenizer)
    assert artifact['text'] == 'XYZ' and artifact['decode_encode_verified']
    assert checkpoint['prompt'].equal(mapping.read_checkpoint(out / 'source_soft.pt')['prompt'])


def test_tiny_pipeline_auto_evaluates_and_repeat_resume_skips_work(tiny_pipeline, monkeypatch):
    path, _, _ = tiny_pipeline
    inline_launch(monkeypatch, ranks=2)
    cfg = mapping.load_settings(path)
    result = mapping.run_pipeline(cfg)
    assert result['status'] == 'complete', result
    root = Path(cfg['output_directory'])
    out = root / 'fixture'
    assert mapping.read_json(out / 'mapped_prompt.json')['text'] == 'XYZ'
    summary = mapping.read_json(out / 'evaluation/summary.json')
    assert summary['records'] == 3
    assert summary['metadata']['condition'] == 'mapped_cosine'
    with (out / 'evaluation/predictions.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 12 and all(r['mapping_name'] == 'fixture' for r in rows)
    assert len({(r['record_id'], r['field']) for r in rows}) == 12
    original = (out / 'evaluation/predictions.csv').read_bytes()
    monkeypatch.setattr(mapping, 'launch_workers', lambda *a, **k: pytest.fail('completed mapping rerun'))
    monkeypatch.setattr(mapped_eval, 'launch_workers', lambda *a, **k: pytest.fail('completed evaluation rerun'))
    cfg['mapping'].update(gpus=[7], device='cuda', vocabulary_chunk_size=13)
    cfg['evaluation']['gpus'] = [4, 6, 7]
    assert mapping.run_pipeline(cfg, 'resume')['status'] == 'complete'
    assert (out / 'evaluation/predictions.csv').read_bytes() == original


def test_changed_soft_checkpoint_refuses_old_outputs(tiny_pipeline, monkeypatch):
    path, _, _ = tiny_pipeline
    cfg = mapping.load_settings(path)
    inline_launch(monkeypatch)
    cfg['evaluation']['enabled'] = False
    assert mapping.run_pipeline(cfg)['status'] == 'complete'
    source = Path(cfg['soft_prefixes'][0]['directory']) / 'best.pt'
    changed = mapping.read_checkpoint(source)
    changed['prompt'][0, 0] += 1
    save_checkpoint(source, changed)
    result = mapping.run_pipeline(cfg, 'resume')
    assert result['status'] == 'failed' and 'source checkpoint changed' in result['failures'][0]['error']


def test_mapping_only_never_reads_dataset_or_loads_transformer(tiny_pipeline, monkeypatch):
    path, _, _ = tiny_pipeline
    import medical_kv.data as data
    from transformers import AutoModelForCausalLM
    def forbidden(*args, **kwargs):
        pytest.fail('data-free mapping attempted dataset access or transformer loading')
    monkeypatch.setattr(data, 'dataset_identity', forbidden)
    monkeypatch.setattr(data, 'load_split', forbidden)
    monkeypatch.setattr(AutoModelForCausalLM, 'from_pretrained', forbidden)
    cfg = mapping.load_settings(path)
    inline_launch(monkeypatch)
    assert mapping.run_pipeline(cfg, 'map')['status'] == 'complete'


def test_process_identity_and_other_host_are_distinguished():
    identity = {'pid': os.getpid(), 'start_token': mapping.process_token(os.getpid())}
    assert identity['start_token'] and mapping.alive(identity)
    assert not mapping.alive(dict(identity, host='definitely-a-different-host'))


def test_changed_model_and_generation_are_rejected(tiny_pipeline, monkeypatch):
    path, _, _ = tiny_pipeline
    cfg = mapping.load_settings(path)
    inline_launch(monkeypatch)
    assert mapping.run_pipeline(cfg)['status'] == 'complete'
    ev = Path(cfg['soft_prefixes'][0]['evaluation_config'])
    changed = mapping.read_json(ev)
    changed['max_new_tokens'] += 1
    write_json(ev, changed)
    result = mapping.run_pipeline(cfg, 'resume')
    assert result['status'] == 'failed' and 'job identity changed' in result['failures'][0]['error']


def test_changed_embedding_refuses_completed_mapping(tiny_pipeline, monkeypatch):
    path, backend, _ = tiny_pipeline
    cfg = mapping.load_settings(path)
    cfg['evaluation']['enabled'] = False
    inline_launch(monkeypatch)
    assert mapping.run_pipeline(cfg)['status'] == 'complete'
    source = mapping.EmbeddingSource(backend.cfg['model_name_or_path'])
    from safetensors.torch import load_file
    weights = load_file(source.path)
    weights[source.key][0, 0] += 1
    save_file(weights, source.path)
    result = mapping.run_pipeline(cfg, 'resume')
    assert result['status'] == 'failed' and 'Mapping model changed' in result['failures'][0]['error']


def test_pipeline_evaluation_resume_after_failure_reuses_records_on_new_topology(tiny_pipeline, monkeypatch):
    path, _, _ = tiny_pipeline
    cfg = mapping.load_settings(path)
    inline_launch(monkeypatch)
    worker = mapped_eval.evaluate_worker_records
    def interrupt(backend, records, assigned, ids, directory, signature, root, seen=None, **kwargs):
        worker(backend, records, assigned[:1], ids, directory, signature, root, seen, **kwargs)
        raise RuntimeError('simulated worker interruption')
    monkeypatch.setattr(mapped_eval, 'evaluate_worker_records', interrupt)
    assert mapping.run_pipeline(cfg)['status'] == 'failed'
    root = Path(cfg['output_directory'])
    record = root / 'fixture/evaluation/records/00000000.json'
    original = record.read_bytes()
    assert mapping.status(root)['jobs'][0]['evaluation']['completed_records'] == 1
    monkeypatch.setattr(mapped_eval, 'evaluate_worker_records', worker)
    inline_launch(monkeypatch, ranks=3)
    cfg['evaluation']['gpus'] = [3, 5, 6]
    assert mapping.run_pipeline(cfg, 'resume')['status'] == 'complete'
    assert record.read_bytes() == original
    assert len(list(record.parent.glob('*.json'))) == 3


def test_pause_request_commits_only_complete_record_and_resumes(tiny_pipeline, monkeypatch):
    path, _, _ = tiny_pipeline
    cfg = mapping.load_settings(path)
    worker = mapped_eval.evaluate_worker_records
    def paused(backend, records, assigned, ids, directory, signature, root, seen=None, **kwargs):
        worker(backend, records, assigned[:1], ids, directory, signature, root, seen, **kwargs)
        (Path(root) / 'pause.request').touch()
        worker(backend, records, assigned[1:], ids, directory, signature, root, seen, **kwargs)
    inline_launch(monkeypatch)
    monkeypatch.setattr(mapped_eval, 'evaluate_worker_records', paused)
    assert mapping.run_pipeline(cfg)['status'] == 'paused'
    root = Path(cfg['output_directory'])
    assert mapping.pause(root, 0)['status'] == 'paused'
    monkeypatch.setattr(mapped_eval, 'evaluate_worker_records', worker)
    inline_launch(monkeypatch, ranks=2)
    assert mapping.run_pipeline(cfg, 'resume')['status'] == 'complete'


def test_generation_boundary_checks_fail_before_forward(tiny_pipeline):
    _, backend, _ = tiny_pipeline
    tokenizer = backend.tokenizer
    class Unstable:
        def encode(self, text, **kwargs):
            values = tokenizer.encode(text, **kwargs)
            return values + [7] if 'XYZ' in text and 'Public.' in text else values
    cfg = backend.cfg
    ids = tokenizer.encode('XYZ', add_special_tokens=False)
    with pytest.raises(ValueError, match='context token boundaries'):
        mapped_eval.validate_boundaries(Unstable(), [{'record_id': 'r', 'private_prefix': 'private'}], cfg, ids, 'XYZ')


@pytest.mark.parametrize('complete', [False, True])
def test_formal_test_requires_completed_source_run_without_running_test_inference(tiny_pipeline, monkeypatch, complete):
    path, _, _ = tiny_pipeline
    cfg = mapping.load_settings(path)
    cfg['evaluation']['split'] = 'test'
    cfg['evaluation']['enabled'] = False
    source = Path(cfg['soft_prefixes'][0]['directory'])
    write_json(source / 'run_summary.json', {'status': 'complete' if complete else 'partial',
        'partial_dataset': False, 'run_signature': 'implementation-fixture'})
    inline_launch(monkeypatch)
    assert mapping.run_pipeline(cfg, 'map')['status'] == 'complete'
    out = Path(cfg['output_directory']) / 'fixture'
    job = mapping.read_json(out / 'job.json')
    # Only CPU metadata/preflight validation; no formal test model forward.
    if complete:
        directory, _, records, _ = mapped_eval.prepare_evaluation(out, job, cfg)
        assert len(records) == 3 and not list((directory / 'records').glob('*.json'))
    else:
        with pytest.raises(ValueError, match='complete full-data run'):
            mapped_eval.prepare_evaluation(out, job, cfg)


def latest_fixture(checkpoint, status='partial'):
    frozen = copy.deepcopy(checkpoint)
    frozen['selected_by_validation'] = False
    frozen['source_training_summary'] = {
        'status': status, 'stop_reason': 'paused', 'partial_dataset': False,
        'run_signature': frozen['run_signature'], 'state': copy.deepcopy(frozen['state'])}
    return frozen


@pytest.mark.parametrize('status,expected', [('partial', 'interim'), ('complete', 'completed_latest')])
def test_latest_policy_allows_paused_or_completed_unselected_snapshot(tiny_pipeline, status, expected):
    _, _, checkpoint = tiny_pipeline
    frozen = latest_fixture(checkpoint, status)
    result = mapping.validate_test_source(frozen, 'paused_or_complete_latest')
    assert result == {'source_stage': expected, 'partial_training': status == 'partial'}
    with pytest.raises(ValueError, match='validation-selected'):
        mapping.validate_test_source(frozen, 'validation_selected')


@pytest.mark.parametrize('change', ['running', 'stale_progress', 'partial_data', 'wrong_signature', 'missing_summary'])
def test_latest_policy_rejects_unsafe_or_mismatched_sources(tiny_pipeline, change):
    _, _, checkpoint = tiny_pipeline
    frozen = latest_fixture(checkpoint)
    if change == 'running':
        frozen['source_training_summary']['stop_reason'] = 'time_limit'
    elif change == 'stale_progress':
        frozen['source_training_summary']['state']['step'] += 1
    elif change == 'partial_data':
        frozen['run_info']['partial_dataset'] = True
    elif change == 'wrong_signature':
        frozen['source_training_summary']['run_signature'] = 'different'
    else:
        frozen['source_training_summary'] = None
    with pytest.raises(ValueError):
        mapping.validate_test_source(frozen, 'paused_or_complete_latest')


def test_paused_latest_mapping_and_interim_metadata_without_test_inference(tiny_pipeline, monkeypatch):
    path, _, checkpoint = tiny_pipeline
    cfg = mapping.load_settings(path)
    cfg['evaluation'].update(split='test', enabled=False)
    entry = cfg['soft_prefixes'][0]
    entry.update(checkpoint_filename='latest.pt', checkpoint_policy='paused_or_complete_latest')
    source = Path(entry['directory'])
    frozen = latest_fixture(checkpoint)
    summary = frozen.pop('source_training_summary')
    save_checkpoint(source / 'latest.pt', frozen)
    write_json(source / 'run_summary.json', summary)
    inline_launch(monkeypatch)
    assert mapping.run_pipeline(cfg, 'map')['status'] == 'complete'
    out = Path(cfg['output_directory']) / entry['name']
    _, _, _, descriptor = mapped_eval.prepare_evaluation(out, mapping.read_json(out / 'job.json'), cfg)
    metadata = descriptor['metadata']
    assert metadata['source_stage'] == 'interim' and metadata['partial_training'] is True
    assert metadata['evaluation_stage'] == 'mapped_cosine_interim'
    assert not metadata['source_checkpoint_selected_by_validation']
    assert not list((out / 'evaluation/records').glob('*.json'))


def test_evaluation_record_restart_multiscene_matches_uninterrupted(setup, tmp_path, monkeypatch):
    backend, cfg, _ = setup
    configure(cfg)
    cfg['split_private_prefill'] = True
    records = multiscene_records()
    monkeypatch.setattr(backend, 'generate_many', perfect)
    directory = tmp_path / 'records'
    directory.mkdir()
    ids = backend.initial_ids()
    # Stop after a whole committed record, then change to three worker slices.
    mapped_eval.evaluate_worker_records(backend, records, [0], ids, directory, 's', tmp_path)
    completed = mapped_eval.committed_records(directory, records, cfg, 's')
    assert list(completed) == [0]
    pending = [i for i in range(len(records)) if i not in completed]
    for rank in range(3):
        mapped_eval.evaluate_worker_records(backend, records, pending[rank::3], ids, directory, 's', tmp_path)
    descriptor = {'signature': 's', 'metadata': {'mapping_name': 'fixture', 'mapping_artifact_sha256': 'digest'}}
    result = mapped_eval.finalize_evaluation(tmp_path, cfg, records, descriptor)
    assert result['records'] == 3 and result['macro_em'] == 1
    saved = [json.loads(line) for line in (tmp_path / 'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    from medical_kv.evaluation import evaluate_records
    _, expected = evaluate_records(backend, records, prompt_ids=ids, seen_birthdays=set(), seen_values={})
    assert saved == expected


def test_partial_record_never_committed_and_corruption_refused(setup, tmp_path, monkeypatch):
    backend, cfg, record = setup
    directory = tmp_path / 'records'
    directory.mkdir()
    monkeypatch.setattr(backend, 'generate_many', lambda ctx, qs: (_ for _ in ()).throw(RuntimeError('power loss')))
    with pytest.raises(RuntimeError, match='power loss'):
        mapped_eval.evaluate_worker_records(backend, [record], [0], backend.initial_ids(), directory, 's', tmp_path)
    assert not list(directory.glob('*.json'))
    (directory / '00000000.json.temporary.tmp').write_bytes(b'{"rows":')
    assert mapped_eval.committed_records(directory, [record], cfg, 's') == {}
    (directory / '00000000.json').write_bytes(b'broken')
    with pytest.raises(json.JSONDecodeError):
        mapped_eval.committed_records(directory, [record], cfg, 's')


def test_real_subprocess_workers_tiny_cpu(tiny_pipeline):
    path, _, _ = tiny_pipeline
    result = subprocess.run([sys.executable, str(mapping.ROOT / 'map_soft_u_cosine.py'),
                             '--config', str(path)], cwd=mapping.ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    cfg = mapping.load_settings(path)
    out = Path(cfg['output_directory']) / 'fixture'
    assert mapping.verify_marker(out / 'evaluation/complete.json')
    assert not mapping.status(cfg['output_directory'])['running']

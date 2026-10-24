import copy
from contextlib import contextmanager
import json
import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from medical_kv.backend import Backend
from medical_kv.cache import Context, causal_mask, slice_layers
from medical_kv.config import load_config
from medical_kv.data import dataset_identity, load_split
from medical_kv.evaluation import paired_compare, summarize
from medical_kv.io import read_checkpoint


class TinyTokenizer:
    """ASCII plus three space-prefixed tokens, with deterministic reversible IDs."""
    eos_token_id = 2
    all_special_ids = [0, 1, 2]
    chat_template = 'tiny-ascii-test-template'
    words = {' a': 130, ' b': 131, ' c': 132}

    def encode(self, text, add_special_tokens=False):
        output = []
        i = 0
        while i < len(text):
            pair = text[i:i + 2]
            if pair in self.words:
                output.append(self.words[pair]); i += 2
            else:
                output.append(ord(text[i]) + 3); i += 1
        return output

    def decode(self, ids, **kwargs):
        reverse = {v: k for k, v in self.words.items()}
        return ''.join(reverse.get(i, chr(i - 3) if i >= 3 else '<eos>') for i in ids)

    def get_vocab(self):
        return {**{chr(i): i + 3 for i in range(127)}, **self.words}


@pytest.fixture
def setup(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(7)
    cfg = load_config('configs/experiment_v1.json', {'device': 'cpu', 'dtype': 'float32',
                      'output_directory': str(tmp_path / 'run'), 'epochs': 2, 'max_new_tokens': 2})
    cfg.update({'prompt_length': 3, 'public_text': 'Public clinic.', 'question_format': 'plain',
                'initialization_words': ['a', 'b', 'c'], 'batch_records': 1, 'warmup_fraction': 0})
    cfg['hard'].update({'steps': 2, 'gradient_records': 1, 'candidate_records': 1,
                        'candidate_count': 2, 'top_k': 3, 'validate_every': 1})
    model = Qwen3ForCausalLM(Qwen3Config(vocab_size=136, hidden_size=32, intermediate_size=64,
            num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=1, head_dim=16,
            initializer_range=.12, eos_token_id=2, max_position_embeddings=2048))
    model.set_attn_implementation('eager')
    backend = Backend(model, TinyTokenizer(), cfg)
    fields = {'name': 'Ax', 'birth_date': '2001-01-01', 'address': '3 Oak', 'diagnosis': 'Cold'}
    record = {'record_id': 'toy_1', 'private_prefix': 'Name: Ax\nBirth: 2001-01-01\nAddress: 3 Oak\nDiagnosis: Cold',
              'fields': fields, 'qa': [{'field': k, 'question': k + '?', 'answer': v} for k, v in fields.items()]}
    return backend, cfg, record


def test_public_cache_equals_dense_mask_logits_and_gradient(setup):
    b, cfg, r = setup
    u = b.embedding(b.tensor(b.initial_ids()))[0].detach().clone().requires_grad_(True)
    ctx = b.context(r, vectors=u)
    two_stage = b.answer_loss(ctx, r['qa'][0])
    two_stage_grad = torch.autograd.grad(two_stage, u)[0]
    u2 = u.detach().clone().requires_grad_(True)
    embeddings, start = b.layout(r, vectors=u2)
    q, target = b.question_and_target(r['qa'][0])
    tail = q + target[:-1]
    n = embeddings.shape[1]
    full = torch.cat((embeddings, b.embedding(b.tensor(tail))), dim=1)
    mask = causal_mask(full.shape[1], 0, device=b.device, dtype=b.dtype)
    mask[:, :, n:, :start] = torch.finfo(b.dtype).min
    logits, _ = b.forward(embeddings=full, mask=mask,
                          logits_slice=slice(n + len(q) - 1, n + len(q) - 1 + len(target)))
    dense_loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), b.tensor(target).reshape(-1))
    dense_grad = torch.autograd.grad(dense_loss, u2)[0]
    torch.testing.assert_close(two_stage, dense_loss, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(two_stage_grad, dense_grad, rtol=2e-4, atol=2e-6)
    assert float(two_stage_grad.norm()) > 0
    assert ctx.physical_length == len(b.public_ids)
    assert all(p.grad is None for p in b.model.parameters())


def test_gradient_finite_difference_and_detach(setup):
    b, cfg, r = setup
    u = b.embedding(b.tensor(b.initial_ids()))[0].detach().clone().requires_grad_(True)
    ctx = b.context(r, vectors=u)
    loss = b.answer_loss(ctx, r['qa'][0])
    gradient = torch.autograd.grad(loss, u)[0]
    direction = gradient / gradient.norm()
    h = .01
    with torch.no_grad():
        plus = b.answer_loss(b.context(r, vectors=u + h * direction), r['qa'][0])
        minus = b.answer_loss(b.context(r, vectors=u - h * direction), r['qa'][0])
    finite = (plus - minus) / (2 * h)
    torch.testing.assert_close(finite, gradient.norm(), rtol=.035, atol=5e-5)
    detached = Context(tuple((k.detach(), v.detach()) for k, v in ctx.layers), ctx.next_position,
                       ctx.public_start, ctx.public_length)
    assert not b.answer_loss(detached, r['qa'][0]).requires_grad


def test_blocked_cache_matches_dense_information_block(setup):
    b, cfg, r = setup
    u = b.embedding(b.tensor(b.initial_ids()))[0].detach().clone().requires_grad_(True)
    blocked = b.context(r, vectors=u, blocked=True)
    embeddings, start = b.layout(r, vectors=u)
    mask = causal_mask(embeddings.shape[1], 0, device=b.device, dtype=b.dtype)
    mask[:, :, start:, :start] = torch.finfo(b.dtype).min
    _, layers = b.forward(embeddings=embeddings, mask=mask)
    for (a, c), (x, y) in zip(blocked.layers, slice_layers(layers, start)):
        torch.testing.assert_close(a, x, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(c, y, rtol=1e-5, atol=1e-6)
    assert not b.answer_loss(blocked, r['qa'][0]).requires_grad


def test_question_isolation_and_generation_positions(setup, monkeypatch):
    b, cfg, r = setup
    ids = b.initial_ids()
    ctx = b.context(r, prompt_ids=ids)
    snapshot = tuple((k.clone(), v.clone()) for k, v in ctx.layers)
    first = b.answer_loss(ctx, r['qa'][0])
    b.answer_loss(ctx, r['qa'][1])
    torch.testing.assert_close(first, b.answer_loss(ctx, r['qa'][0]))
    for original, working in zip(snapshot, ctx.layers):
        torch.testing.assert_close(original[0], working[0]); torch.testing.assert_close(original[1], working[1])
    calls = []
    original_forward = b.forward
    def tracked(**kwargs):
        calls.append(kwargs['position_start'])
        logits, layers = original_forward(**kwargs)
        logits.fill_(-100); logits[:, :, 70] = 100
        return logits, layers
    monkeypatch.setattr(b, 'forward', tracked)
    cfg['max_new_tokens'] = 4
    _, info = b.generate(ctx, r['qa'][0])
    q = b.ids(b.question_text(r['qa'][0]['question']))
    assert calls == [ctx.next_position] + [ctx.next_position + len(q) + i for i in range(3)]
    assert info['truncated'] is True


def test_resume_matches_uninterrupted_optimizer_steps(setup, tmp_path, monkeypatch):
    from medical_kv import training
    b, cfg, r = setup
    second = copy.deepcopy(r); second['record_id'] = 'toy_2'
    train = [r, second]
    subsets = {'development_only': True, 'train_ids': ['toy_1', 'toy_2'], 'validation_ids': ['toy_v']}
    cfg_a = copy.deepcopy(cfg); cfg_a['output_directory'] = str(tmp_path / 'uninterrupted')
    complete = training.train_soft(b, cfg_a, train, [r], {'toy': True}, subsets)
    expected = read_checkpoint(tmp_path / 'uninterrupted/latest.pt')
    cfg_b = copy.deepcopy(cfg); cfg_b['output_directory'] = str(tmp_path / 'resumed')
    signal_state = {'requested': False}
    triggered = {'once': False}
    @contextmanager
    def stop_context():
        signal_state['requested'] = False
        yield signal_state
    save = training.save_checkpoint
    def interrupt_after_first_step(path, checkpoint):
        save(path, checkpoint)
        if checkpoint['state']['step'] == 1 and not triggered['once']:
            signal_state['requested'] = True; triggered['once'] = True
    monkeypatch.setattr(training, 'stop_at_boundary', stop_context)
    monkeypatch.setattr(training, 'save_checkpoint', interrupt_after_first_step)
    partial = training.train_soft(b, cfg_b, train, [r], {'toy': True}, subsets)
    assert partial['status'] == 'partial'
    result = training.train_soft(b, cfg_b, train, [r], {'toy': True}, subsets,
                                 resume=tmp_path / 'resumed/latest.pt')
    actual = read_checkpoint(tmp_path / 'resumed/latest.pt')
    assert complete['status'] == result['status'] == 'complete'
    torch.testing.assert_close(expected['prompt'], actual['prompt'], rtol=0, atol=0)
    assert expected['state']['step'] == actual['state']['step'] == 4
    assert expected['state']['field_exposures'] == actual['state']['field_exposures']


@pytest.mark.discrete
def test_discrete_search_serialization_and_real_candidate_losses(setup, tmp_path):
    from medical_kv.discrete import train_hard
    b, cfg, r = setup
    cfg['output_directory'] = str(tmp_path / 'hard')
    cfg['hard']['enabled'] = True
    result = train_hard(b, cfg, [r], [r], {'toy': True}, {'development_only': True})
    assert result['status'] == 'complete'
    selected = read_checkpoint(tmp_path / 'hard/best.pt')
    assert b.ids(selected['prompt_text']) == selected['prompt_ids']
    b.serialized_ids(r, selected['prompt_ids'])
    rows = [json.loads(x) for x in (tmp_path / 'hard/training.jsonl').read_text().splitlines()]
    assert any(r['evaluated_candidates'] > 1 for r in rows)
    assert all(r['candidate_loss_after'] <= r['candidate_loss_before'] for r in rows)
    assert selected['selected_by_validation'] is True


def test_actual_dataset_hashes_and_schema():
    cfg = load_config('configs/experiment_v1.json')
    identity = dataset_identity(cfg)
    assert identity['schema'] == 'source_preserving_v3'
    assert len(load_split(cfg, 'train')) == 1000
    assert len(load_split(cfg, 'validation')) == 200


@pytest.mark.discrete
def test_discrete_resume_preserves_search_rng(setup, tmp_path, monkeypatch):
    from medical_kv import discrete
    b, cfg, r = setup
    cfg['hard']['enabled'] = True
    subsets = {'development_only': True}
    a = copy.deepcopy(cfg); a['output_directory'] = str(tmp_path / 'hard_full')
    discrete.train_hard(b, a, [r], [r], {'toy': True}, subsets)
    expected = read_checkpoint(tmp_path / 'hard_full/latest.pt')
    signal_state, triggered = {'requested': False}, {'once': False}
    @contextmanager
    def stop_context():
        signal_state['requested'] = False
        yield signal_state
    original_save = discrete.save_checkpoint
    def interrupt_after_one(path, checkpoint):
        original_save(path, checkpoint)
        if checkpoint['state']['step'] == 1 and not triggered['once']:
            triggered['once'] = True; signal_state['requested'] = True
    monkeypatch.setattr(discrete, 'stop_at_boundary', stop_context)
    monkeypatch.setattr(discrete, 'save_checkpoint', interrupt_after_one)
    c = copy.deepcopy(cfg); c['output_directory'] = str(tmp_path / 'hard_resume')
    partial = discrete.train_hard(b, c, [r], [r], {'toy': True}, subsets)
    assert partial['status'] == 'partial'
    final = discrete.train_hard(b, c, [r], [r], {'toy': True}, subsets, resume=tmp_path / 'hard_resume/latest.pt')
    actual = read_checkpoint(tmp_path / 'hard_resume/latest.pt')
    assert final['status'] == 'complete'
    assert actual['prompt_ids'] == expected['prompt_ids']
    assert actual['search_rng'] == expected['search_rng']


def test_metrics_require_complete_records_and_preserve_exact_text(tmp_path):
    fields = ('name', 'birth_date', 'address', 'diagnosis')
    rows = [{'record_id': 'p1', 'field': f, 'question': f, 'gold': 'A1', 'prediction': 'A1',
             'em': True, 'nll': 1., 'truncated': False} for f in fields]
    summary = summarize(rows)
    assert summary['macro_em'] == summary['all_fields_em'] == summary['name_diagnosis_joint_em'] == 1
    with pytest.raises(ValueError):
        summarize(rows[:-1])
    a, b = tmp_path / 'a.jsonl', tmp_path / 'b.jsonl'
    a.write_text(''.join(json.dumps(x) + '\n' for x in rows))
    rows[0]['em'] = False
    b.write_text(''.join(json.dumps(x) + '\n' for x in rows))
    result = paired_compare(a, b, bootstrap_samples=10)
    assert result['macro_em_difference']['difference'] == .25
    rows[0]['gold'] = 'changed'
    b.write_text(''.join(json.dumps(x) + '\n' for x in rows))
    with pytest.raises(ValueError):
        paired_compare(a, b)


@pytest.mark.parametrize('method', ['soft', pytest.param('hard', marks=pytest.mark.discrete)])
def test_resume_does_not_restart_an_early_stopped_run(setup, tmp_path, method):
    from medical_kv.training import train_soft
    from medical_kv.discrete import train_hard
    b, cfg, r = setup
    for trainer in [train_soft if method == 'soft' else train_hard]:
        c = copy.deepcopy(cfg)
        c['hard']['enabled'] = method == 'hard'
        c.update({'output_directory': str(tmp_path / method), 'epochs': 5, 'patience': 1})
        c['hard']['steps'] = 4
        first = trainer(b, c, [r], [r], {'toy': True}, {'development_only': True})
        assert first['stop_reason'] == 'early_stopping'
        saved = read_checkpoint(tmp_path / method / 'latest.pt')
        again = trainer(b, c, [r], [r], {'toy': True}, {'development_only': True}, resume=tmp_path / method / 'latest.pt')
        assert again['state']['step'] == first['state']['step']
        restored = read_checkpoint(tmp_path / method / 'latest.pt')
        if method == 'soft':
            torch.testing.assert_close(saved['prompt'], restored['prompt'], rtol=0, atol=0)
        else:
            assert saved['prompt_ids'] == restored['prompt_ids']


@pytest.mark.parametrize('command', [['train-hard'], ['evaluate', '--condition', 'hard']])
def test_discrete_disabled_before_dataset_or_model_loading(monkeypatch, command):
    from medical_kv import cli
    def unexpected(*args, **kwargs):
        pytest.fail('Disabled discrete command reached dataset/model loading')
    monkeypatch.setattr(cli, 'dataset_identity', unexpected)
    monkeypatch.setattr(cli.Backend, 'load', unexpected)
    with pytest.raises(SystemExit) as stopped:
        cli.main(command)
    assert stopped.value.code == 2


def test_discrete_trainer_disabled_without_side_effects(tmp_path):
    from medical_kv.discrete import train_hard
    out = tmp_path / 'must_not_exist'
    cfg = load_config('configs/experiment_v1.json', {'output_directory': str(out)})
    with pytest.raises(ValueError, match='disabled'):
        train_hard(None, cfg, [], [], {}, {})
    assert not out.exists()


def test_loss_curve_resume_excludes_uncommitted_future_logs(tmp_path):
    from medical_kv.plotting import LossCurve
    from matplotlib.image import imread
    training = [{'step': 1, 'epoch': 1, 'loss': 4.}, {'step': 2, 'epoch': 1, 'loss': 3.},
                {'step': 2, 'epoch': 1, 'loss': 2.9}, {'step': 5, 'epoch': 2, 'loss': .1}]
    validation = [{'step': 2, 'epoch': 1, 'metrics': {'macro_nll': 3.2}},
                  {'step': 5, 'epoch': 2, 'metrics': {'macro_nll': .2}}]
    for name, rows in [('training', training), ('validation', validation)]:
        (tmp_path / (name + '.jsonl')).write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
    curve = LossCurve(tmp_path, max_step=2, max_epoch=1)
    assert [row['loss'] for row in curve.training] == [4., 2.9]
    assert [row['metrics']['macro_nll'] for row in curve.validation] == [3.2]
    path = curve.save()
    assert imread(path).shape == (825, 1500, 4)
    curve.update_training({'step': 3, 'epoch': 2, 'loss': 2.7})
    curve.update_validation({'step': 3, 'epoch': 2, 'metrics': {'macro_nll': 3.}})
    assert curve.training[-1]['step'] == curve.validation[-1]['step'] == 3
    assert not (tmp_path / 'loss_curve.tmp.png').exists()


@pytest.mark.parametrize('condition', ['public', 'full', 'blocked'])
def test_batched_questions_match_sequential_loss_gradient_generation(setup, condition):
    b, cfg, record = setup
    u = b.embedding(b.tensor(b.initial_ids()))[0].detach().clone().requires_grad_(True)
    context = b.context(record, vectors=u, full=condition == 'full', blocked=condition == 'blocked')
    snapshot = [(k.detach().clone(), v.detach().clone()) for k, v in context.layers]
    sequential = torch.stack([b.answer_loss(context, qa) for qa in record['qa']])
    batched = b.answer_losses(context, record['qa'])
    torch.testing.assert_close(batched, sequential, rtol=1e-5, atol=2e-6)
    if condition != 'blocked':
        expected = torch.autograd.grad(sequential.mean(), u, retain_graph=True)[0]
        actual = torch.autograd.grad(batched.mean(), u)[0]
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-6)
    else:
        assert not batched.requires_grad
    cfg['max_new_tokens'] = 4
    assert b.generate_many(context, record['qa']) == [b.generate(context, qa) for qa in record['qa']]
    for (k, v), (original_k, original_v) in zip(context.layers, snapshot):
        torch.testing.assert_close(k, original_k)
        torch.testing.assert_close(v, original_v)


def test_batched_generation_handles_independent_eos_and_positions(setup, monkeypatch):
    b, cfg, record = setup
    context = b.context(record, prompt_ids=b.initial_ids())
    original = b.forward
    calls = []
    def tracked(**kwargs):
        logits, layers = original(**kwargs)
        positions = kwargs['position_ids']
        calls.append(positions.detach().clone())
        logits.fill_(-100)
        for row in range(len(record['qa'])):
            # Each query stops at a different decoding iteration.
            token = b.eos if len(calls) > row else 70
            logits[row, -1, token] = 100
        return logits, layers
    monkeypatch.setattr(b, 'forward', tracked)
    cfg['max_new_tokens'] = 6
    results = b.generate_many(context, record['qa'])
    assert [info['generated_tokens'] for _, info in results] == [0, 1, 2, 3]
    assert all(info['stop_reason'] == 'eos' and not info['truncated'] for _, info in results)
    lengths = torch.tensor([len(b.ids(b.question_text(qa['question']))) for qa in record['qa']])
    for i, positions in enumerate(calls[1:]):
        torch.testing.assert_close(positions[:, 0], context.next_position + lengths + i)


def test_gpu_launcher_preserves_ids_and_arguments(monkeypatch):
    from medical_kv.launch import launch, parse_gpus
    import types
    import subprocess
    import os
    monkeypatch.delenv('WORLD_SIZE', raising=False)
    captured = []
    def run(command, **kwargs):
        captured.append((command, kwargs['env']))
        return types.SimpleNamespace(returncode=0)
    monkeypatch.setattr(subprocess, 'run', run)
    original_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    launch('2,5', ['train-soft', '--gpus', '2,5', '--output-directory', 'path with spaces'])
    command, env = captured[-1]
    assert env['CUDA_VISIBLE_DEVICES'] == '2,5'
    assert '--nproc-per-node=2' in command and '--max-restarts=0' in command
    assert command[-3:] == ['train-soft', '--output-directory', 'path with spaces']
    assert '--gpus' not in command
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == original_visible
    launch('5', ['evaluate', '--gpus=5', '--condition', 'random'])
    assert 'torch.distributed.run' not in captured[-1][0]
    for invalid in ('', '0,0', '-1', '0,,2', 'gpu0', '0;2'):
        with pytest.raises(ValueError):
            parse_gpus(invalid)


@pytest.mark.parametrize('patients,world_size', [(3, 2), (1, 2), (3, 4)])
def test_patient_parallel_gradient_preserves_global_mean_with_uneven_shards(setup, patients, world_size):
    from medical_kv.training import local_batch_backward
    b, cfg, record = setup
    records = []
    for i in range(patients):
        item = copy.deepcopy(record)
        item['record_id'] = str(i)
        item['qa'][0]['answer'] = 'A' * (i + 1)
        records.append(item)
    initial = b.embedding(b.tensor(b.initial_ids()))[0].detach().clone()
    reference = initial.clone().requires_grad_(True)
    expected_loss = torch.stack([b.record_loss(r, vectors=reference) for r in records]).mean()
    expected_loss.backward()
    summed_gradient = torch.zeros_like(reference)
    loss_sum = 0.
    for rank in range(world_size):
        prompt = initial.clone().requires_grad_(True)
        losses = local_batch_backward(b, prompt, records, list(range(patients)), rank, world_size)
        summed_gradient += prompt.grad
        loss_sum += sum(losses)
    torch.testing.assert_close(summed_gradient, reference.grad, rtol=2e-4, atol=2e-6)
    assert loss_sum / patients == pytest.approx(float(expected_loss.detach()), abs=2e-6)


def test_csv_preserves_raw_multiline_answers_and_epoch_metadata(tmp_path):
    import csv
    from medical_kv.evaluation import write_predictions_csv
    row = {'record_id': 'preview_014', 'field': 'name', 'question': 'Name, exactly?',
           'gold': 'Brock Toby Lockman', 'prediction': 'Name: "Brock"\nToby, Lockman',
           'em': False, 'nll': 1.25, 'generated_tokens': 10, 'stop_reason': 'eos', 'truncated': False}
    path = tmp_path / 'predictions.csv'
    write_predictions_csv(path, [row], {'epoch': 18, 'step': 2250, 'split': 'validation', 'condition': 'soft'})
    assert path.read_bytes().startswith(b'\xef\xbb\xbf')
    with path.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    for field in ('record_id', 'field', 'question', 'gold', 'prediction'):
        assert rows[0][field] == row[field]
    assert rows[0]['epoch'] == '18' and rows[0]['step'] == '2250'
    assert rows[0]['birth_date_seen_in_train'] == ''


def test_training_saves_every_validation_answer_and_preserves_prior_attempt(setup, tmp_path, monkeypatch):
    import csv
    from medical_kv.training import train_soft
    b, cfg, record = setup
    cfg['epochs'] = 1
    out = tmp_path / 'saved_validation'
    cfg['output_directory'] = str(out)
    interrupted = out / 'validation_predictions/epoch_001_step_000001'
    interrupted.mkdir(parents=True)
    (interrupted / 'sentinel.txt').write_text('prior attempt', encoding='utf-8')
    calls = []
    original = b.generate_many
    def capture(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(b, 'generate_many', capture)
    result = train_soft(b, cfg, [record], [record], {'toy': True}, {'development_only': True})
    assert result['status'] == 'complete' and len(calls) == 1
    validation = json.loads((out / 'validation.jsonl').read_text(encoding='utf-8').strip())
    assert '_attempt_002/' in validation['predictions_csv']
    assert (interrupted / 'sentinel.txt').read_text(encoding='utf-8') == 'prior attempt'
    with (out / validation['predictions_csv']).open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    answers = [json.loads(line) for line in (out / validation['predictions_jsonl']).read_text(encoding='utf-8').splitlines()]
    assert len(rows) == len(answers) == 4
    for row, answer, qa in zip(rows, answers, record['qa']):
        assert row['field'] == qa['field'] and row['gold'] == qa['answer']
        assert row['prediction'] == answer['prediction']
        assert row['epoch'] == row['step'] == '1'
    assert validation['metrics']['birth_date_subgroups']['train_seen']['records'] == 1

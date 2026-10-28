"""Finance configurable-field losses/routing and legacy medical compatibility."""
import copy
import json
import sys

import pytest
import torch

from test_experiment import setup
from test_multiscene_runtime import configure, records, perfect
from medical_kv.config import ROOT, load_config
from medical_kv.evaluation import evaluate_records
from medical_kv.io import checkpoint_compatibility, read_checkpoint
from medical_kv.prompt_partition import optimizer_parameters, assign_partition_gradients, partition_focal_fields
from medical_kv.tasks import questions_for
from medical_kv.training import local_batch_backward, train_soft
from multiscene_common import annotation, split_name, make_qa

FOCAL_FIELDS = ['timestamp', 'identity_number', 'card_number', 'account_balance', 'transfer_amount']


def finance_config(cfg):
    configure(cfg)
    cfg.update(scene='finance', prompt_length=5, prompt_initialization='fresh_token_embeddings',
               prompt_partition={'other_length': 3, 'focal_length': 2}, focal_fields=list(FOCAL_FIELDS),
               loss={'type': 'field_focal_split', 'gamma': 2., 'alpha': 1.}, split_private_prefill=True)
    cfg['hard']['enabled'] = False


def finance_record(case='full'):
    fields = {'name': 'Ax Bo', 'address': '3 Oak', 'amount_paid': '12 USD', 'amount_received': '12 USD',
              'payment_format': 'Wire', 'timestamp': '2022/09/09 23:43', 'identity_number': '123-45-6789',
              'card_number': '1234 5678', 'account_balance': '$12.34', 'transfer_amount': 'twelve dollars',
              'recorded_account_number': 'A123', 'recipient_name': 'Cy', 'transfer_time': '3'}
    if case == 'partial':
        fields = {k: v for k, v in fields.items() if k in ('name', 'address', 'timestamp')}
    elif case == 'none':
        fields = {k: v for k, v in fields.items() if k not in FOCAL_FIELDS}
    elif case == 'only':
        fields = {k: v for k, v in fields.items() if k in FOCAL_FIELDS}
    row = copy.deepcopy(records()[1])
    row.update(record_id='finance_' + case, fields=fields,
               private_prefix='\n'.join(f'{k}: {v}' for k, v in fields.items()))
    row['field_annotations'] = {k: annotation(row['private_prefix'], v, 'fixture:finance') for k, v in fields.items()}
    row['name_annotation'] = split_name(fields['name']) if 'name' in fields else None
    row['source_name_fields'] = row['name_annotation']['components'] if row['name_annotation'] else {}
    make_qa(row)
    return row


@pytest.mark.parametrize('batched', [False, True])
@pytest.mark.parametrize('case', ['full', 'partial', 'none', 'only'])
def test_finance_original_field_weights_and_routed_gradient(setup, monkeypatch, batched, case):
    b, cfg, _ = setup
    finance_config(cfg)
    cfg['batch_questions'] = batched
    row = finance_record(case)
    initial = b.embedding(b.tensor(b.initial_ids()))[0].detach()
    u = initial.clone().requires_grad_(True)
    ctx = b.context(row, vectors=u)
    qs = questions_for(row, cfg)
    values = b.answer_losses(ctx, qs, training=True)
    # Independent grouping: each original field gets 1/G, including each of
    # the five selected fields separately. Name components still count as one.
    losses = {f: values[[i for i, q in enumerate(qs) if q['group'] == f],
                         0 if f in FOCAL_FIELDS else 1].mean() for f in row['fields']}
    zero = values.sum() * 0
    other = sum((v for f, v in losses.items() if f not in FOCAL_FIELDS), zero) / len(losses)
    focal = sum((v for f, v in losses.items() if f in FOCAL_FIELDS), zero) / len(losses)
    go = torch.autograd.grad(other, u, retain_graph=True)[0]
    gf = torch.autograd.grad(focal, u)[0]
    expected = torch.cat((go[:3], gf[3:]))
    calls = []
    context = b.context
    def counted(*args, **kwargs):
        result = context(*args, **kwargs)
        assert result.physical_length == len(b.public_ids)
        calls.append(1)
        return result
    monkeypatch.setattr(b, 'context', counted)
    actual = initial.clone().requires_grad_(True)
    diagnostics = []
    objectives = local_batch_backward(b, actual, [row], [0], 0, 1, diagnostics=diagnostics)
    assert len(calls) == 1
    torch.testing.assert_close(actual.grad, expected, rtol=3e-4, atol=3e-6)
    assert objectives[0] == pytest.approx((other + focal).item())
    stats = diagnostics[0]
    assert stats['ce_other_loss'] == pytest.approx(other.item())
    assert stats['focal_loss'] == pytest.approx(focal.item())
    active = len(set(row['fields']).intersection(FOCAL_FIELDS))
    assert stats['focal_fields_per_record'] == active
    assert stats['focal_group_loss'] == pytest.approx(focal.item() * len(losses) / active if active else 0.)
    assert all(p.grad is None for p in b.model.parameters())
    if case == 'none':
        assert torch.count_nonzero(actual.grad[3:]) == 0
    if case == 'only':
        assert torch.count_nonzero(actual.grad[:3]) == 0


def test_finance_focal_math_and_global_batch_with_empty_rank(setup):
    b, cfg, _ = setup
    finance_config(cfg)
    logits = torch.tensor([[[2., 0., -1.], [0., 1., -1.]]], requires_grad=True)
    labels = torch.tensor([[0, 1]])
    probability = logits.softmax(-1)[0, torch.arange(2), labels[0]]
    expected_focal = (-(1 - probability).square() * probability.log()).mean()
    values = b._mean_answer_losses(logits, labels, training=True)
    torch.testing.assert_close(values[0, 0], expected_focal)
    assert values[0, 1].requires_grad
    rows = [finance_record('full'), finance_record('partial'), finance_record('none')]
    initial = b.embedding(b.tensor(b.initial_ids()))[0].detach()
    whole = initial.clone().requires_grad_(True)
    local_batch_backward(b, whole, rows, [0, 1, 2], 0, 1)
    summed = torch.zeros_like(initial)
    for rank in range(4):
        u = initial.clone().requires_grad_(True)
        local_batch_backward(b, u, rows, [0, 1, 2], rank, 4)
        summed += u.grad
    torch.testing.assert_close(summed, whole.grad, rtol=3e-4, atol=3e-6)


def test_finance_inactive_segments_skip_adam_updates():
    cfg = {'loss': {'type': 'field_focal_split'}, 'prompt_partition': {'other_length': 3},
           'focal_fields': FOCAL_FIELDS}
    initial = torch.randn(5, 4)
    parameters = optimizer_parameters(initial, cfg)
    optimizer = torch.optim.AdamW(parameters, lr=.1, weight_decay=0)
    rows = [finance_record('full'), finance_record('none'), finance_record('only')]
    for n, active in enumerate(((True, True), (True, False), (False, True))):
        previous = [p.detach().clone() for p in parameters]
        optimizer.zero_grad(set_to_none=True)
        assign_partition_gradients(parameters, torch.ones_like(initial), rows, [n], focal_fields=FOCAL_FIELDS)
        optimizer.step()
        for i in range(2):
            assert (parameters[i].grad is not None) == active[i]
            if not active[i]:
                torch.testing.assert_close(parameters[i], previous[i], rtol=0, atol=0)
    assert optimizer.state[parameters[0]]['step'] == 2
    assert optimizer.state[parameters[1]]['step'] == 2


def test_finance_fresh_pause_resume_logs_and_nll(setup, tmp_path, monkeypatch):
    b, cfg, _ = setup
    finance_config(cfg)
    cfg.update(epochs=2, early_stopping=False, comparison_epochs=1, output_directory=str(tmp_path / 'finance'))
    row = finance_record()
    monkeypatch.setattr(b, 'generate_many', perfect)
    initial = b.embedding(b.tensor(b.initial_ids()))[0].detach()
    out = tmp_path / 'finance'
    from medical_kv import training
    save = training.save_checkpoint
    pause_once = [True]
    def save_and_pause(path, value):
        save(path, value)
        if path.parent == out and path.name == 'latest.pt' and value['state']['step'] == 1 and pause_once[0]:
            pause_once[0] = False
            (out / 'pause_request.json').write_text('{}')
    monkeypatch.setattr(training, 'save_checkpoint', save_and_pause)
    train_soft(b, cfg, [row], [row], {'fixture': 'finance'}, {'development_only': True}, epoch_validation=False)
    paused = read_checkpoint(out / 'latest.pt')
    torch.testing.assert_close(paused['initial_prompt'], initial, rtol=0, atol=0)
    assert paused['run_info']['focal_fields'] == FOCAL_FIELDS
    assert paused['run_info']['prompt_initialization']['focal_length'] == 2
    train_soft(b, cfg, [row], [row], {'fixture': 'finance'}, {'development_only': True}, resume=out / 'latest.pt')
    resumed = read_checkpoint(out / 'final.pt')
    assert resumed['run_signature'] == paused['run_signature']
    cfg['output_directory'] = str(tmp_path / 'continuous')
    train_soft(b, cfg, [row], [row], {'fixture': 'finance'}, {'development_only': True}, epoch_validation=False)
    continuous = read_checkpoint(tmp_path / 'continuous/final.pt')
    torch.testing.assert_close(resumed['prompt'], continuous['prompt'], rtol=0, atol=0)
    for a, c in zip(resumed['optimizer']['state'].values(), continuous['optimizer']['state'].values()):
        for key in ('step', 'exp_avg', 'exp_avg_sq'):
            torch.testing.assert_close(a[key], c[key], rtol=0, atol=0)
    for log in map(json.loads, (out / 'training.jsonl').read_text().splitlines()):
        assert log['loss_type'] == 'field_focal_split'
        assert log['focal_fields'] == FOCAL_FIELDS
        assert log['loss'] == pytest.approx(log['ce_other_loss'] + log['focal_loss'])
        assert log['focal_fields_per_record'] == 5
        assert log['focal_group_loss'] == pytest.approx(log['focal_loss'] * len(row['fields']) / 5)
        assert 'birth_gradient_norm' not in log and 'birth_date_focal_loss' not in log
    checkpoint_compatibility(resumed, b)
    from medical_kv.resume import validate_resume
    changed = copy.deepcopy(resumed['run_info']['resume_contract'])
    changed['config']['focal_fields'].pop()
    with pytest.raises(ValueError, match='config changed'):
        validate_resume(resumed, changed)
    _, actual = evaluate_records(b, [row], vectors=resumed['prompt'])
    cfg['loss']['type'] = 'ce'
    _, expected = evaluate_records(b, [row], vectors=resumed['prompt'])
    assert [r['nll'] for r in actual] == [r['nll'] for r in expected]


@pytest.mark.parametrize('value', [None, [], ['timestamp', 'timestamp'], ['phone'], ['sender_account'],
                                    ['birth_date'], ['name_first'], [None], 'timestamp'])
def test_invalid_finance_focal_fields_are_rejected(value):
    with pytest.raises(ValueError, match='focal_fields'):
        load_config('configs/experiment_v12_finance_split_u.json', {'focal_fields': value or []})


def test_medical_config_stays_fixed_and_finance_config_is_independent():
    medical = load_config('configs/experiment_v10_medical_split_u.json')
    assert 'focal_fields' not in medical
    assert medical['prompt_partition'] == {'other_length': 32, 'birth_length': 16}
    assert partition_focal_fields(medical) == ('birth_date',)
    with pytest.raises(ValueError, match='legacy medical'):
        load_config(medical, {'focal_fields': FOCAL_FIELDS})
    cfg = load_config('configs/experiment_v12_finance_split_u.json')
    evaluation = load_config('configs/evaluation_v8_finance_split_u.json')
    assert cfg['focal_fields'] == FOCAL_FIELDS
    assert cfg['prompt_length'] == 48 and cfg['prompt_partition'] == {'other_length': 32, 'focal_length': 16}
    assert {k for k in cfg if cfg[k] != evaluation[k]} == {'output_directory'}
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / 'tools/tokenizer/Qwen3-4B'), local_files_only=True)
    assert len(tokenizer.encode(cfg['manual_prompt'], add_special_tokens=False)) == 48
    assert len(tokenizer.encode(cfg['public_text'], add_special_tokens=False)) == 56
    with pytest.raises(ValueError, match='finance multiscene'):
        load_config(cfg, {'scene': 'medical'})
    with pytest.raises(ValueError, match='prompt_partition'):
        load_config(cfg, {'prompt_partition': {'other_length': 32, 'focal_length': 15}})


def test_finance_pipeline_uses_independent_configs_and_outputs(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / 'scripts'))
    import run_multiscene_experiment as pipeline
    train, test, run, output, log = pipeline.experiment_paths(tmp_path, 'finance', 'field_focal_split')
    cfg = load_config('configs/experiment_v12_finance_split_u.json')
    train.parent.mkdir()
    train.write_text(json.dumps(cfg))
    test.write_text((ROOT / 'configs/evaluation_v8_finance_split_u.json').read_text())
    calls = []
    def stage(command, **kwargs):
        calls.append(command)
        if len(calls) == 1:
            run.mkdir(parents=True)
            (run / 'best.pt').write_bytes(b'fixture')
            (run / 'run.json').write_text(json.dumps({'config': cfg, 'run_signature': 'fixture'}))
            (run / 'run_summary.json').write_text(json.dumps({'status': 'complete', 'partial_dataset': False,
                'stop_reason': 'epochs_completed', 'state': {'epoch': 100}, 'run_signature': 'fixture'}))
    monkeypatch.setattr(pipeline, 'run_child', stage)
    pipeline.run_experiment(tmp_path, '1,2', scene='finance', loss='field_focal_split', epoch_validation=False)
    assert str(train) in calls[0] and '--no-epoch-validation' in calls[0]
    assert '--resume' not in calls[0]
    assert str(run / 'best.pt') in calls[1] and str(output) in calls[1] and log.exists()
    with pytest.raises(FileExistsError):
        pipeline.run_experiment(tmp_path, '1', scene='finance', loss='field_focal_split')
    with pytest.raises(ValueError):
        pipeline.experiment_paths(tmp_path, 'medical', 'field_focal_split')


def test_finance_recovery_routes_to_finance_split_pipeline(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import training_control as command
    import medical_kv.control as control
    run = tmp_path / 'runs/finance'
    run.mkdir(parents=True)
    (run / 'run.json').write_text(json.dumps({'method': 'soft', 'config': {
        'scene': 'finance', 'question_protocol': 'multiscene_privacy_v1',
        'loss': {'type': 'field_focal_split', 'gamma': 2, 'alpha': 1}}}))
    monkeypatch.setattr(command, 'ROOT', tmp_path)
    monkeypatch.setattr(command, 'os', SimpleNamespace(name='posix'))
    monkeypatch.setattr(command, 'training_status', lambda *a: {
        'running': False, 'checkpoint_exists': True, 'status': 'paused'})
    monkeypatch.setattr(control, 'process_start', lambda pid: 'fixture')
    calls = []
    monkeypatch.setattr(command.subprocess, 'Popen', lambda args, **kw: calls.append(args) or SimpleNamespace(pid=123))
    command.recover(run, ['1', '2'])
    assert calls[0][calls[0].index('--scene') + 1] == 'finance'
    assert calls[0][calls[0].index('--loss') + 1] == 'field_focal_split'
    assert '--resume' in calls[0]

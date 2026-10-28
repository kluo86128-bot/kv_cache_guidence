"""HR six-field routing, birthday grouping, production paths and recovery."""
import copy
import json
import sys

import pytest
import torch

from test_experiment import setup
from test_multiscene_runtime import configure, records, perfect
from medical_kv.config import ROOT, load_config
from medical_kv.evaluation import evaluate_records
from medical_kv.io import read_checkpoint, checkpoint_compatibility
from medical_kv.tasks import questions_for
from medical_kv.training import local_batch_backward, train_soft
from multiscene_common import annotation, split_name, make_qa

FOCAL_FIELDS = ['identity_number', 'email', 'address', 'birth_date', 'hire_date', 'leave_start']


def hr_config(cfg):
    configure(cfg)
    cfg.update(scene='hr', prompt_length=5, prompt_initialization='fresh_token_embeddings',
               prompt_partition={'other_length': 3, 'focal_length': 2}, focal_fields=list(FOCAL_FIELDS),
               loss={'type': 'field_focal_split', 'gamma': 2., 'alpha': 1.}, split_private_prefill=True)
    cfg['hard']['enabled'] = False


def hr_record(case='full'):
    fields = {'name': 'Ax Bo', 'birth_date': '1980-02-03', 'identity_number': '123-45-6789',
              'email': 'ax@example.com', 'address': '3 Oak', 'hire_date': '2005-06-07',
              'salary': '40000', 'department': 'Sales', 'title': 'Analyst', 'leave_start': '2023-08-09',
              'leave_duration': '3 days', 'leave_reason': 'a family appointment',
              'complaint_target': 'my manager', 'complaint_content': 'My manager yelled insults at me.'}
    if case == 'partial':
        fields = {f: v for f, v in fields.items() if f in ('name', 'birth_date', 'hire_date', 'email')}
    elif case == 'none':
        fields = {f: v for f, v in fields.items() if f not in FOCAL_FIELDS}
    elif case == 'only':
        fields = {f: v for f, v in fields.items() if f in FOCAL_FIELDS}
    r = copy.deepcopy(records()[1])
    r.update(scene='hr', record_id='hr_' + case, fields=fields,
             private_prefix='\n'.join(f + ': ' + v for f, v in fields.items()))
    r['field_annotations'] = {f: annotation(r['private_prefix'], v, 'fixture:hr') for f, v in fields.items()}
    r['name_annotation'] = split_name(fields['name']) if 'name' in fields else None
    r['source_name_fields'] = r['name_annotation']['components'] if r['name_annotation'] else {}
    make_qa(r)
    return r


@pytest.mark.parametrize('batched', [True, False])
@pytest.mark.parametrize('case', ['full', 'partial', 'none', 'only'])
def test_hr_original_field_weights_shared_cache_and_routed_gradient(setup, monkeypatch, batched, case):
    b, cfg, _ = setup
    hr_config(cfg)
    cfg['batch_questions'] = batched
    r = hr_record(case)
    qs = questions_for(r, cfg)
    initial = b.embedding(b.tensor(b.initial_ids()))[0].detach()
    u = initial.clone().requires_grad_(True)
    values = b.answer_losses(b.context(r, vectors=u), qs, training=True)
    # Independently average components BEFORE applying the original-field mean.
    grouped = {f: values[[i for i, q in enumerate(qs) if q['group'] == f],
                        0 if f in FOCAL_FIELDS else 1].mean() for f in r['fields']}
    zero = values.sum() * 0
    other = sum((v for f, v in grouped.items() if f not in FOCAL_FIELDS), zero) / len(grouped)
    focal = sum((v for f, v in grouped.items() if f in FOCAL_FIELDS), zero) / len(grouped)
    go = torch.autograd.grad(other, u, retain_graph=True)[0]
    gf = torch.autograd.grad(focal, u)[0]
    expected = torch.cat((go[:3], gf[3:]))
    calls = []
    context = b.context
    def counted(*args, **kwargs):
        ctx = context(*args, **kwargs)
        assert ctx.physical_length == len(b.public_ids)
        calls.append(1)
        return ctx
    monkeypatch.setattr(b, 'context', counted)
    actual = initial.clone().requires_grad_(True)
    diagnostics = []
    losses = local_batch_backward(b, actual, [r], [0], 0, 1, diagnostics=diagnostics)
    assert calls == [1]
    torch.testing.assert_close(actual.grad, expected, rtol=3e-4, atol=3e-6)
    assert losses[0] == pytest.approx((other + focal).item())
    active = len(set(r['fields']) & set(FOCAL_FIELDS))
    assert diagnostics[0]['focal_fields_per_record'] == active
    assert diagnostics[0]['focal_group_loss'] == pytest.approx(focal.item() * len(grouped) / active if active else 0.)
    if 'birth_date' in r['fields']:
        assert sum(q['group'] == 'birth_date' for q in qs) == 3
    if 'hire_date' in r['fields']:
        assert sum(q['group'] == 'hire_date' for q in qs) == 1
    if 'leave_start' in r['fields']:
        assert sum(q['group'] == 'leave_start' for q in qs) == 1
    assert all(p.grad is None for p in b.model.parameters())


def test_hr_global_batch_empty_rank_and_missing_fields(setup):
    b, cfg, _ = setup
    hr_config(cfg)
    rows = [hr_record(case) for case in ('full', 'partial', 'none')]
    initial = b.embedding(b.tensor(b.initial_ids()))[0].detach()
    whole = initial.clone().requires_grad_(True)
    local_batch_backward(b, whole, rows, [0, 1, 2], 0, 1)
    shards = []
    for rank in range(4):
        u = initial.clone().requires_grad_(True)
        local_batch_backward(b, u, rows, [0, 1, 2], rank, 4)
        shards.append(u.grad)
    assert torch.count_nonzero(shards[-1]) == 0
    torch.testing.assert_close(sum(shards), whole.grad, rtol=3e-4, atol=3e-6)


def test_hr_pause_resume_exact_adam_logs_and_ce_evaluation(setup, tmp_path, monkeypatch):
    b, cfg, _ = setup
    hr_config(cfg)
    cfg.update(epochs=2, early_stopping=False, comparison_epochs=1, output_directory=str(tmp_path / 'hr'))
    r = hr_record()
    monkeypatch.setattr(b, 'generate_many', perfect)
    out = tmp_path / 'hr'
    from medical_kv import training
    save, requested = training.save_checkpoint, []
    def save_pause(path, value):
        save(path, value)
        if path.parent == out and path.name == 'latest.pt' and value['state']['step'] == 1 and not requested:
            requested.append(True)
            (out / 'pause_request.json').write_text('{}')
    monkeypatch.setattr(training, 'save_checkpoint', save_pause)
    initial = b.embedding(b.tensor(b.initial_ids()))[0].detach()
    dataset, subsets = {'fixture': 'hr'}, {'development_only': True}
    train_soft(b, cfg, [r], [r], dataset, subsets, epoch_validation=False)
    paused = read_checkpoint(out / 'latest.pt')
    torch.testing.assert_close(paused['initial_prompt'], initial, rtol=0, atol=0)
    assert len(paused['optimizer']['state']) == 2
    train_soft(b, cfg, [r], [r], dataset, subsets, resume=out / 'latest.pt')
    resumed = read_checkpoint(out / 'final.pt')
    cfg['output_directory'] = str(tmp_path / 'continuous')
    train_soft(b, cfg, [r], [r], dataset, subsets, epoch_validation=False)
    continuous = read_checkpoint(tmp_path / 'continuous/final.pt')
    torch.testing.assert_close(resumed['prompt'], continuous['prompt'], rtol=0, atol=0)
    assert resumed['scheduler'] == continuous['scheduler']
    for a, c in zip(resumed['optimizer']['state'].values(), continuous['optimizer']['state'].values()):
        for key in ('step', 'exp_avg', 'exp_avg_sq'):
            torch.testing.assert_close(a[key], c[key], rtol=0, atol=0)
    for log in map(json.loads, (out / 'training.jsonl').read_text().splitlines()):
        assert log['loss'] == pytest.approx(log['ce_other_loss'] + log['focal_loss'])
        assert log['focal_fields'] == FOCAL_FIELDS and log['focal_fields_per_record'] == 6
        assert log['gradient_routing'] == 'other_ce_to_original__field_focal_to_appended'
        assert 'birth_date_focal_loss' not in log and 'pcgrad_projections' not in log
    checkpoint_compatibility(resumed, b)
    from medical_kv.resume import validate_resume
    for key, value in [('focal_fields', FOCAL_FIELDS[:-1]), ('dataset_directory', 'old_v2')]:
        changed = copy.deepcopy(resumed['run_info']['resume_contract'])
        changed['config'][key] = value
        with pytest.raises(ValueError, match='config changed'):
            validate_resume(resumed, changed)
    _, actual = evaluate_records(b, [r], vectors=resumed['prompt'])
    cfg['loss']['type'] = 'ce'
    _, expected = evaluate_records(b, [r], vectors=resumed['prompt'])
    assert [r['nll'] for r in actual] == [r['nll'] for r in expected]


@pytest.mark.parametrize('fields', [[], ['timestamp'], ['phone'], ['employment_status'], ['birth_year'],
                                     ['identity_number', 'identity_number'], 'birth_date'])
def test_hr_rejects_wrong_field_lists(fields):
    with pytest.raises(ValueError, match='focal_fields'):
        load_config('configs/experiment_v14_hr_split_u.json', {'focal_fields': fields})


def test_hr_production_configs_and_previous_scene_constraints():
    cfg = load_config('configs/experiment_v14_hr_split_u.json')
    ev = load_config('configs/evaluation_v10_hr_split_u.json')
    assert {k for k in cfg if cfg[k] != ev[k]} == {'output_directory'}
    assert cfg['focal_fields'] == FOCAL_FIELDS
    assert cfg['scene'] == 'hr' and cfg['prompt_partition'] == {'other_length': 32, 'focal_length': 16}
    assert cfg['prompt_length'] == 48 and cfg['input_cache_directory'] == 'data/input_cache_hr_v3_balanced'
    assert cfg['dataset_directory'] == 'data/dataset_multiscene_v3_hr_balanced'
    assert cfg['prompt_initialization'] == 'fresh_token_embeddings'
    for overrides in ({'scene': 'finance'}, {'loss': {'type': 'field_focal_pcgrad', 'gamma': 2, 'alpha': 1}},
                      {'prompt_partition': {'other_length': 32, 'focal_length': 15}}):
        with pytest.raises(ValueError):
            load_config(cfg, overrides)
    old = load_config('configs/experiment_v12_finance_split_u.json')
    with pytest.raises(ValueError):
        load_config(old, {'focal_fields': ['birth_date']})
    medical = load_config('configs/experiment_v10_medical_split_u.json')
    assert 'focal_fields' not in medical and medical['prompt_partition']['birth_length'] == 16
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(ROOT / 'tools/tokenizer/Qwen3-4B'), local_files_only=True)
    assert len(tok.encode(cfg['public_text'], add_special_tokens=False)) == 56
    assert len(tok.encode(cfg['manual_prompt'], add_special_tokens=False)) == 48


def test_hr_pipeline_uses_independent_configs_and_outputs(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / 'scripts'))
    import run_multiscene_experiment as pipeline
    train, test, run, output, log = pipeline.experiment_paths(tmp_path, 'hr', 'field_focal_split')
    cfg = load_config('configs/experiment_v14_hr_split_u.json')
    train.parent.mkdir()
    train.write_text(json.dumps(cfg))
    test.write_text((ROOT / 'configs/evaluation_v10_hr_split_u.json').read_text())
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
    pipeline.run_experiment(tmp_path, '1,2', scene='hr', loss='field_focal_split', epoch_validation=False)
    assert str(train) in calls[0] and '--no-epoch-validation' in calls[0]
    assert '--resume' not in calls[0]
    assert str(run / 'best.pt') in calls[1] and str(output) in calls[1] and log.exists()
    with pytest.raises(FileExistsError):
        pipeline.run_experiment(tmp_path, '1', scene='hr', loss='field_focal_split')
    with pytest.raises(ValueError):
        pipeline.experiment_paths(tmp_path, 'medical', 'field_focal_split')


def test_hr_recovery_routes_to_hr_split_pipeline(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import training_control as command
    import medical_kv.control as control
    run = tmp_path / 'runs/hr'
    run.mkdir(parents=True)
    (run / 'run.json').write_text(json.dumps({'method': 'soft', 'config': {
        'scene': 'hr', 'question_protocol': 'multiscene_privacy_v1',
        'loss': {'type': 'field_focal_split', 'gamma': 2, 'alpha': 1}}}))
    monkeypatch.setattr(command, 'ROOT', tmp_path)
    monkeypatch.setattr(command, 'os', SimpleNamespace(name='posix'))
    monkeypatch.setattr(command, 'training_status', lambda *a: {
        'running': False, 'checkpoint_exists': True, 'status': 'paused'})
    monkeypatch.setattr(control, 'process_start', lambda pid: 'fixture')
    calls = []
    monkeypatch.setattr(command.subprocess, 'Popen', lambda args, **kw: calls.append(args) or SimpleNamespace(pid=123))
    command.recover(run, ['1', '2'])
    assert calls[0][calls[0].index('--scene') + 1] == 'hr'
    assert calls[0][calls[0].index('--loss') + 1] == 'field_focal_split'
    assert '--resume' in calls[0]

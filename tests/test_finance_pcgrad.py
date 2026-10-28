"""Finance field objectives, shared-U PCGrad, deployment and exact CPU recovery."""
import copy
import json
import sys

import pytest
import torch

from test_experiment import setup
from test_finance_split_prompt import finance_config, finance_record, FOCAL_FIELDS
from test_multiscene_runtime import perfect
from medical_kv.config import ROOT, load_config
from medical_kv.evaluation import evaluate_records
from medical_kv.io import read_checkpoint
from medical_kv.pcgrad import local_field_gradients, project_field_gradients, is_pcgrad
from medical_kv.prompt_partition import is_partitioned
from medical_kv.tasks import questions_for
from medical_kv.training import train_soft


def configure_finance_pcgrad(cfg):
    finance_config(cfg)
    cfg.pop('prompt_partition')
    cfg.update(prompt_length=3, loss={'type': 'field_focal_pcgrad', 'gamma': 2., 'alpha': 1.})


@pytest.mark.parametrize('batched', [False, True])
@pytest.mark.parametrize('case', ['full', 'partial', 'none', 'only'])
def test_each_original_field_matches_independent_objective(setup, monkeypatch, batched, case):
    b, cfg, _ = setup
    configure_finance_pcgrad(cfg)
    cfg['batch_questions'] = batched
    row = finance_record(case)
    initial = b.embedding(b.tensor(b.initial_ids()))[0].detach()
    u = initial.clone().requires_grad_(True)
    qs = questions_for(row, cfg)
    values = b.answer_losses(b.context(row, vectors=u), qs, training=True)
    fields = sorted(row['fields'])
    objectives = [values[[i for i, q in enumerate(qs) if q['group'] == f],
                         0 if f in FOCAL_FIELDS else 1].mean() / len(fields) for f in fields]
    expected = torch.stack([torch.autograd.grad(v, u, retain_graph=True)[0] for v in objectives])
    calls = []
    original = b.context
    def counted(*args, **kwargs):
        ctx = original(*args, **kwargs)
        assert ctx.physical_length == len(b.public_ids)
        calls.append(1)
        return ctx
    monkeypatch.setattr(b, 'context', counted)
    losses, gradients = local_field_gradients(b, initial.clone().requires_grad_(True),
                                             [row], [0], 0, 1, fields)
    assert calls == [1]
    assert losses[0] == pytest.approx(sum(objectives).item())
    torch.testing.assert_close(gradients, expected, rtol=3e-4, atol=3e-6)
    assert all(p.grad is None for p in b.model.parameters())


def test_global_field_reduction_with_missing_fields_and_empty_rank(setup):
    b, cfg, _ = setup
    configure_finance_pcgrad(cfg)
    rows = [finance_record(case) for case in ('full', 'partial', 'none')]
    fields = sorted({f for row in rows for f in row['fields']}) + ['account_type']
    initial = b.embedding(b.tensor(b.initial_ids()))[0].detach()
    _, whole = local_field_gradients(b, initial.clone().requires_grad_(True), rows, [0, 1, 2], 0, 1, fields)
    shards = []
    for rank in range(4):
        _, shard = local_field_gradients(b, initial.clone().requires_grad_(True), rows, [0, 1, 2], rank, 4, fields)
        shards.append(shard)
    assert torch.count_nonzero(shards[3]) == 0
    assert torch.count_nonzero(whole[-1]) == 0
    torch.testing.assert_close(sum(shards), whole, rtol=3e-4, atol=3e-6)
    expected, _ = project_field_gradients(whole, fields, seed=17, step=3)
    actual, _ = project_field_gradients(sum(shards), fields, seed=17, step=3)
    torch.testing.assert_close(actual, expected, rtol=3e-4, atol=3e-6)


def test_fresh_pause_resume_optimizer_logs_and_evaluation(setup, tmp_path, monkeypatch):
    b, cfg, _ = setup
    configure_finance_pcgrad(cfg)
    cfg.update(epochs=2, early_stopping=False, comparison_epochs=1, output_directory=str(tmp_path / 'paused'))
    row = finance_record()
    monkeypatch.setattr(b, 'generate_many', perfect)
    from medical_kv import training
    save = training.save_checkpoint
    requested = []
    out = tmp_path / 'paused'
    def pause(path, value):
        save(path, value)
        if path.parent == out and path.name == 'latest.pt' and value['state']['step'] == 1 and not requested:
            requested.append(True)
            (out / 'pause_request.json').write_text('{}')
    monkeypatch.setattr(training, 'save_checkpoint', pause)
    initial = b.embedding(b.tensor(b.initial_ids()))[0].detach()
    identity, subsets = {'fixture': 'finance_pcgrad'}, {'development_only': True}
    train_soft(b, cfg, [row], [row], identity, subsets, epoch_validation=False)
    paused = read_checkpoint(out / 'latest.pt')
    torch.testing.assert_close(paused['initial_prompt'], initial, rtol=0, atol=0)
    assert len(paused['optimizer']['state']) == 1
    assert paused['run_info']['focal_fields'] == FOCAL_FIELDS
    assert paused['run_info']['gradient_surgery']['fields'] == sorted(row['fields'])
    train_soft(b, cfg, [row], [row], identity, subsets, resume=out / 'latest.pt')
    resumed = read_checkpoint(out / 'final.pt')
    cfg['output_directory'] = str(tmp_path / 'continuous')
    train_soft(b, cfg, [row], [row], identity, subsets, epoch_validation=False)
    continuous = read_checkpoint(tmp_path / 'continuous/final.pt')
    torch.testing.assert_close(resumed['prompt'], continuous['prompt'], rtol=0, atol=0)
    assert resumed['scheduler'] == continuous['scheduler']
    for a, c in zip(resumed['optimizer']['state'].values(), continuous['optimizer']['state'].values()):
        for key in ('step', 'exp_avg', 'exp_avg_sq'):
            torch.testing.assert_close(a[key], c[key], rtol=0, atol=0)
    for log in map(json.loads, (out / 'training.jsonl').read_text().splitlines()):
        assert log['loss_type'] == 'field_focal_pcgrad'
        assert log['loss'] == pytest.approx(log['ce_other_loss'] + log['focal_loss'])
        assert log['focal_fields'] == FOCAL_FIELDS and log['focal_fields_per_record'] == 5
        assert log['pcgrad_active_fields'] == 13
        assert log['gradient_norm'] == pytest.approx(log['pcgrad_projected_norm'], rel=1e-5)
        assert 'birth_date_focal_loss' not in log and 'focal_prompt_length' not in log
    from medical_kv.resume import validate_resume
    for key, value in [('focal_fields', FOCAL_FIELDS[:-1]), ('loss', {'type': 'field_focal_split'})]:
        changed = copy.deepcopy(resumed['run_info']['resume_contract'])
        changed['config'][key] = value
        with pytest.raises(ValueError, match='config changed'):
            validate_resume(resumed, changed)
    _, actual = evaluate_records(b, [row], vectors=resumed['prompt'])
    cfg['loss']['type'] = 'ce'
    _, expected = evaluate_records(b, [row], vectors=resumed['prompt'])
    assert [r['nll'] for r in actual] == [r['nll'] for r in expected]


@pytest.mark.parametrize('override', [
    {'scene': 'medical'}, {'prompt_initialization': 'old_checkpoint'},
    {'prompt_partition': {'other_length': 16, 'focal_length': 16}},
    {'focal_fields': []}, {'focal_fields': ['phone']}, {'focal_fields': ['name_first']},
    {'focal_fields': ['timestamp', 'timestamp']}])
def test_invalid_finance_pcgrad_config(override):
    with pytest.raises(ValueError):
        load_config('configs/experiment_v13_finance_pcgrad.json', override)


def test_production_configuration_and_isolation():
    cfg = load_config('configs/experiment_v13_finance_pcgrad.json')
    evaluation = load_config('configs/evaluation_v9_finance_pcgrad.json')
    assert {k for k in cfg if cfg[k] != evaluation[k]} == {'output_directory'}
    assert cfg['model_name_or_path'] == '/data1/kevin/model/Qwen3-4B'
    assert cfg['focal_fields'] == FOCAL_FIELDS
    assert cfg['prompt_length'] == 32 and is_pcgrad(cfg) and not is_partitioned(cfg)
    split = load_config('configs/experiment_v12_finance_split_u.json')
    assert {k for k in cfg if cfg[k] != split[k]} == {
        'model_name_or_path', 'output_directory', 'loss', 'prompt_length', 'manual_prompt'}
    assert set(split) - set(cfg) == {'prompt_partition'}
    sys.path.insert(0, str(ROOT / 'scripts'))
    import run_multiscene_experiment as pipeline
    paths = [pipeline.experiment_paths(ROOT, scene, loss) for scene, loss in [
        ('medical', 'birth_focal_pcgrad'), ('medical', 'birth_focal_split'),
        ('finance', 'field_focal_split'), ('finance', 'field_focal_pcgrad')]]
    assert len({str(p[2]) for p in paths}) == len(paths)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(ROOT / 'tools/tokenizer/Qwen3-4B'), local_files_only=True)
    assert len(tok.encode(cfg['public_text'], add_special_tokens=False)) == 56
    assert len(tok.encode(cfg['manual_prompt'], add_special_tokens=False)) == 32


def test_finance_pipeline_uses_independent_configs_and_outputs(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / 'scripts'))
    import run_multiscene_experiment as pipeline
    train, test, run, output, log = pipeline.experiment_paths(tmp_path, 'finance', 'field_focal_pcgrad')
    cfg = load_config('configs/experiment_v13_finance_pcgrad.json')
    train.parent.mkdir()
    train.write_text(json.dumps(cfg))
    test.write_text((ROOT / 'configs/evaluation_v9_finance_pcgrad.json').read_text())
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
    pipeline.run_experiment(tmp_path, '1,2', scene='finance', loss='field_focal_pcgrad', epoch_validation=False)
    assert str(train) in calls[0] and '--no-epoch-validation' in calls[0]
    assert '--resume' not in calls[0]
    assert str(run / 'best.pt') in calls[1] and str(output) in calls[1] and log.exists()
    with pytest.raises(FileExistsError):
        pipeline.run_experiment(tmp_path, '1', scene='finance', loss='field_focal_pcgrad')
    with pytest.raises(ValueError):
        pipeline.experiment_paths(tmp_path, 'medical', 'field_focal_pcgrad')


def test_finance_recovery_routes_to_finance_pcgrad_pipeline(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import training_control as command
    import medical_kv.control as control
    run = tmp_path / 'runs/finance'
    run.mkdir(parents=True)
    (run / 'run.json').write_text(json.dumps({'method': 'soft', 'config': {
        'scene': 'finance', 'question_protocol': 'multiscene_privacy_v1',
        'loss': {'type': 'field_focal_pcgrad', 'gamma': 2, 'alpha': 1}}}))
    monkeypatch.setattr(command, 'ROOT', tmp_path)
    monkeypatch.setattr(command, 'os', SimpleNamespace(name='posix'))
    monkeypatch.setattr(command, 'training_status', lambda *a: {
        'running': False, 'checkpoint_exists': True, 'status': 'paused'})
    monkeypatch.setattr(control, 'process_start', lambda pid: 'fixture')
    calls = []
    monkeypatch.setattr(command.subprocess, 'Popen', lambda args, **kw: calls.append(args) or SimpleNamespace(pid=123))
    command.recover(run, ['1', '2'])
    assert calls[0][calls[0].index('--scene') + 1] == 'finance'
    assert calls[0][calls[0].index('--loss') + 1] == 'field_focal_pcgrad'
    assert '--resume' in calls[0]

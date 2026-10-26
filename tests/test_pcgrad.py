"""PCGrad geometry, field weights, global reduction, independent runs and recovery."""
import copy
import json
import random
import sys
from types import SimpleNamespace

import pytest
import torch

from test_experiment import setup
from test_multiscene_runtime import configure, records, perfect
from medical_kv.config import ROOT, load_config
from medical_kv.evaluation import evaluate_records
from medical_kv.io import checkpoint_compatibility, read_checkpoint
from medical_kv.pcgrad import local_field_gradients, project_field_gradients, global_pcgrad_backward
from medical_kv.tasks import questions_for
from medical_kv.training import train_soft


def configure_pcgrad(cfg):
    configure(cfg)
    cfg.update(loss={'type': 'birth_focal_pcgrad', 'gamma': 2., 'alpha': 1.},
               prompt_initialization='fresh_token_embeddings', scene='medical', split_private_prefill=True)
    cfg['hard']['enabled'] = False


def test_conflicting_pair_matches_closed_form_and_keeps_originals():
    g = torch.tensor([[1., 0.], [-1., 1.]])
    saved = g.clone()
    merged, stats = project_field_gradients(g, ['a', 'b'], seed=17, step=0)
    # a -> (.5,.5), b -> (0,1), both projections use the original other vector.
    torch.testing.assert_close(merged, torch.tensor([.5, 1.5]))
    torch.testing.assert_close(g, saved, rtol=0, atol=0)
    assert stats['pcgrad_projections'] == 2 and stats['pcgrad_active_fields'] == 2
    assert merged @ saved[0] >= 0 and merged @ saved[1] >= 0


@pytest.mark.parametrize('g', [
    [[1., 0.], [0., 2.]], [[1., 0.], [2., 1.]], [[0., 0.], [1., 2.]], [[0., 0.], [0., 0.]]])
def test_nonconflicting_and_zero_tasks_are_unchanged(g):
    gradients = torch.tensor(g)
    result, stats = project_field_gradients(gradients, ['a', 'b'], seed=17, step=1)
    torch.testing.assert_close(result, gradients.sum(0), rtol=0, atol=0)
    assert stats['pcgrad_projections'] == 0


def test_tiny_norms_do_not_overflow_projection_coefficient():
    # Scalar coefficient exceeds FP32, while the correction vector is finite.
    gradients = torch.tensor([[1., 0.], [-1.e-40, 1.e-40]])
    result, _ = project_field_gradients(gradients, ['a', 'b'], seed=17, step=0)
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result, torch.tensor([.5, .5]))


def test_multiple_tasks_match_paper_reference_and_do_not_touch_rng():
    gradients = torch.tensor([[1., 0., 2.], [-2., 1., -1.], [0., -1., 1.], [0., 0., 0.]])
    labels = ['address', 'birth_date', 'diagnosis', 'missing']
    reference = []
    # Independent scalar-list implementation of Algorithm 1 with immutable g_j.
    base = gradients.tolist()
    for i, field in enumerate(labels):
        h = base[i][:]
        others = [j for j in range(3) if j != i]
        random.Random(f'17|pcgrad|23|{field}').shuffle(others)
        for j in others:
            dot = sum(a * b for a, b in zip(h, base[j]))
            if dot < 0:
                scale = dot / sum(x * x for x in base[j])
                h = [a - scale * b for a, b in zip(h, base[j])]
        reference.append(h)
    rng_before = random.getstate(), torch.get_rng_state().clone()
    actual, _ = project_field_gradients(gradients, labels, seed=17, step=23)
    expected = torch.tensor(reference).sum(0)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    assert random.getstate() == rng_before[0]
    assert torch.equal(torch.get_rng_state(), rng_before[1])
    repeated, _ = project_field_gradients(gradients, labels, seed=17, step=23)
    torch.testing.assert_close(actual, repeated, rtol=0, atol=0)


@pytest.mark.parametrize('batched', [True, False])
def test_field_gradients_sum_to_existing_birth_focal_objective_and_share_cache(setup, monkeypatch, batched):
    b, cfg, _ = setup
    configure_pcgrad(cfg)
    cfg['batch_questions'] = batched
    rows = records()
    fields = sorted({f for row in rows for f in row['fields']})
    initial = b.embedding(b.tensor(b.initial_ids()))[0].detach()
    u = initial.clone().requires_grad_(True)
    baseline = sum(b.record_loss(row, vectors=u) for row in rows) / len(rows)
    expected = torch.autograd.grad(baseline, u)[0]
    calls = []
    context = b.context
    def counted(*args, **kwargs):
        ctx = context(*args, **kwargs)
        assert ctx.physical_length == len(b.public_ids)
        calls.append(1)
        return ctx
    monkeypatch.setattr(b, 'context', counted)
    prompt = initial.clone().requires_grad_(True)
    diagnostics = []
    losses, gradients = local_field_gradients(b, prompt, rows, [0, 1, 2], 0, 1, fields, diagnostics=diagnostics)
    assert len(calls) == len(rows)
    assert sum(losses) / 3 == pytest.approx(baseline.item())
    torch.testing.assert_close(gradients.sum(0), expected, rtol=3e-4, atol=3e-6)
    assert all(p.grad is None for p in b.model.parameters())
    assert len(diagnostics) == 3
    # Disabling projection is the same objective as the existing birth_focal run.
    cfg['loss']['type'] = 'birth_focal'
    reference = sum(b.record_loss(row, vectors=prompt) for row in rows) / len(rows)
    torch.testing.assert_close(reference, baseline, rtol=0, atol=0)


def test_field_normalization_and_four_shards_with_missing_and_empty_rank(setup):
    b, cfg, _ = setup
    configure_pcgrad(cfg)
    rows = records()
    fields = sorted({f for row in rows for f in row['fields']})
    initial = b.embedding(b.tensor(b.initial_ids()))[0].detach()
    _, whole = local_field_gradients(b, initial.clone().requires_grad_(True), rows, [0, 1, 2], 0, 1, fields)
    total = torch.zeros_like(whole)
    for rank in range(4):
        u = initial.clone().requires_grad_(True)
        _, shard = local_field_gradients(b, u, rows, [0, 1, 2], rank, 4, fields)
        total += shard
        if rank == 3:
            assert torch.count_nonzero(shard) == 0
    torch.testing.assert_close(total, whole, rtol=3e-4, atol=3e-6)
    a, _ = project_field_gradients(total, fields, seed=17, step=0)
    c, _ = project_field_gradients(whole, fields, seed=17, step=0)
    torch.testing.assert_close(a, c, rtol=3e-4, atol=3e-6)
    # Name and birthday components count as one original field each.
    loss, _, tasks = b.record_loss_components(rows[0], vectors=initial, return_fields=True)
    assert set(tasks) == {'name', 'birth_date', 'diagnosis'}
    torch.testing.assert_close(sum(tasks.values()), loss)


def test_reduce_each_field_before_projection(monkeypatch):
    from medical_kv import pcgrad
    local = torch.tensor([[[1., 0.]], [[-1., 1.]]])
    remote = torch.tensor([[[0., 2.]], [[2., 0.]]])
    fields = ['a', 'b']
    monkeypatch.setattr(pcgrad, 'local_field_gradients', lambda *a, **kw: ([1.], local.clone()))
    reductions = []
    def reduce(value):
        reductions.append(value.shape)
        value.add_(remote)
    runtime = SimpleNamespace(rank=0, world_size=2, sum_=reduce)
    u = torch.zeros(1, 2, requires_grad=True)
    global_pcgrad_backward(None, u, [], [0], runtime, fields, seed=17, step=0)
    expected, _ = project_field_gradients(local + remote, fields, seed=17, step=0)
    wrong = sum(project_field_gradients(g, fields, seed=17, step=0)[0] for g in (local, remote))
    torch.testing.assert_close(u.grad, expected)
    assert reductions == [local.shape]
    assert not torch.allclose(u.grad, wrong)


def test_pcgrad_fresh_training_pause_resume_logs_and_nll(setup, tmp_path, monkeypatch):
    b, cfg, _ = setup
    configure_pcgrad(cfg)
    cfg.update(epochs=2, early_stopping=False, comparison_epochs=1, output_directory=str(tmp_path / 'pcgrad'))
    row = records()[0]
    monkeypatch.setattr(b, 'generate_many', perfect)
    out = tmp_path / 'pcgrad'
    from medical_kv import training
    save = training.save_checkpoint
    pause_once = [True]
    def save_and_pause(path, value):
        save(path, value)
        if path.parent == out and path.name == 'latest.pt' and value['state']['step'] == 1 and pause_once[0]:
            pause_once[0] = False
            (out / 'pause_request.json').write_text('{}')
    monkeypatch.setattr(training, 'save_checkpoint', save_and_pause)
    initial = b.embedding(b.tensor(b.initial_ids()))[0].detach()
    train_soft(b, cfg, [row], [row], {'fixture': 'pcgrad'}, {'development_only': True}, epoch_validation=False)
    paused = read_checkpoint(out / 'latest.pt')
    torch.testing.assert_close(paused['initial_prompt'], initial, rtol=0, atol=0)
    assert len(paused['optimizer']['state']) == 1
    assert paused['run_info']['gradient_surgery']['fields'] == ['birth_date', 'diagnosis', 'name']
    assert paused['run_info']['prompt_initialization']['mode'] == 'fresh_token_embeddings'
    train_soft(b, cfg, [row], [row], {'fixture': 'pcgrad'}, {'development_only': True}, resume=out / 'latest.pt')
    resumed = read_checkpoint(out / 'final.pt')
    cfg['output_directory'] = str(tmp_path / 'continuous')
    train_soft(b, cfg, [row], [row], {'fixture': 'pcgrad'}, {'development_only': True}, epoch_validation=False)
    continuous = read_checkpoint(tmp_path / 'continuous/final.pt')
    torch.testing.assert_close(resumed['prompt'], continuous['prompt'], rtol=0, atol=0)
    assert resumed['run_signature'] == paused['run_signature']
    for a, c in zip(resumed['optimizer']['state'].values(), continuous['optimizer']['state'].values()):
        for key in ('step', 'exp_avg', 'exp_avg_sq'):
            torch.testing.assert_close(a[key], c[key], rtol=0, atol=0)
    assert all(p.grad is None for p in b.model.parameters())
    for log in map(json.loads, (out / 'training.jsonl').read_text().splitlines()):
        assert log['loss_type'] == 'birth_focal_pcgrad'
        assert log['pcgrad_active_fields'] == 3
        assert log['loss'] == pytest.approx(log['ce_other_loss'] + log['focal_loss'])
        assert log['gradient_norm'] == pytest.approx(log['pcgrad_projected_norm'], rel=1e-5)
    checkpoint_compatibility(resumed, b)
    from medical_kv.resume import validate_resume
    changed = copy.deepcopy(resumed['run_info']['resume_contract'])
    changed['config']['loss']['type'] = 'birth_focal'
    with pytest.raises(ValueError, match='config changed'):
        validate_resume(resumed, changed)
    _, actual = evaluate_records(b, [row], vectors=resumed['prompt'])
    cfg['loss']['type'] = 'birth_focal'
    _, expected = evaluate_records(b, [row], vectors=resumed['prompt'])
    assert [v['nll'] for v in actual] == [v['nll'] for v in expected]


@pytest.mark.parametrize('relocated_model', [None, '/data1/kevin/model/Qwen3-4B'])
def test_production_config_and_independent_routes(tmp_path, monkeypatch, relocated_model):
    cfg = load_config('configs/experiment_v11_medical_pcgrad.json')
    baseline = load_config('configs/experiment_v9_medical_birth_focal.json')
    evaluation = load_config('configs/evaluation_v7_medical_pcgrad.json')
    if relocated_model is not None:
        cfg['model_name_or_path'] = relocated_model
        evaluation['model_name_or_path'] = relocated_model
    # A new server may relocate only this experiment's model. Deployment paths
    # need not match the historical control config; training settings still must.
    assert {k for k in baseline if k != 'model_name_or_path' and baseline[k] != cfg[k]} == {'loss', 'output_directory'}
    assert evaluation['model_name_or_path'] == cfg['model_name_or_path']
    assert set(cfg) - set(baseline) == {'prompt_initialization'}
    assert cfg['prompt_length'] == 32 and cfg['loss']['gamma'] == 2
    sys.path.insert(0, str(ROOT / 'scripts'))
    import run_multiscene_experiment as pipeline
    train, test, run, output, log = pipeline.experiment_paths(tmp_path, 'medical', 'birth_focal_pcgrad')
    train.parent.mkdir()
    train.write_text(json.dumps(cfg))
    test.write_text(json.dumps(evaluation))
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
    pipeline.run_experiment(tmp_path, '0,1,3', scene='medical', loss='birth_focal_pcgrad', epoch_validation=False)
    assert str(train) in calls[0] and '--no-epoch-validation' in calls[0]
    assert '--resume' not in calls[0] and '--initialize-from' not in calls[0]
    assert str(run / 'best.pt') in calls[1] and str(output) in calls[1]
    assert log.exists()
    with pytest.raises(FileExistsError):
        pipeline.run_experiment(tmp_path, '0', scene='medical', loss='birth_focal_pcgrad')
    with pytest.raises(ValueError):
        pipeline.experiment_paths(tmp_path, 'hr', 'birth_focal_pcgrad')
    cfg['prompt_partition'] = {'other_length': 16, 'birth_length': 16}
    with pytest.raises(ValueError, match='prompt_partition'):
        load_config(cfg)

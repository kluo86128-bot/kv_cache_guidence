"""Field-routed gradients, fresh initialization, one cache and exact CPU continuation."""
import copy
import json
import sys

import pytest
import torch

from test_experiment import setup
from test_multiscene_runtime import configure, records, perfect
from medical_kv.config import ROOT, load_config
from medical_kv.io import read_checkpoint, checkpoint_compatibility
from medical_kv.prompt_partition import optimizer_parameters, assign_partition_gradients
from medical_kv.tasks import grouped_loss, questions_for
from medical_kv.training import local_batch_backward, train_soft


def split_config(cfg):
    configure(cfg)
    cfg.update(prompt_length=5, prompt_partition={'other_length': 3, 'birth_length': 2},
               prompt_initialization='fresh_token_embeddings',
               loss={'type': 'birth_focal_split', 'gamma': 2., 'alpha': 1.},
               scene='medical', split_private_prefill=True)
    cfg['hard']['enabled'] = False


@pytest.mark.parametrize('batched', [False, True])
@pytest.mark.parametrize('record_index', [0, 1, 2])
def test_routed_gradient_matches_separate_objectives_and_one_cache(setup, monkeypatch, batched, record_index):
    b, cfg, _ = setup
    split_config(cfg)
    cfg['batch_questions'] = batched
    row = records()[record_index]
    initial = b.embedding(b.tensor(b.initial_ids()))[0].detach()
    u = initial.clone().requires_grad_(True)
    ctx = b.context(row, vectors=u)
    assert ctx.physical_length == len(b.public_ids)
    assert ctx.public_start == len(b.private_ids(row)) + 5 + len(b.separator_ids)
    qs = questions_for(row, cfg)
    values = b.answer_losses(ctx, qs, training=True)
    selected = torch.tensor([q['group'] == 'birth_date' for q in qs])
    other = grouped_loss(values[:, 1].masked_fill(selected, 0), qs)
    birth = grouped_loss(values[:, 0].masked_fill(~selected, 0), qs)
    go = torch.autograd.grad(other, u, retain_graph=True)[0]
    gd = torch.autograd.grad(birth, u)[0]
    expected = torch.cat((go[:3], gd[3:]))
    # Each objective can affect BOTH segments through the shared cache.
    assert go[3:].norm() > 0
    if selected.any():
        assert gd[:3].norm() > 0
    calls = []
    context = b.context
    def counted(*a, **kw):
        calls.append(1)
        return context(*a, **kw)
    monkeypatch.setattr(b, 'context', counted)
    u2 = initial.clone().requires_grad_(True)
    diagnostics = []
    losses = local_batch_backward(b, u2, [row], [0], 0, 1, diagnostics=diagnostics)
    assert len(calls) == 1
    torch.testing.assert_close(u2.grad, expected, rtol=3e-4, atol=3e-6)
    assert losses[0] == pytest.approx((other + birth).item())
    assert losses[0] == pytest.approx(diagnostics[0]['ce_other_loss'] + diagnostics[0]['focal_loss'])
    if not selected.any():
        assert torch.count_nonzero(u2.grad[3:]) == 0
    assert all(p.grad is None for p in b.model.parameters())


def test_routed_global_batch_matches_four_shards_with_empty_rank(setup):
    b, cfg, _ = setup
    split_config(cfg)
    initial = b.embedding(b.tensor(b.initial_ids()))[0].detach()
    whole = initial.clone().requires_grad_(True)
    rows = records()
    local_batch_backward(b, whole, rows, [0, 1, 2], 0, 1)
    summed = torch.zeros_like(initial)
    for rank in range(4):
        u = initial.clone().requires_grad_(True)
        local_batch_backward(b, u, rows, [0, 1, 2], rank, 4)
        summed += u.grad
    torch.testing.assert_close(summed, whole.grad, rtol=3e-4, atol=3e-6)


def test_inactive_segment_skips_adam_momentum_and_step():
    cfg = {'loss': {'type': 'birth_focal_split'}, 'prompt_partition': {'other_length': 3}}
    initial = torch.randn(5, 4)
    params = optimizer_parameters(initial, cfg)
    optimizer = torch.optim.AdamW(params, lr=.1, weight_decay=0)
    rows = [{'fields': {'birth_date': '2000-01-01', 'name': 'Ax'}}, {'fields': {'name': 'Ax'}}]
    assign_partition_gradients(params, torch.ones_like(initial), rows, [0])
    optimizer.step()
    saved = params[1].detach().clone()
    previous_step = optimizer.state[params[1]]['step'].clone()
    optimizer.zero_grad(set_to_none=True)
    assign_partition_gradients(params, torch.ones_like(initial), rows, [1])
    optimizer.step()
    torch.testing.assert_close(params[1], saved, rtol=0, atol=0)
    assert optimizer.state[params[1]]['step'] == previous_step
    assert optimizer.state[params[0]]['step'] == previous_step + 1


def test_split_initialization_pause_resume_and_evaluation(setup, tmp_path, monkeypatch):
    b, cfg, _ = setup
    split_config(cfg)
    cfg.update(epochs=2, comparison_epochs=1, early_stopping=False, output_directory=str(tmp_path / 'split'))
    monkeypatch.setattr(b, 'generate_many', perfect)
    expected_initial = b.embedding(b.tensor(b.initial_ids()))[0].detach().float()
    row = records()[0]
    out = tmp_path / 'split'
    from medical_kv import training
    save = training.save_checkpoint
    pause_once = [True]
    def save_and_pause(path, value):
        save(path, value)
        if path.parent == out and path.name == 'latest.pt' and value['state']['step'] == 1 and pause_once[0]:
            pause_once[0] = False
            (out / 'pause_request.json').write_text('{}')
    monkeypatch.setattr(training, 'save_checkpoint', save_and_pause)
    def reject_checkpoint_read(*args, **kwargs):
        raise AssertionError('Fresh training must not read any old checkpoint')
    monkeypatch.setattr(training, 'read_checkpoint', reject_checkpoint_read)
    train_soft(b, cfg, [row], [row], {'fixture': 'split'}, {'development_only': True},
               epoch_validation=False)
    paused = read_checkpoint(out / 'latest.pt')
    torch.testing.assert_close(paused['initial_prompt'], expected_initial, rtol=0, atol=0)
    assert paused['prompt'].shape == (5, 32)
    assert paused['state']['step'] == 1
    assert len(paused['optimizer']['param_groups'][0]['params']) == 2
    assert paused['run_info']['prompt_initialization']['mode'] == 'fresh_token_embeddings'
    assert 'checkpoint' not in paused['run_info']['prompt_initialization']
    assert (paused['prompt'][:3] - expected_initial[:3]).norm() > 0
    assert (paused['prompt'][3:] - expected_initial[3:]).norm() > 0
    assert paused['run_info']['shared_public_cache']
    assert paused['run_info']['prompt_partition'] == {'other_length': 3, 'birth_length': 2}
    checkpoint_compatibility(paused, b)
    # Recovery reads only this new experiment's saved U/Adam/metadata.
    monkeypatch.setattr(training, 'read_checkpoint', read_checkpoint)
    train_soft(b, cfg, [row], [row], {'fixture': 'split'}, {'development_only': True}, resume=out / 'latest.pt')
    resumed = read_checkpoint(out / 'final.pt')
    assert resumed['state']['step'] == 2 and resumed['state']['epoch'] == 2
    assert resumed['run_signature'] == paused['run_signature']
    # An independent fresh run reproduces both initialization and updates.
    cfg['output_directory'] = str(tmp_path / 'continuous')
    train_soft(b, cfg, [row], [row], {'fixture': 'split'}, {'development_only': True},
               epoch_validation=False)
    continuous = read_checkpoint(tmp_path / 'continuous/final.pt')
    torch.testing.assert_close(resumed['prompt'], continuous['prompt'], rtol=0, atol=0)
    for a, c in zip(resumed['optimizer']['state'].values(), continuous['optimizer']['state'].values()):
        for key in ('step', 'exp_avg', 'exp_avg_sq'):
            torch.testing.assert_close(a[key], c[key], rtol=0, atol=0)
    logs = [json.loads(line) for line in (out / 'training.jsonl').read_text().splitlines()]
    assert all(r['loss_type'] == 'birth_focal_split' for r in logs)
    assert all(r['loss'] == pytest.approx(r['ce_other_loss'] + r['focal_loss']) for r in logs)
    from medical_kv.resume import validate_resume
    changed = copy.deepcopy(resumed['run_info']['resume_contract'])
    changed['config']['prompt_partition']['other_length'] = 2
    with pytest.raises(ValueError, match='config changed'):
        validate_resume(resumed, changed)
    # Evaluation remains ordinary CE even with the routed training config.
    u = resumed['prompt']
    ctx = b.context(row, vectors=u)
    qs = questions_for(row, cfg)
    nll = b.answer_losses(ctx, qs)
    cfg['loss']['type'] = 'ce'
    torch.testing.assert_close(nll, b.answer_losses(ctx, qs), rtol=0, atol=0)


def test_partition_config_requires_fresh_initialization():
    cfg = load_config('configs/experiment_v10_medical_split_u.json')
    cfg.pop('prompt_initialization')
    with pytest.raises(ValueError, match='fresh_token_embeddings'):
        load_config(cfg)


def test_production_config_and_routes():
    cfg = load_config('configs/experiment_v10_medical_split_u.json')
    assert cfg['prompt_length'] == 48 and cfg['prompt_partition'] == {'other_length': 32, 'birth_length': 16}
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / 'tools/tokenizer/Qwen3-4B'), local_files_only=True)
    assert len(tokenizer.encode(cfg['public_text'], add_special_tokens=False)) == 56
    assert len(tokenizer.encode(cfg['manual_prompt'], add_special_tokens=False)) == 48
    sys.path.insert(0, str(ROOT / 'scripts'))
    from run_multiscene_experiment import experiment_paths
    paths = experiment_paths(ROOT, 'medical', 'birth_focal_split')
    assert paths[0].name == 'experiment_v10_medical_split_u.json'
    assert paths[1].name == 'evaluation_v6_medical_split_u.json'
    assert paths[2].name == 'soft_medical_v2_clean_split_u32_16_fresh_seed17_100epochs'
    with pytest.raises(ValueError):
        experiment_paths(ROOT, 'hr', 'birth_focal_split')
    cfg['prompt_partition']['birth_length'] = 15
    with pytest.raises(ValueError, match='prompt_partition'):
        load_config(cfg)


def test_partition_pipeline_starts_fresh_and_tests_new_checkpoint(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / 'scripts'))
    import run_multiscene_experiment as pipeline
    train, test, run, output, log = pipeline.experiment_paths(tmp_path, 'medical', 'birth_focal_split')
    train.parent.mkdir()
    cfg = json.loads((ROOT / 'configs/experiment_v10_medical_split_u.json').read_text())
    train.write_text(json.dumps(cfg))
    test.write_text((ROOT / 'configs/evaluation_v6_medical_split_u.json').read_text())
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
    pipeline.run_experiment(tmp_path, '0,1,3', scene='medical', loss='birth_focal_split',
                            epoch_validation=False)
    assert '--initialize-from' not in calls[0] and '--resume' not in calls[0]
    assert '--no-epoch-validation' in calls[0]
    assert '--initialize-from' not in calls[1]
    assert str(run / 'best.pt') in calls[1] and str(output) in calls[1] and log.exists()

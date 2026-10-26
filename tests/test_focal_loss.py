"""Focal math, public-KV gradients, NLL semantics and independent run routing."""
import copy
import json
import sys

import pytest
import torch

from test_experiment import setup
from test_multiscene_runtime import configure, records, perfect
from medical_kv.config import ROOT, load_config
from medical_kv.evaluation import evaluate_records
from medical_kv.io import read_checkpoint
from medical_kv.tasks import questions_for, grouped_loss
from medical_kv.training import local_batch_backward, train_soft


def focal_config(cfg, gamma=2.0):
    configure(cfg)
    cfg.update(loss={'type': 'focal', 'gamma': gamma, 'alpha': 1.0},
               split_private_prefill=True, question_batch_size=4)
    cfg['hard']['enabled'] = False


@pytest.mark.parametrize('gamma', [0.0, 0.5, 2.0])
def test_token_math_gradient_mask_and_fp32(setup, gamma):
    backend, cfg, _ = setup
    focal_config(cfg, gamma)
    logits = torch.tensor([[[2., -1., 0.], [-2., 1., 0.], [1., 0., 2.]],
                           [[-1., 2., 0.], [1., 2., 3.], [3., 2., 1.]]], requires_grad=True)
    labels = torch.tensor([[0, 2, 1], [1, -100, -100]])
    actual = backend._mean_answer_losses(logits, labels, training=True)
    p = logits.softmax(-1)
    expected = torch.stack([torch.stack([
        -(1 - p[i, j, label]).pow(gamma) * p[i, j, label].log()
        for j, label in enumerate(row) if label != -100]).mean()
        for i, row in enumerate(labels)])
    torch.testing.assert_close(actual[:, 0], expected)
    grad = torch.autograd.grad(actual[:, 0].sum(), logits, retain_graph=True)[0]
    reference = torch.autograd.grad(expected.sum(), logits)[0]
    torch.testing.assert_close(grad, reference)
    assert torch.count_nonzero(grad[1, 1:]) == 0
    if gamma == 0:
        torch.testing.assert_close(actual[:, 0], actual[:, 1])
    half = logits.detach().to(torch.bfloat16).requires_grad_(True)
    values = backend._mean_answer_losses(half, labels, training=True)
    assert values.dtype == torch.float32
    values[:, 0].sum().backward()
    assert torch.isfinite(half.grad).all()
    # Saturated correct predictions and masked positions must not yield NaN gradients.
    saturated = torch.tensor([[[1000., -1000.], [0., 0.]]], requires_grad=True)
    backend._mean_answer_losses(saturated, torch.tensor([[0, -100]]), training=True)[:, 0].sum().backward()
    assert torch.isfinite(saturated.grad).all()


@pytest.mark.parametrize('gamma', [0.0, 2.0])
def test_focal_chunk_checkpoint_matches_serial_and_reaches_u(setup, gamma):
    backend, cfg, _ = setup
    focal_config(cfg, gamma)
    row = records()[0]  # includes an empty name component (EOS-only target).
    qs = questions_for(row, cfg)
    initial = backend.embedding(backend.tensor(backend.initial_ids()))[0].detach()
    reference = None
    for batched in (False, True):
        cfg['batch_questions'] = batched
        prompt = initial.clone().requires_grad_(True)
        ctx = backend.context(row, vectors=prompt)
        values = backend.answer_losses(ctx, qs, training=True)
        nll = backend.answer_losses(ctx, qs)
        torch.testing.assert_close(values[:, 1], nll)
        loss = grouped_loss(values[:, 0], qs)
        if gamma == 0:
            torch.testing.assert_close(loss, grouped_loss(nll, qs))
        gradient = torch.autograd.grad(loss, prompt)[0]
        assert torch.isfinite(gradient).all() and gradient.norm() > 0
        if reference is None:
            reference = loss.detach(), gradient
        else:
            torch.testing.assert_close(loss, reference[0], rtol=1e-5, atol=2e-6)
            torch.testing.assert_close(gradient, reference[1], rtol=3e-4, atol=3e-6)
    assert all(p.grad is None for p in backend.model.parameters())


@pytest.mark.parametrize('loss_type', ['focal', 'ce_focal_birth', 'birth_focal'])
def test_focal_global_batch_with_empty_rank_and_nll_evaluation(setup, tmp_path, monkeypatch, loss_type):
    backend, cfg, _ = setup
    focal_config(cfg)
    cfg['loss']['type'] = loss_type
    if loss_type == 'ce_focal_birth':
        cfg['loss'].update(type=loss_type, weight=0.5)
    rows = records()
    initial = backend.embedding(backend.tensor(backend.initial_ids()))[0].detach()
    prompt = initial.clone().requires_grad_(True)
    expected = sum(backend.record_loss(r, vectors=prompt) for r in rows) / len(rows)
    expected.backward()
    gradient = torch.zeros_like(prompt)
    diagnostics, losses = [], []
    for rank in range(4):
        shard = initial.clone().requires_grad_(True)
        losses.extend(local_batch_backward(backend, shard, rows, [0, 1, 2], rank, 4, diagnostics=diagnostics))
        gradient += shard.grad
    torch.testing.assert_close(gradient, prompt.grad, rtol=3e-4, atol=3e-6)
    assert sum(losses) / 3 == pytest.approx(expected.item())
    assert all(r['ce_loss'] > r['focal_loss'] for r in diagnostics)
    monkeypatch.setattr(backend, 'generate_many', perfect)
    _, focal_rows = evaluate_records(backend, rows, vectors=initial, output=tmp_path / 'focal')
    cfg.pop('loss')
    _, ce_rows = evaluate_records(backend, rows, vectors=initial, output=tmp_path / 'ce')
    assert [r['nll'] for r in focal_rows] == [r['nll'] for r in ce_rows]
    assert [r['em'] for r in focal_rows] == [r['em'] for r in ce_rows]


def test_focal_tiny_training_logs_resume_identity_and_validation(setup, tmp_path, monkeypatch):
    backend, cfg, _ = setup
    focal_config(cfg)
    cfg.update(epochs=2, comparison_epochs=1, early_stopping=False,
               scene='medical', output_directory=str(tmp_path / 'focal_run'))
    row = records()[0]
    monkeypatch.setattr(backend, 'generate_many', perfect)
    out = tmp_path / 'focal_run'
    identity, subsets = {'fixture': 'focal'}, {'development_only': True}
    from medical_kv import training
    save = training.save_checkpoint
    pause_once = [True]
    def save_and_pause(path, value):
        save(path, value)
        if path.name == 'latest.pt' and value['state']['step'] == 1 and pause_once[0]:
            pause_once[0] = False
            (out / 'pause_request.json').write_text('{}')
    monkeypatch.setattr(training, 'save_checkpoint', save_and_pause)
    train_soft(backend, cfg, [row], [row], identity, subsets, epoch_validation=False)
    checkpoint = read_checkpoint(out / 'latest.pt')
    from medical_kv.resume import validate_resume
    changed = copy.deepcopy(checkpoint['run_info']['resume_contract'])
    changed['config']['loss']['gamma'] = 1.0
    with pytest.raises(ValueError, match='config changed'):
        validate_resume(checkpoint, changed)
    train_soft(backend, cfg, [row], [row], identity, subsets, resume=out / 'latest.pt')
    resumed = read_checkpoint(out / 'final.pt')
    assert resumed['state']['epoch'] == 2 and resumed['state']['step'] == 2
    assert resumed['run_signature'] == checkpoint['run_signature']
    # A new run completes two epochs and writes ordinary validation NLL.
    cfg.update(max_updates=None, output_directory=str(tmp_path / 'complete'))
    train_soft(backend, cfg, [row], [row], identity, subsets, epoch_validation=False)
    complete = tmp_path / 'complete'
    logs = [json.loads(line) for line in (complete / 'training.jsonl').read_text().splitlines()]
    assert len(logs) == 2
    assert all(r['loss_type'] == 'focal' and r['loss'] == r['focal_loss'] < r['ce_loss'] for r in logs)
    assert (complete / 'loss_curve.png').is_file()
    saved = read_checkpoint(complete / 'best.pt')
    assert saved['run_info']['config']['loss'] == cfg['loss']
    assert 'focal' in saved['run_info']['loss_definition']
    assert saved['state']['epoch'] == 2
    torch.testing.assert_close(saved['prompt'], resumed['prompt'], rtol=0, atol=0)


@pytest.mark.parametrize('options', [None, {}, {'type':'other'}, {'type':'focal', 'gamma':-1, 'alpha':1},
    {'type':'focal', 'gamma':float('nan'), 'alpha':1}, {'type':'focal', 'gamma':2, 'alpha':0},
    {'type':'ce', 'gamma':2}])
def test_invalid_loss_config_rejected(options):
    with pytest.raises(ValueError):
        load_config('configs/experiment_v7_medical_focal.json', {'loss': options or {}})


@pytest.mark.parametrize('loss_type,train_file,test_file', [
    ('focal', 'experiment_v7_medical_focal.json', 'evaluation_v3_medical_focal.json'),
    ('ce_focal_birth', 'experiment_v8_medical_ce_focal_birth.json', 'evaluation_v4_medical_ce_focal_birth.json'),
    ('birth_focal', 'experiment_v9_medical_birth_focal.json', 'evaluation_v5_medical_birth_focal.json')])
def test_focal_configs_and_pipeline_routing(tmp_path, monkeypatch, loss_type, train_file, test_file):
    sys.path.insert(0, str(ROOT / 'scripts'))
    import run_multiscene_experiment as pipeline
    ce = load_config('configs/experiment_v6_medical_no_negatives.json')
    focal = load_config('configs/' + train_file)
    assert {k for k in ce if ce[k] != focal[k]} == {'output_directory'}
    assert set(focal) - set(ce) == {'loss'}
    train, test, run, output, log = pipeline.experiment_paths(tmp_path, 'medical', loss_type)
    train.parent.mkdir()
    train.write_text(json.dumps(focal))
    test.write_text(json.dumps(load_config('configs/' + test_file)))
    calls = []
    def stage(command, **kwargs):
        calls.append(command)
        if len(calls) == 1:
            run.mkdir(parents=True)
            (run / 'best.pt').write_bytes(b'fixture')
            (run / 'run.json').write_text(json.dumps({'config': focal, 'run_signature': 'fixture'}))
            (run / 'run_summary.json').write_text(json.dumps({'status':'complete', 'partial_dataset':False,
                'stop_reason':'epochs_completed', 'state':{'epoch':100}, 'run_signature':'fixture'}))
    monkeypatch.setattr(pipeline, 'run_child', stage)
    pipeline.run_experiment(tmp_path, '0,1,3', scene='medical', loss=loss_type, epoch_validation=False)
    assert len(calls) == 2 and '--no-epoch-validation' in calls[0]
    assert str(train) in calls[0] and str(test) in calls[1]
    assert str(run / 'best.pt') in calls[1] and str(output) in calls[1] and log.exists()
    with pytest.raises(FileExistsError):
        pipeline.run_experiment(tmp_path, '0', scene='medical', loss=loss_type)


@pytest.mark.parametrize('loss_type', ['focal', 'ce_focal_birth', 'birth_focal', 'birth_focal_split', 'birth_focal_pcgrad'])
def test_focal_recovery_selects_focal_pipeline(tmp_path, monkeypatch, loss_type):
    from types import SimpleNamespace
    import training_control as command
    import medical_kv.control as control
    run = tmp_path / 'runs/focal'
    run.mkdir(parents=True)
    (run / 'run.json').write_text(json.dumps({'method':'soft', 'config':{
        'scene':'medical', 'question_protocol':'multiscene_privacy_v1',
        'loss':{'type':loss_type, 'gamma':2, 'alpha':1, **({'weight':0.5} if loss_type == 'ce_focal_birth' else {})}}}))
    # Existing CE results must not interfere with Focal recovery.
    (tmp_path / 'runs/test_medical_v2_clean_best').mkdir()
    monkeypatch.setattr(command, 'ROOT', tmp_path)
    monkeypatch.setattr(command, 'os', SimpleNamespace(name='posix'))
    monkeypatch.setattr(command, 'training_status', lambda *a: {'running':False, 'checkpoint_exists':True, 'status':'paused'})
    monkeypatch.setattr(control, 'process_start', lambda pid: 'fixture')
    calls = []
    monkeypatch.setattr(command.subprocess, 'Popen', lambda args, **kw: calls.append(args) or SimpleNamespace(pid=123))
    command.recover(run, ['0','1','3'])
    assert calls[0][calls[0].index('--loss') + 1] == loss_type
    assert calls[0][calls[0].index('--scene') + 1] == 'medical'


@pytest.mark.parametrize('record_index,weight', [(0,0.0), (0,0.5), (1,0.5), (2,0.5)])
def test_joint_keeps_ce_gradient_and_only_adds_birth_focal(setup, record_index, weight):
    backend, cfg, _ = setup
    focal_config(cfg)
    cfg['loss'].update(type='ce_focal_birth', weight=weight)
    row = records()[record_index]
    qs = questions_for(row,cfg)
    prompt = backend.embedding(backend.tensor(backend.initial_ids()))[0].detach().clone().requires_grad_(True)
    ctx = backend.context(row, vectors=prompt)
    nll = backend.answer_losses(ctx, qs)
    reference_ce = grouped_loss(nll, qs)
    focals = torch.stack([backend.answer_loss(ctx, q, training=True)[0] for q in qs])
    birth = [i for i,q in enumerate(qs) if q['group'] == 'birth_date']
    field_count = len({q['group'] for q in qs})
    extra = focals[birth].mean() / field_count if birth else reference_ce * 0
    reference = reference_ce + weight * extra
    expected_gradient = torch.autograd.grad(reference,prompt)[0]
    actual,stats = backend.record_loss_components(row,vectors=prompt)
    actual_gradient = torch.autograd.grad(actual,prompt)[0]
    torch.testing.assert_close(actual,reference,rtol=1e-5,atol=2e-6)
    torch.testing.assert_close(actual_gradient,expected_gradient,rtol=3e-4,atol=3e-6)
    assert stats['focal_loss'].item() == pytest.approx(extra.item(),rel=1e-5)
    assert actual.item() == pytest.approx((stats['ce_loss']+stats['focal_weighted_loss']).item())
    if not birth or weight == 0:
        torch.testing.assert_close(actual,reference_ce)


def test_joint_tiny_training_logs_and_resume_parameter_guard(setup,tmp_path,monkeypatch):
    backend,cfg,_=setup
    focal_config(cfg)
    cfg['loss'].update(type='ce_focal_birth',weight=0.5)
    cfg.update(epochs=1,comparison_epochs=1,early_stopping=False,scene='medical',output_directory=str(tmp_path/'joint'))
    monkeypatch.setattr(backend,'generate_many',perfect)
    train_soft(backend,cfg,[records()[0]],[records()[0]],{'fixture':'joint'},{'development_only':True},epoch_validation=False)
    out=tmp_path/'joint'
    log=json.loads((out/'training.jsonl').read_text().splitlines()[0])
    assert log['loss_type']=='ce_focal_birth' and log['focal_scope']=='birth_date'
    assert log['loss']==pytest.approx(log['ce_loss']+log['focal_weighted_loss'])
    assert log['focal_weighted_loss']==pytest.approx(0.5*log['focal_loss'])
    assert log['birth_date_focal_loss']==pytest.approx(3*log['focal_loss'])
    saved=read_checkpoint(out/'best.pt')
    from medical_kv.resume import validate_resume
    changed=copy.deepcopy(saved['run_info']['resume_contract'])
    changed['config']['loss']['weight']=1.0
    with pytest.raises(ValueError,match='config changed'):
        validate_resume(saved,changed)


@pytest.mark.parametrize('weight', [-1,float('nan'),True])
def test_joint_invalid_weight_rejected(weight):
    cfg=load_config('configs/experiment_v8_medical_ce_focal_birth.json')
    cfg['loss']['weight']=weight
    with pytest.raises(ValueError,match='gamma/weight'):
        load_config(cfg)


@pytest.mark.parametrize('record_index,gamma', [(0,2.0),(1,2.0),(2,2.0),(0,0.0)])
def test_birth_focal_replaces_birth_ce_and_preserves_other_ce_gradient(setup,record_index,gamma):
    backend,cfg,_=setup
    focal_config(cfg,gamma)
    cfg['loss']['type']='birth_focal'
    row=records()[record_index]
    qs=questions_for(row,cfg)
    prompt=backend.embedding(backend.tensor(backend.initial_ids()))[0].detach().clone().requires_grad_(True)
    ctx=backend.context(row,vectors=prompt)
    # Independently choose one loss per question, then aggregate original fields.
    reference_values=[]
    nlls=[]
    for q in qs:
        ce=backend.answer_loss(ctx,q)
        nlls.append(ce)
        if q['group']=='birth_date':
            reference_values.append(backend.answer_loss(ctx,q,training=True)[0])
        else:
            reference_values.append(ce)
    reference=grouped_loss(torch.stack(reference_values),qs)
    ce_reference=grouped_loss(torch.stack(nlls),qs)
    expected_grad=torch.autograd.grad(reference,prompt)[0]
    actual,stats=backend.record_loss_components(row,vectors=prompt)
    actual_grad=torch.autograd.grad(actual,prompt)[0]
    torch.testing.assert_close(actual,reference,rtol=1e-5,atol=2e-6)
    torch.testing.assert_close(actual_grad,expected_grad,rtol=3e-4,atol=3e-6)
    torch.testing.assert_close(stats['ce_loss'],ce_reference)
    assert actual.item()==pytest.approx((stats['ce_other_loss']+stats['focal_loss']).item())
    if gamma==0 or not any(q['group']=='birth_date' for q in qs):
        torch.testing.assert_close(actual,ce_reference)
    else:
        assert actual.item()<ce_reference.item()


def test_birth_focal_tiny_training_logs_and_rejects_old_loss_resume(setup,tmp_path,monkeypatch):
    backend,cfg,_=setup
    focal_config(cfg)
    cfg['loss']['type']='birth_focal'
    cfg.update(epochs=1,comparison_epochs=1,early_stopping=False,scene='medical',output_directory=str(tmp_path/'birth_focal'))
    monkeypatch.setattr(backend,'generate_many',perfect)
    train_soft(backend,cfg,[records()[0]],[records()[0]],{'fixture':'birth_focal'},{'development_only':True},epoch_validation=False)
    out=tmp_path/'birth_focal'
    log=json.loads((out/'training.jsonl').read_text().splitlines()[0])
    assert log['loss_type']=='birth_focal' and log['ce_objective_scope']=='other_fields'
    assert log['loss']==pytest.approx(log['ce_other_loss']+log['focal_loss'])
    assert log['birth_date_focal_loss']==pytest.approx(3*log['focal_loss'])
    assert 'focal_weight' not in log
    saved=read_checkpoint(out/'best.pt')
    assert 'Birth-date Focal only' in saved['run_info']['loss_definition']
    from medical_kv.resume import validate_resume
    changed=copy.deepcopy(saved['run_info']['resume_contract'])
    changed['config']['loss']={'type':'ce_focal_birth','gamma':2.0,'alpha':1.0,'weight':0.5}
    with pytest.raises(ValueError,match='config changed'):
        validate_resume(saved,changed)
    cfg=load_config('configs/experiment_v9_medical_birth_focal.json')
    cfg['loss']['weight']=0.5
    with pytest.raises(ValueError,match='Invalid loss keys'):
        load_config(cfg)

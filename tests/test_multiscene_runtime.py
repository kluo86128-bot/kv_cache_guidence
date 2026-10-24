"""New-schema loss, cache, missing-field and scoring contracts on a tiny frozen model."""
import copy
import csv
import json
import sys
from pathlib import Path

import pytest
import torch

from test_experiment import setup
from medical_kv.config import load_config
from medical_kv.data import development_partition, load_split, dataset_identity
from medical_kv.tasks import questions_for, grouped_loss, validate_multiscene_questions
from medical_kv.evaluation import evaluate_records, summarize, paired_compare
from medical_kv.io import read_checkpoint, checkpoint_compatibility, inference_identity

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from multiscene_common import make_qa, annotation, split_name


def records():
    result = []
    for i, (scene, form, fields) in enumerate([
        ('medical', 'structured', {'name': 'Ax Bo', 'birth_date': '2001-01-02', 'diagnosis': 'Cold'}),
        ('finance', 'dialogue', {'name': 'Cy', 'sender_account': '000123', 'amount_paid': '11 USD'}),
        ('hr', 'narrative', {'name': 'De Fo', 'birth_date': '1979', 'salary': '90000', 'department': 'Sales'}),
    ]):
        text = '\n'.join(f'{f}: {v}' for f, v in fields.items())
        ref = f'fixture:{i}'
        row = {'schema_version': 'multiscene_privacy_v1', 'record_id': f'new_{i}',
               'split': 'train', 'scene': scene, 'record_form': form, 'source': 'fixture',
               'group_id': ref, 'private_prefix': text, 'fields': fields,
               'field_annotations': {f: annotation(text, v, ref) for f, v in fields.items()},
               'name_annotation': split_name(fields['name'])}
        row['source_name_fields'] = row['name_annotation']['components']
        make_qa(row)
        result.append(row)
    return result


def configure(cfg):
    cfg.update(question_protocol='multiscene_privacy_v1', name_mode='parts', birth_date_mode='parts',
               question_batch_size=2, checkpoint_questions=True, diagnosis_contrastive={'enabled': False})


def perfect(context, questions):
    return [(q['answer'], {'generated_tokens': len(q['answer']), 'stop_reason': 'eos', 'truncated': False})
            for q in questions]


def test_frozen_views_keep_partial_date_and_forbid_full_name_training(setup):
    _, cfg, _ = setup
    configure(cfg)
    original = records()
    before = copy.deepcopy(original)
    for row in original:
        assert questions_for(row, cfg) is row['training_qa']
        assert questions_for(row, cfg, evaluation=True) is row['evaluation_qa']
        assert 'name_full_direct' not in {q['field'] for q in questions_for(row, cfg)}
    assert 'birth_date' in {q['field'] for q in questions_for(original[-1], cfg)}
    assert original == before
    bad = copy.deepcopy(original[0])
    bad['training_qa'].append(bad['evaluation_qa'][-1])
    with pytest.raises(ValueError, match='Training questions'):
        validate_multiscene_questions(bad)


def test_dynamic_field_weighting_and_missing_groups(setup):
    _, cfg, _ = setup
    configure(cfg)
    qs = questions_for(records()[0], cfg)
    values = torch.arange(1., len(qs) + 1, requires_grad=True)
    loss = grouped_loss(values, qs)
    # Name = (1+2+3)/3, DOB = (4+5+6)/3, diagnosis = 7.
    assert loss.item() == pytest.approx((2 + 5 + 7) / 3)
    loss.backward()
    torch.testing.assert_close(values.grad, torch.tensor([1/9] * 6 + [1/3]))
    with pytest.raises(ValueError, match='complete unique'):
        grouped_loss(values[:-2], qs[:-2])


def test_chunk_checkpoint_loss_and_gradient_equal_independent_queries(setup):
    backend, cfg, _ = setup
    configure(cfg)
    row = records()[1]  # single name: both MIDDLE and LAST are EOS-only.
    prompt = backend.embedding(backend.tensor(backend.initial_ids()))[0].detach().clone().requires_grad_(True)
    context = backend.context(row, vectors=prompt)
    qs = questions_for(row, cfg)
    reference = torch.stack([backend.answer_loss(context, q) for q in qs])
    gradient = torch.autograd.grad(grouped_loss(reference, qs), prompt, retain_graph=True)[0]
    actual = backend.answer_losses(context, qs)
    torch.testing.assert_close(actual, reference, rtol=1e-5, atol=2e-6)
    actual_gradient = torch.autograd.grad(grouped_loss(actual, qs), prompt)[0]
    torch.testing.assert_close(actual_gradient, gradient, rtol=3e-4, atol=3e-6)
    assert actual_gradient.norm() > 0 and torch.isfinite(actual_gradient).all()
    assert all(p.grad is None for p in backend.model.parameters())
    assert backend.generate_many(context, qs) == [backend.generate(context, q) for q in qs]


def test_variable_field_global_batch_gradient_with_empty_rank(setup):
    from medical_kv.training import local_batch_backward
    backend, cfg, _ = setup
    configure(cfg)
    rows = records()
    initial = backend.embedding(backend.tensor(backend.initial_ids()))[0].detach().clone()
    prompt = initial.clone().requires_grad_(True)
    (sum(backend.record_loss(r, vectors=prompt) for r in rows) / 3).backward()
    actual = torch.zeros_like(initial)
    for rank in range(4):
        p = initial.clone().requires_grad_(True)
        local_batch_backward(backend, p, rows, [0, 1, 2], rank, 4)
        actual += p.grad
    torch.testing.assert_close(actual, prompt.grad, rtol=3e-4, atol=3e-6)


@pytest.mark.parametrize('full', [False, True])
def test_constant_private_prefill_preserves_cache_loss_and_u_gradient(setup, full):
    backend, cfg, _ = setup
    configure(cfg)
    row = records()[0]
    initial = backend.embedding(backend.tensor(backend.initial_ids()))[0].detach().clone()
    prompt = initial.clone().requires_grad_(True)
    dense = backend.context(row, vectors=prompt, full=full)
    qs = questions_for(row, cfg)
    reference_loss = grouped_loss(backend.answer_losses(dense, qs), qs)
    expected = torch.autograd.grad(reference_loss, prompt)[0]
    cfg['split_private_prefill'] = True
    p = initial.clone().requires_grad_(True)
    split = backend.context(row, vectors=p, full=full)
    actual_loss = grouped_loss(backend.answer_losses(split, qs), qs)
    actual = torch.autograd.grad(actual_loss, p)[0]
    torch.testing.assert_close(actual_loss, reference_loss, rtol=1e-5, atol=2e-6)
    torch.testing.assert_close(actual, expected, rtol=3e-4, atol=3e-6)
    for (ka, va), (kb, vb) in zip(dense.layers, split.layers):
        torch.testing.assert_close(ka, kb, rtol=1e-5, atol=2e-6)
        torch.testing.assert_close(va, vb, rtol=1e-5, atol=2e-6)
    assert split.physical_length == dense.physical_length
    assert split.next_position == dense.next_position
    if not full:
        assert split.physical_length == len(backend.public_ids)
    blocked = backend.context(row, vectors=p, blocked=True)
    assert not backend.answer_losses(blocked, qs).requires_grad


def test_full_name_is_independent_and_absent_fields_not_counted(setup, tmp_path, monkeypatch):
    backend, cfg, _ = setup
    configure(cfg)
    def generate(ctx, qs):
        result = perfect(ctx, qs)
        for i, q in enumerate(qs):
            if q['field'] in ('name_full_direct', 'salary'):
                result[i] = ('Wrong', result[i][1])
        return result
    monkeypatch.setattr(backend, 'generate_many', generate)
    summary, rows = evaluate_records(backend, records(), prompt_ids=backend.initial_ids(), output=tmp_path/'eval',
                                     seen_values={'diagnosis': {'Cold'}})
    assert summary['field_em']['name'] == 1
    assert summary['name_full_direct']['em'] == 0
    assert summary['field_records']['salary'] == summary['field_records']['diagnosis'] == 1
    assert summary['macro_em'] == pytest.approx((1 + 1 + .75) / 3)
    assert summary['field_macro_em'] == pytest.approx(6 / 7)
    assert summary['by_scene']['hr']['macro_em'] == .75
    assert summary['by_scene_record_form']['finance/dialogue']['records'] == 1
    assert summary['seen_unseen_by_field']['diagnosis']['train_seen'] == {'records': 1, 'em': 1.}
    assert summary['qa_pairs'] == sum(len(r['evaluation_qa']) for r in records())
    with (tmp_path/'eval/predictions.csv').open(encoding='utf-8-sig', newline='') as stream:
        csv_rows = list(csv.DictReader(stream))
    assert len(csv_rows) == len(rows)
    assert set(r['scene'] for r in csv_rows) == {'medical', 'finance', 'hr'}
    assert next(r for r in csv_rows if r['field'] == 'name_full_direct')['prediction_kind'] == 'generated_full_name'
    with pytest.raises(ValueError, match='Incomplete multiscene'):
        summarize([r for r in rows if not (r['record_id'] == 'new_2' and r['field'] == 'salary')])
    comparison = paired_compare(tmp_path/'eval/predictions.jsonl', tmp_path/'eval/predictions.jsonl', bootstrap_samples=20)
    assert comparison['macro_em_difference']['difference'] == 0
    assert 'sender_account' in comparison['field_em_difference']


def test_development_uses_disjoint_groups():
    rows = [{'record_id': str(i), 'group_id': str(i // 2)} for i in range(131)]
    left, right = development_partition(rows)
    assert {r['group_id'] for r in left}.isdisjoint(r['group_id'] for r in right)


def test_new_config_rejects_legacy_checkpoint_and_disabled_paths(setup):
    backend, cfg, _ = setup
    checkpoint = {'inference_signature': inference_identity(backend), 'run_info': {'config': copy.deepcopy(cfg)}}
    configure(cfg)
    with pytest.raises(ValueError, match='mismatch'):
        checkpoint_compatibility(checkpoint, backend)
    c = load_config('configs/experiment_v5_multiscene_no_negatives.json')
    assert c['max_new_tokens'] == 128 and c['epochs'] == 100 and not c['early_stopping']
    assert c['diagnosis_contrastive'] == {'enabled': False} and not c['hard']['enabled']
    with pytest.raises(ValueError, match='must be parts'):
        load_config(c, {'name_mode': 'full'})


def test_all_new_records_load_with_frozen_labels_and_selectors():
    c = load_config('configs/experiment_v5_multiscene_no_negatives.json')
    identity = dataset_identity(c)
    assert identity['schema'] == 'multiscene_privacy_v1'
    assert identity['manual_review_complete'] is False
    for split, expected in [('train', 9565), ('validation', 1913), ('test', 9565)]:
        rows = load_split(c, split)
        assert len(rows) == expected
        assert len({(r['scene'], r['record_form']) for r in rows}) == 10
    finance = load_split(dict(c, scene='finance', record_form='dialogue'), 'train')
    assert len(finance) == 565
    with pytest.raises(ValueError, match='No records'):
        load_split(dict(c, scene='hr', record_form='dialogue'), 'train')


def test_new_training_retains_checkpoints_csv_and_skips_negatives(setup, tmp_path, monkeypatch):
    from medical_kv import training, diagnosis_contrastive
    backend, cfg, _ = setup
    configure(cfg)
    cfg.update(epochs=2, comparison_epochs=1, early_stopping=False, batch_records=3)
    monkeypatch.setattr(backend, 'generate_many', perfect)
    def forbidden(*a, **kw):
        raise AssertionError('Disabled negatives must not be prepared or scored')
    monkeypatch.setattr(diagnosis_contrastive, 'build_bank', forbidden)
    monkeypatch.setattr(backend, 'record_contrastive_loss', forbidden)
    result = training.train_soft(backend, cfg, records(), records(), {'fixture': True}, {'development_only': True})
    out = tmp_path/'run'
    assert result['status'] == 'complete'
    for name in ('best.pt', 'final.pt', 'best_first1.pt', 'last_first1.pt', 'latest.pt', 'loss_curve.png'):
        assert (out/name).exists()
    assert result['state']['field_exposures']['salary'] == 2
    assert result['state']['field_exposures']['name'] == 6
    assert len(list((out/'validation_predictions').glob('*/predictions.csv'))) == 2
    assert read_checkpoint(out/'final.pt')['run_info']['diagnosis_negatives'] is None
    assert not (out/'diagnosis_negatives.json').exists()

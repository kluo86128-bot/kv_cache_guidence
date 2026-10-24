import copy
import csv
import json
import pytest
import torch
from test_experiment import setup
from medical_kv.config import load_config, FIELDS
from medical_kv.evaluation import evaluate_records, summarize, paired_compare
from medical_kv.io import read_checkpoint, file_hash
from medical_kv.tasks import BIRTH_PARTS, questions_for, grouped_loss
from medical_kv.training import train_soft


def test_derived_targets_preserve_frozen_data_and_prefix():
    from medical_kv.data import dataset_identity, load_split
    cfg = load_config('configs/experiment_v2_birth_parts.json')
    before = dataset_identity(cfg)
    records = load_split(cfg, 'train')
    original = copy.deepcopy(records)
    for record in records:
        parts = {qa['field']: qa['answer'] for qa in questions_for(record, cfg)}
        assert len(parts) == 6
        assert '-'.join(parts[field] for field in BIRTH_PARTS) == record['fields']['birth_date']
        assert len(parts['birth_year']) == 4 and len(parts['birth_month']) == len(parts['birth_day']) == 2
    assert records == original and dataset_identity(cfg) == before
    assert cfg['epochs'] == 100 and cfg['early_stopping'] is False and cfg['comparison_epochs'] == 20


def test_grouped_loss_gives_birthday_one_quarter_of_weight(setup):
    _, cfg, record = setup
    cfg['birth_date_mode'] = 'parts'
    questions = questions_for(record, cfg)
    losses = torch.tensor([1., 3., 6., 9., 2., 4.], requires_grad=True)
    loss = grouped_loss(losses, questions)
    assert float(loss.detach()) == pytest.approx((1 + (3 + 6 + 9) / 3 + 2 + 4) / 4)
    loss.backward()
    torch.testing.assert_close(losses.grad, torch.tensor([1/4, 1/12, 1/12, 1/12, 1/4, 1/4]))


def test_six_question_batched_loss_gradient_matches_independent_queries(setup):
    backend, cfg, record = setup
    cfg['birth_date_mode'] = 'parts'
    questions = questions_for(record, cfg)
    prompt = backend.embedding(backend.tensor(backend.initial_ids()))[0].detach().clone().requires_grad_(True)
    context = backend.context(record, vectors=prompt)
    original = [(k.detach().clone(), v.detach().clone()) for k, v in context.layers]
    sequential = torch.stack([backend.answer_loss(context, q) for q in questions])
    batched = backend.answer_losses(context, questions)
    torch.testing.assert_close(batched, sequential, rtol=1e-5, atol=2e-6)
    expected = torch.autograd.grad(grouped_loss(sequential, questions), prompt, retain_graph=True)[0]
    actual_loss = backend.record_loss(record, vectors=prompt)
    actual = torch.autograd.grad(actual_loss, prompt)[0]
    torch.testing.assert_close(actual_loss, grouped_loss(batched, questions), rtol=1e-5, atol=2e-6)
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-6)
    assert backend.generate_many(context, questions) == [backend.generate(context, q) for q in questions]
    for (k, v), (original_k, original_v) in zip(context.layers, original):
        torch.testing.assert_close(k, original_k)
        torch.testing.assert_close(v, original_v)


def test_split_evaluation_keeps_raw_components_and_scores_full_date(setup, tmp_path, monkeypatch):
    backend, cfg, record = setup
    cfg['birth_date_mode'] = 'parts'
    calls = []
    def generated(context, questions):
        calls.append([q['field'] for q in questions])
        # A missing leading zero must remain wrong, even if the date can be parsed.
        return [('1' if q['field'] == 'birth_month' else q['answer'],
                 {'generated_tokens': len(q['answer']), 'stop_reason': 'eos', 'truncated': False}) for q in questions]
    monkeypatch.setattr(backend, 'generate_many', generated)
    summary, rows = evaluate_records(backend, [record], prompt_ids=backend.initial_ids(),
                                    output=tmp_path / 'evaluation', seen_birthdays={'2001-01-01'})
    assert len(calls) == 1 and len(calls[0]) == 6
    assert summary['qa_pairs'] == 6 and summary['prediction_rows'] == len(rows) == 7
    assert summary['field_em'] == dict(name=1., birth_date=0., address=1., diagnosis=1.)
    assert summary['macro_em'] == .75 and summary['all_fields_em'] == 0
    assert summary['birth_date_component_em'] == dict(birth_year=1., birth_month=0., birth_day=1.)
    assert summary['birth_date_subgroups']['train_seen'] == {'records': 1, 'em': 0.}
    reconstructed = rows[-1]
    assert reconstructed['prediction'] == '2001-1-01' and reconstructed['prediction_kind'] == 'reconstructed'
    assert summary['field_nll']['birth_date'] == pytest.approx(sum(r['nll'] for r in rows if r['field'] in BIRTH_PARTS) / 3)
    with (tmp_path / 'evaluation/predictions.csv').open(encoding='utf-8-sig', newline='') as stream:
        csv_rows = list(csv.DictReader(stream))
    assert len(csv_rows) == 7 and csv_rows[-1]['prediction_kind'] == 'reconstructed'
    broken = copy.deepcopy(rows)
    broken[-1]['em'] = True
    with pytest.raises(ValueError, match='reconstructed'):
        summarize(broken)
    old_rows = [dict(r) for r in rows if r['field'] in FIELDS]
    for row in old_rows:
        row['birth_date_mode'] = 'full'
    old_rows[-1].update(prediction=record['fields']['birth_date'], em=True, birth_date_mode='full', prediction_kind='generated')
    # Put the original four-query protocol in canonical field order.
    old_rows = sorted(old_rows, key=lambda row: FIELDS.index(row['field']))
    old_path = tmp_path / 'original.jsonl'
    old_path.write_text(''.join(json.dumps(r) + '\n' for r in old_rows), encoding='utf-8')
    comparison = paired_compare(tmp_path / 'evaluation/predictions.jsonl', old_path, bootstrap_samples=10)
    assert comparison['macro_em_difference']['difference'] == -.25
    assert comparison['birth_date_protocol_a'] == 'parts'


def test_first20_snapshots_survive_later_training_and_resume(setup, tmp_path, monkeypatch):
    import medical_kv.training as training
    backend, cfg, record = setup
    cfg.update(birth_date_mode='parts', epochs=22, early_stopping=False, patience=1,
               comparison_epochs=20, output_directory=str(tmp_path / 'long_run'))
    # Deliberately pick an earlier window best, a different epoch-20 state,
    # and a better global best after epoch 20.
    def evaluate(*args, **kwargs):
        out = kwargs['output']
        out.mkdir(parents=True)
        epoch = kwargs['metadata']['epoch']
        value = .9 if epoch == 21 else (.8 if epoch == 3 else .1)
        metrics = {'macro_em': value, 'macro_nll': float(epoch)}
        (out / 'predictions.csv').write_text(f'epoch,{epoch}\n', encoding='utf-8')
        (out / 'predictions.jsonl').write_text('{}\n', encoding='utf-8')
        return metrics, []
    class NoPlot:
        def __init__(self, *args, **kwargs): pass
        def update_training(self, *args): pass
        def update_validation(self, *args): pass
        def save(self): pass
    monkeypatch.setattr(training, 'evaluate_records', evaluate)
    monkeypatch.setattr(training, 'LossCurve', NoPlot)
    from contextlib import contextmanager
    # Interrupt at epoch 20 and resume with exactly the same configuration.
    original_context = training.stop_at_boundary
    @contextmanager
    def controlled_stop():
        flag = {'requested': False}
        original_evaluate = training.evaluate_records
        def stop_on_validation(*args, **kwargs):
            result = original_evaluate(*args, **kwargs)
            if kwargs['metadata']['epoch'] == 20:
                flag['requested'] = True
            return result
        monkeypatch.setattr(training, 'evaluate_records', stop_on_validation)
        try: yield flag
        finally: monkeypatch.setattr(training, 'evaluate_records', original_evaluate)
    monkeypatch.setattr(training, 'stop_at_boundary', controlled_stop)
    first = train_soft(backend, cfg, [record], [record], {'toy': True}, {'development_only': True})
    out = tmp_path / 'long_run'
    assert first['status'] == 'partial' and first['state']['epoch'] == 20
    best20, last20 = out / 'best_first20.pt', out / 'last_first20.pt'
    hashes = [file_hash(best20), file_hash(last20)]
    assert read_checkpoint(best20)['state']['epoch'] == 3
    assert read_checkpoint(last20)['state']['epoch'] == 20
    assert not (out / 'final.pt').exists()
    monkeypatch.setattr(training, 'stop_at_boundary', original_context)
    result = train_soft(backend, cfg, [record], [record], {'toy': True}, {'development_only': True}, resume=out / 'latest.pt')
    assert result['status'] == 'complete' and result['stop_reason'] == 'epochs_completed'
    assert result['state']['epoch'] == 22 and result['state']['step'] == 22
    assert [file_hash(best20), file_hash(last20)] == hashes
    assert read_checkpoint(out / 'best.pt')['state']['epoch'] == 21
    assert read_checkpoint(out / 'final.pt')['state']['epoch'] == 22
    torch.testing.assert_close(read_checkpoint(out / 'latest.pt')['prompt'], read_checkpoint(out / 'final.pt')['prompt'])
    manifest = json.loads((out / 'checkpoint_results.json').read_text(encoding='utf-8'))
    assert manifest['comparison_window_complete'] is True
    assert manifest['checkpoints']['best_first20']['epoch'] == 3
    assert manifest['checkpoints']['last_first20']['epoch'] == 20
    assert manifest['checkpoints']['final']['epoch'] == 22
    assert manifest['checkpoints']['best']['epoch'] == 21
    assert all('predictions_csv' in entry for entry in manifest['checkpoints'].values())

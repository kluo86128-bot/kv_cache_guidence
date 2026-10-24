"""Read-only audit of saved predictions and frozen train/validation sources."""
import argparse
import collections
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import difflib

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ('name', 'birth_date', 'address', 'diagnosis')


def letters(value):
    return ' '.join(re.sub(r'\d+', '', value).split())


def numbers(value):
    return re.findall(r'\d+', value)


def first_word(value):
    return value.split()[0] if value.split() else ''


def edit_distance(left, right):
    row = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        nxt = [i]
        for j, b in enumerate(right, 1):
            nxt.append(min(nxt[-1] + 1, row[j] + 1, row[j - 1] + (a != b)))
        row = nxt
    return row[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='predictions.csv')
    parser.add_argument('--output-directory', default='analysis/validation_seed17_step2250')
    args = parser.parse_args()
    source = ROOT / args.input
    output = ROOT / args.output_directory
    output.mkdir(parents=True, exist_ok=True)
    with source.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    train = [json.loads(line) for line in (ROOT / 'data/dataset_v1/train.jsonl').read_text(encoding='utf-8').splitlines()]
    validation = [json.loads(line) for line in (ROOT / 'data/dataset_v1/validation.jsonl').read_text(encoding='utf-8').splitlines()]
    records = {record['record_id']: record for record in validation}
    grouped = collections.defaultdict(dict)
    for row in rows:
        rid, field = row['record_id'], row['field']
        assert rid in records and field in FIELDS and field not in grouped[rid]
        qa = next(qa for qa in records[rid]['qa'] if qa['field'] == field)
        assert row['gold'] == qa['answer'] and row['question'] == qa['question']
        row['computed_em'] = row['prediction'].strip() == row['gold'].strip()
        assert row['em'].lower() == str(row['computed_em']).lower()
        assert math.isfinite(float(row['nll']))
        grouped[rid][field] = row
    assert set(grouped) == set(records) and all(set(v) == set(FIELDS) for v in grouped.values())
    metrics = {}
    details = []
    for field in FIELDS:
        selected = [row for row in rows if row['field'] == field]
        item = {'records': len(selected), 'exact': sum(row['computed_em'] for row in selected),
                'truncated': sum(row['truncated'].lower() == 'true' for row in selected),
                'mean_nll': statistics.mean(float(row['nll']) for row in selected),
                'unique_gold': len({row['gold'] for row in selected}),
                'unique_prediction': len({row['prediction'] for row in selected})}
        first_errors = collections.Counter()
        for row in selected:
            gold, pred = row['gold'].strip(), row['prediction'].strip()
            if gold == pred:
                continue
            index = next((i for i, (a, b) in enumerate(zip(gold, pred)) if a != b), min(len(gold), len(pred)))
            char = gold[index] if index < len(gold) else ''
            category = 'digit' if char.isdigit() else 'letter' if char.isalpha() else 'whitespace' if char.isspace() else 'gold_already_ended' if not char else 'punctuation'
            first_errors[category] += 1
        item['first_wrong_character_gold_type_among_errors'] = dict(first_errors)
        item['exploratory_char_similarity_definitions_not_confirmed_original_loose_score'] = {
            'SequenceMatcher_ratio_mean_percent': statistics.mean(difflib.SequenceMatcher(None, r['gold'], r['prediction']).ratio() for r in selected) * 100,
            'one_minus_levenshtein_over_max_length_mean_percent': statistics.mean(1 - edit_distance(r['gold'], r['prediction']) / max(len(r['gold']), len(r['prediction'])) for r in selected) * 100}
        if field != 'diagnosis':
            item.update({'digitless_exact': 0, 'all_numeric_groups_exact': 0, 'first_numeric_group_exact': 0,
                         'only_numeric_error': 0, 'only_text_error': 0, 'numeric_and_text_error': 0})
            for row in selected:
                gold, pred = row['gold'].strip(), row['prediction'].strip()
                text_ok, numeric_ok = letters(gold) == letters(pred), numbers(gold) == numbers(pred)
                item['digitless_exact'] += text_ok
                item['all_numeric_groups_exact'] += numeric_ok
                item['first_numeric_group_exact'] += bool(numbers(gold) and numbers(pred) and numbers(gold)[0] == numbers(pred)[0])
                category = 'correct' if row['computed_em'] else ('only_numeric_error' if text_ok and not numeric_ok
                    else 'only_text_error' if numeric_ok and not text_ok else 'numeric_and_text_error')
                if category != 'correct':
                    item[category] += 1
                details.append({'record_id': row['record_id'], 'field': field, 'gold': gold, 'prediction': pred,
                    'em': row['computed_em'], 'digitless_exact': text_ok, 'numeric_groups_exact': numeric_ok,
                    'error_category': category, 'gold_number_groups': json.dumps(numbers(gold)),
                    'predicted_number_groups': json.dumps(numbers(pred)), 'truncated': row['truncated']})
        metrics[field] = item
    name_rows = [row for row in rows if row['field'] == 'name']
    name = metrics['name']
    name['first_word_letters_exact'] = sum(letters(first_word(r['gold'])) == letters(first_word(r['prediction'])) for r in name_rows)
    name['first_word_exact_including_digits'] = sum(first_word(r['gold']) == first_word(r['prediction']) for r in name_rows)
    name['extra_whitespace_words'] = sum(len(r['prediction'].split()) > len(r['gold'].split()) for r in name_rows)
    name['fewer_whitespace_words'] = sum(len(r['prediction'].split()) < len(r['gold'].split()) for r in name_rows)
    name['word_positions_by_gold_word_count'] = {}
    for count in sorted({len(r['gold'].split()) for r in name_rows}):
        selected = [r for r in name_rows if len(r['gold'].split()) == count]
        name['word_positions_by_gold_word_count'][str(count)] = {'records': len(selected),
            'letter_matches': [sum(len(r['prediction'].split()) > i and letters(r['gold'].split()[i]) == letters(r['prediction'].split()[i]) for r in selected) for i in range(count)],
            'whole_word_matches': [sum(len(r['prediction'].split()) > i and r['gold'].split()[i] == r['prediction'].split()[i] for r in selected) for i in range(count)]}
    name['multiword_source_component_examples'] = [
        {'record_id': r['record_id'], 'source_name_fields': records[r['record_id']]['source_name_fields'],
         'gold': r['gold'], 'prediction': r['prediction']}
        for r in name_rows if any(' ' in value.strip() for value in records[r['record_id']]['source_name_fields'].values())]
    birthdays = [r for r in rows if r['field'] == 'birth_date']
    date_metrics = metrics['birth_date']
    valid_dates, invalid = [], []
    for row in birthdays:
        try:
            valid_dates.append(datetime.date.fromisoformat(row['prediction']))
        except ValueError:
            invalid.append({'record_id': row['record_id'], 'prediction': row['prediction']})
    date_metrics['yyyy_mm_dd_format'] = sum(bool(re.fullmatch(r'\d{4}-\d{2}-\d{2}', row['prediction'])) for row in birthdays)
    date_metrics['valid_calendar_dates'] = len(valid_dates)
    date_metrics['invalid_calendar_examples'] = invalid[:8]
    parts = {'year': (0, 4), 'month': (5, 7), 'day': (8, 10)}
    date_metrics['parts'] = {}
    for part, (start, stop) in parts.items():
        target_counts = collections.Counter(r['gold'][start:stop] for r in birthdays)
        pred_counts = collections.Counter(r['prediction'][start:stop] for r in birthdays)
        expected = sum(target_counts[value] * pred_counts[value] for value in target_counts) / len(birthdays)
        date_metrics['parts'][part] = {'exact': sum(r['gold'][start:stop] == r['prediction'][start:stop] for r in birthdays),
            'random_repairing_expected_matches_fixed_marginals': expected,
            'gold_counts': dict(sorted(target_counts.items())), 'prediction_counts': dict(sorted(pred_counts.items()))}
    date_metrics['mean_component_em'] = sum(value['exact'] for value in date_metrics['parts'].values()) / (3 * len(birthdays))
    date_metrics['months_04_05_06'] = {}
    for label, values in [('train_gold', [r['fields']['birth_date'] for r in train]),
                          ('validation_gold', [r['gold'] for r in birthdays]),
                          ('predictions', [r['prediction'] for r in birthdays])]:
        counts = collections.Counter(value[5:7] for value in values)
        date_metrics['months_04_05_06'][label] = {'count': sum(counts[m] for m in ('04', '05', '06')), 'total': len(values),
            'fraction': sum(counts[m] for m in ('04', '05', '06')) / len(values), 'months': dict(sorted(counts.items()))}
    train_dates = {r['fields']['birth_date'] for r in train}
    validation_dates = {r['fields']['birth_date'] for r in validation}
    date_metrics['gold_dates_seen_in_training'] = sum(r['gold'] in train_dates for r in birthdays)
    date_metrics['predicted_dates_in_training_set'] = sum(r['prediction'] in train_dates for r in birthdays)
    date_metrics['predicted_dates_in_validation_set'] = sum(r['prediction'] in validation_dates for r in birthdays)
    date_metrics['most_common_predictions'] = collections.Counter(r['prediction'] for r in birthdays).most_common(10)
    date_metrics['prediction_internal_digit_coupling'] = {}
    for label, pairs in [('predicted_year_last_digit_equals_day_last_digit', [(r['prediction'][3], r['prediction'][9]) for r in birthdays]),
                         ('predicted_month_last_digit_equals_day_last_digit', [(r['prediction'][6], r['prediction'][9]) for r in birthdays]),
                         ('gold_year_last_digit_equals_day_last_digit', [(r['gold'][3], r['gold'][9]) for r in birthdays])]:
        left, right = collections.Counter(a for a, _ in pairs), collections.Counter(b for _, b in pairs)
        date_metrics['prediction_internal_digit_coupling'][label] = {
            'matches': sum(a == b for a, b in pairs), 'total': len(pairs),
            'independent_pairing_expected_matches': sum(left[a] * right[a] for a in left) / len(pairs)}
    date_metrics['predicted_days_below_10'] = sum(int(r['prediction'][8:10]) < 10 for r in birthdays)
    date_metrics['gold_days_below_10'] = sum(int(r['gold'][8:10]) < 10 for r in birthdays)
    date_metrics['month_exact_given_correct_year'] = {
        'correct_year_records': sum(r['gold'][:4] == r['prediction'][:4] for r in birthdays),
        'correct_year_and_month': sum(r['gold'][:7] == r['prediction'][:7] for r in birthdays)}
    diagnostics = [row for row in rows if row['field'] == 'diagnosis']
    diagnosis_counts = collections.Counter(row['gold'] for row in diagnostics)
    true_counts = collections.Counter(records[rid]['token_lengths']['private_prefix'] for rid in grouped)
    length_classes = collections.defaultdict(collections.Counter)
    for rid, record in records.items():
        length_classes[record['token_lengths']['private_prefix']][record['fields']['diagnosis']] += 1
    summary = {'verification_status': 'ANALYZED; saved-output statistics recomputed, model inference not rerun',
        'source_file': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'rows': len(rows), 'patients': len(grouped), 'epoch_values': sorted({r.get('epoch', '') for r in rows}),
        'step_values': sorted({r.get('step', '') for r in rows}), 'splits': sorted({r.get('split', '') for r in rows}),
        'conditions': sorted({r.get('condition', '') for r in rows}),
        'gold_and_questions_match_frozen_validation': True, 'csv_em_matches_recomputed': True,
        'field_metrics': metrics, 'macro_em': statistics.mean(metrics[f]['exact'] / metrics[f]['records'] for f in FIELDS),
        'macro_em_without_diagnosis': statistics.mean(metrics[f]['exact'] / metrics[f]['records'] for f in FIELDS[:-1]),
        'all_fields_exact_records': sum(all(record[f]['computed_em'] for f in FIELDS) for record in grouped.values()),
        'diagnosis_share_of_total_correct_answers': metrics['diagnosis']['exact'] / sum(m['exact'] for m in metrics.values()),
        'diagnosis_categories': len(diagnosis_counts), 'diagnosis_counts': dict(sorted(diagnosis_counts.items())),
        'constant_label_diagnosis_baseline': max(diagnosis_counts.values()) / len(diagnostics),
        'private_prefix_token_length_histogram': dict(sorted(true_counts.items())),
        'prefix_length_in_sample_majority_lookup_accuracy_not_causal': sum(max(counter.values()) for counter in length_classes.values()) / len(records)}
    (output / 'audit_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    with (output / 'per_answer_error_categories.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(details)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

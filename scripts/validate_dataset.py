"""Independently audit final JSONL against the untouched generated source CSVs."""
import collections
import argparse
import csv
import datetime
import hashlib
import itertools
import json
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from medical_kv.name_policy import NAME_FORMAT, has_name_digits
sys.path.insert(0, str(ROOT / 'tools/python'))
from tokenizers import Tokenizer


def normalize(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while block := f.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--write-report', action='store_true', help='Explicitly refresh report and its artifact hash')
    args = parser.parse_args()
    cfg = json.loads((ROOT / 'configs/dataset_v1.json').read_text(encoding='utf-8'))
    out, src = ROOT / cfg['output_directory'], ROOT / cfg['source_csv_directory']
    manifest = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))
    patients = {p['Id']: p for p in read_csv(src / 'patients.csv')}
    encounters = {e['Id']: e for e in read_csv(src / 'encounters.csv')}
    condition_rows = read_csv(src / 'conditions.csv')
    conditions = {(c['PATIENT'], c['ENCOUNTER'], c['CODE'], c['DESCRIPTION'], c['START'], c['SYSTEM']) for c in condition_rows}
    tokenizer = Tokenizer.from_file(str(ROOT / cfg['tokenizer_file']))
    expected_questions = {
        'name': "What is the patient's full name? Return only the field value.",
        'birth_date': "What is the patient's date of birth? Return only the field value.",
        'address': "What is the patient's home street address? Return only the field value.",
        'diagnosis': 'What diagnosis is recorded? Return only the field value.',
    }
    splits, checks = {}, {}
    ids, pids, names, addresses, groups = {}, {}, {}, {}, {}
    for split, size in cfg['split_sizes'].items():
        records = [json.loads(line) for line in (out / f'{split}.jsonl').read_text(encoding='utf-8').splitlines()]
        assert len(records) == size
        splits[split] = records
        for r in records:
            f, p = r['fields'], r['provenance']
            assert r['split'] == split and r['is_synthetic'] is True
            assert r['schema_version'] == 'source_preserving_v3'
            assert r['name_format'] == NAME_FORMAT and not has_name_digits(f['name'])
            assert set(f) == {'name', 'birth_date', 'address', 'diagnosis'}
            patient, encounter = patients[p['source_patient_id']], encounters[p['source_encounter_id']]
            assert encounter['PATIENT'] == patient['Id']
            assert (patient['Id'], encounter['Id'], p['source_condition_code'], f['diagnosis'],
                    p['source_condition_start'], p['source_condition_system']) in conditions
            assert f['name'] == ' '.join(patient[k] for k in ('FIRST', 'MIDDLE', 'LAST') if patient[k] != '')
            assert f['birth_date'] == patient['BIRTHDATE']
            assert f['address'] == patient['ADDRESS']
            assert all(f.values())
            assert r['source_name_fields'] == {k: patient[k] for k in ('FIRST', 'MIDDLE', 'LAST')}
            expected_prefix = '\n'.join([f'patients.{k}: {patient[k]}' for k in ('FIRST', 'MIDDLE', 'LAST')] + [
                f"patients.BIRTHDATE: {patient['BIRTHDATE']}", f"patients.ADDRESS: {patient['ADDRESS']}",
                f"conditions.DESCRIPTION: {f['diagnosis']}"])
            assert r['private_prefix'] == expected_prefix
            assert patient['Id'] not in expected_prefix and encounter['Id'] not in expected_prefix
            assert 'encounters.START:' not in expected_prefix and 'MR-' not in expected_prefix
            datetime.date.fromisoformat(f['birth_date'])
            assert f['birth_date'] <= encounter['START'][:10]
            assert len(r['qa']) == 4 and {q['field'] for q in r['qa']} == set(f)
            for q in r['qa']:
                assert q['answer'] == f[q['field']]
                assert q['question'] == expected_questions[q['field']]
            text = expected_prefix + ' '.join(q['question'] + q['answer'] for q in r['qa'])
            assert not any('\u4e00' <= c <= '\u9fff' for c in text)
            assert r['token_lengths']['private_prefix'] == len(tokenizer.encode(expected_prefix, add_special_tokens=False).ids)
            assert r['token_lengths']['answers'] == {k: len(tokenizer.encode(v, add_special_tokens=False).ids) for k, v in f.items()}
        ids[split] = {r['record_id'] for r in records}
        pids[split] = {r['provenance']['source_patient_id'] for r in records}
        names[split] = {normalize(r['fields']['name']) for r in records}
        addresses[split] = {normalize(r['fields']['address']) for r in records}
        groups[split] = {r['group_id'] for r in records}
        assert len(pids[split]) == size
        counts = collections.Counter(r['fields']['diagnosis'] for r in records)
        assert set(counts) == set(manifest['diagnosis_selection']['labels'])
        assert max(counts.values()) - min(counts.values()) <= 1
        checks[split] = {'records': size, 'qa_pairs': size * 4, 'diagnosis_categories': len(counts),
                         'min_records_per_diagnosis': min(counts.values()), 'max_records_per_diagnosis': max(counts.values())}

    intersections = []
    for a, b in itertools.combinations(cfg['split_sizes'], 2):
        result = {'splits': [a, b]}
        for key, sets in [('record_ids', ids), ('patient_ids', pids), ('normalized_names', names),
                          ('normalized_addresses', addresses), ('group_ids', groups)]:
            result[key] = len(sets[a] & sets[b])
            assert result[key] == 0
        intersections.append(result)
    preview = [json.loads(x) for x in (ROOT / 'data/preview_100/samples_100.jsonl').read_text(encoding='utf-8').splitlines()]
    for records in splits.values():
        assert not ({r['provenance']['source_patient_id'] for r in records} & {r['provenance']['source_patient_id'] for r in preview})
        assert not ({normalize(r['fields']['name']) for r in records} & {normalize(r['fields']['name']) for r in preview})
        assert not ({normalize(r['fields']['address']) for r in records} & {normalize(r['fields']['address']) for r in preview})
    all_records = [json.loads(x) for x in (out / 'all_records.jsonl').read_text(encoding='utf-8').splitlines()]
    assert all_records == [r for split in cfg['split_sizes'] for r in splits[split]]
    assert len(all_records) == 2328
    originals = json.loads((out / 'source_rows_selected.json').read_text(encoding='utf-8'))
    for i, record in enumerate(all_records):
        p = record['provenance']
        assert originals['patients'][i] == patients[p['source_patient_id']]
        assert originals['encounters'][i] == encounters[p['source_encounter_id']]
        c = originals['conditions'][i]
        assert c in condition_rows
        assert c['PATIENT'] == p['source_patient_id'] and c['ENCOUNTER'] == p['source_encounter_id']
        assert c['DESCRIPTION'] == record['fields']['diagnosis']
    for table, expected in manifest['source_csv_sha256'].items():
        assert sha(src / (table + '.csv')) == expected
    artifact_hashes = json.loads((out / 'artifact_hashes.json').read_text(encoding='utf-8'))
    for filename, expected in artifact_hashes.items():
        assert sha(out / filename) == expected
    report = {
        'status': 'PASS', 'validated_date': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).date().isoformat(), 'records': 2328, 'qa_pairs': 9312,
        'splits': checks, 'cross_split_intersections': intersections,
        'name_format': NAME_FORMAT,
        'names_with_numeric_suffix': 0,
        'source_field_equality': 'PASS (all records checked against the documented natural-name source CSV)',
        'source_join_integrity': 'PASS', 'preview_patient_name_address_exclusion': 'PASS',
        'no_translations_or_substituted_values': 'PASS',
        'no_identifiers_in_private_prefix_or_qa': 'PASS',
        'token_lengths_recomputed': 'PASS', 'source_and_artifact_hashes': 'PASS',
        'validator_sha256': sha(Path(__file__)),
        'limitations': ['Synthetic data with preserved natural generator correlations.',
                       'All diagnosis categories are selected by availability, not natural prevalence.',
                       'Token lengths cover plain strings, not a chat template or U/C.',
                       'No model training or cache-recovery evaluation was run.'],
    }
    if args.write_report:
        (out / 'validation_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        artifact_hashes['validation_report.json'] = sha(out / 'validation_report.json')
        (out / 'artifact_hashes.json').write_text(json.dumps(artifact_hashes, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

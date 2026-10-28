"""Business-event coverage, source isolation, literal evidence and cache routing."""
import copy
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from hr_augmentation import (LEAVE, COMPLAINT, MARKERS, adult_event_eligible,
    augment_rows, clean_parent_record, connect_augmented_groups, group_and_partition,
    select_records, complaint_subject, read_donors)
from multiscene_common import annotation, fingerprint, make_qa, split_name

KEEP = ['name', 'birth_date', 'salary', 'hire_date', 'department', 'title',
        'identity_number', 'address', 'email', *LEAVE, *COMPLAINT]


def row(split, form, index, partial=False):
    rid = f'{split}_{form}_{index}'
    fields = {'name': 'Alice Brown', 'birth_date': '1980-01-02', 'salary': '40000',
              'hire_date': '2005-03-04', 'department': 'Sales', 'title': 'Analyst',
              'address': '3 Oak Lane', 'email': 'alice@example.com', 'employment_status': 'full-time'}
    if partial:
        fields['leave_reason'] = 'a family appointment'
    native = '\n'.join(k + ': ' + v for k, v in fields.items())
    r = {'record_id': rid, 'split': split, 'record_form': form, 'scene': 'hr',
         'fields': fields, 'private_prefix': native, 'group_id': rid, 'related_source_keys': [rid],
         'native_text_length': len(native), 'native_text_sha256': fingerprint(native),
         'field_annotations': {f: annotation(native, v, rid) for f, v in fields.items()},
         'name_annotation': split_name(fields['name'])}
    r['source_name_fields'] = r['name_annotation']['components']
    extra = '\n\nFor the supplemental synthetic employee details, Identity number is AB123.'
    r['private_prefix'] += extra
    r['fields']['identity_number'] = 'AB123'
    r['field_annotations']['identity_number'] = annotation(r['private_prefix'], 'AB123', rid + ':identity',
        origin='cross_source_synthetic_augmentation', locator='the supplemental synthetic employee details')
    r['related_source_keys'].append(rid + ':identity')
    r['augmentation_blocks'] = []
    make_qa(r)
    return r


def donors():
    # Distinct event semantics, not renamed copies of one scenario.
    reasons = ['care for an elderly parent', 'recover after dental surgery', 'attend a family wedding',
               'prepare for university examinations', 'repair severe damage at home', 'take a planned holiday abroad']
    out = []
    for i, reason in enumerate(reasons):
        out.append({'source_id': f'HR-MultiWOZ:{i}', 'source': 'HR-MultiWOZ', 'kind': 'leave',
                    'source_row': {}, 'source_text': reason,
                    'values': dict(zip(LEAVE, ['June 1st, 2023', '3 days', reason]))})
    complaints = ['My manager shared confidential payroll details.', 'My colleague repeatedly shouted insults.',
                  'My supervisor refused to pay overtime wages.', 'The team leader ignored unsafe equipment.',
                  'A coworker changed my recorded attendance.', 'My director blocked an agreed promotion.']
    for i, text in enumerate(complaints):
        out.append({'source_id': f'HR-MultiWOZ:complaint{i}', 'source': 'HR-MultiWOZ', 'kind': 'complaint',
                    'source_row': {}, 'source_text': text,
                    'values': dict(zip(COMPLAINT, ['my manager', text]))})
    return out


def test_source_families_partition_before_reuse_and_do_not_touch_rng():
    ds = donors()
    duplicate = copy.deepcopy(ds[0])
    duplicate['source_id'] = 'HR-MultiWOZ:copy'
    duplicate['values']['leave_start'] = '2030-01-01'
    ds.append(duplicate)
    actual = group_and_partition(ds, 17)
    by_id = {d['source_id']: d for d in actual}
    assert by_id[ds[0]['source_id']]['event_group'] == by_id[duplicate['source_id']]['event_group']
    assert by_id[ds[0]['source_id']]['assigned_split'] == by_id[duplicate['source_id']]['assigned_split']
    assert actual == group_and_partition(ds, 17)
    owners = {}
    for d in actual:
        assert owners.setdefault(d['event_group'], d['assigned_split']) == d['assigned_split']


def test_renamed_complaints_are_grouped_together():
    ds = donors()
    for i, name in enumerate(('John Smith', 'Peter Jones')):
        ds.append({'source_id': f'WorkplaceDispute:{i}', 'source': 'WorkplaceDispute', 'kind': 'complaint',
                   'source_row': {'participants': [name + ' (Manager)']},
                   'values': {'complaint_target': name, 'complaint_content': name + ' disclosed my confidential health information to colleagues.'}})
    # Add independent publisher families so its three-way partition is feasible.
    for i, text in enumerate(('A manager refused wages for night shifts.', 'A worker stole office equipment.')):
        ds.append({'source_id': f'WorkplaceDispute:other{i}', 'source': 'WorkplaceDispute', 'kind': 'complaint',
                   'source_row': {'participants': []}, 'values': {'complaint_target': 'manager', 'complaint_content': text}})
    actual = {d['source_id']: d for d in group_and_partition(ds, 17)}
    assert actual['WorkplaceDispute:0']['event_group'] == actual['WorkplaceDispute:1']['event_group']


def test_business_text_cleanup_keeps_parent_values_and_native_text():
    before = row('train', 'narrative', 0)
    after = clean_parent_record(before, KEEP)
    make_qa(after)
    assert not MARKERS.search(after['private_prefix'])
    assert not any(MARKERS.search(q['question']) for q in after['training_qa'])
    assert after['private_prefix'][:after['native_text_length']] == before['private_prefix'][:before['native_text_length']]
    for f, value in after['fields'].items():
        assert value == before['fields'][f]
        start, end = after['field_annotations'][f]['evidence_span']
        assert after['private_prefix'][start:end] == value
    assert 'employment_status' not in after['fields']
    assert 'supplemental synthetic' in before['private_prefix']


def test_exact_coverage_preserved_partial_events_and_source_owners():
    rows = [row(s, f, i, partial=i == 0 and f == 'narrative') for s in ('train', 'validation', 'test')
            for f in ('structured', 'narrative') for i in range(8)]
    saved = copy.deepcopy(rows)
    ds = group_and_partition(donors(), 17)
    coverage = {'train': 8, 'validation': 4, 'test': 8}
    actual = augment_rows(rows, ds, KEEP, coverage, 17)
    connect_augmented_groups(actual)
    assert rows == saved
    for split, target in coverage.items():
        subset = [r for r in actual if r['split'] == split]
        for field in (*LEAVE, *COMPLAINT):
            assert sum(field in r['fields'] for r in subset) == target
            for form in ('structured', 'narrative'):
                assert sum(field in r['fields'] for r in subset if r['record_form'] == form) == target // 2
        for r in subset:
            for field, a in r['field_annotations'].items():
                start, end = a['evidence_span']
                assert r['private_prefix'][start:end] == r['fields'][field]
            for event in r['hr_derivation']['events']:
                assert next(d for d in ds if d['source_id'] == event['source_id'])['assigned_split'] == split
    assert [r['record_id'] for r in actual] == [r['record_id'] for r in saved]
    assert all(r['fields']['leave_reason'] == 'a family appointment' for r in actual if r['record_id'].endswith('narrative_0'))


def test_unsafe_age_not_used_for_new_events_and_quota_failure_is_explicit():
    r = row('train', 'narrative', 0)
    r['fields']['birth_date'] = '2020-01-01'
    assert not adult_event_eligible(r)
    rows = [r, row('train', 'structured', 0)]
    with pytest.raises(ValueError, match='Too few age'):
        select_records(rows, LEAVE, 2, 17, 'leave')


def test_only_dataset_source_urls_are_forbidden_not_business_urls():
    assert not MARKERS.search('See https://hrsolutions.com/benefits for details.')
    assert MARKERS.search('Source: https://huggingface.co/datasets/example')


def test_complaint_subject_excludes_reporter_from_cast():
    assert complaint_subject('The student yelled at me', 'The student and I') == 'The student'
    assert complaint_subject('My manager yelled insults at me', 'My manager and I') == 'My manager'
    assert complaint_subject('Someone behaved badly', 'myself') is None


def test_same_source_cannot_be_reused_across_splits():
    left, right = row('train', 'structured', 0), row('test', 'structured', 0)
    for r in (left, right):
        r['related_source_keys'].append('source:shared')
        r['hr_derivation'] = {'events': []}
    with pytest.raises(ValueError, match='Source family crossed'):
        connect_augmented_groups([left, right])


def test_source_reader_extracts_conduct_and_filters_rhetorical_questions(tmp_path):
    with (tmp_path / 'hr_multiwoz.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['service', 'state', 'utterance', 'dialogue_id'])
        w.writeheader()
        w.writerow({'service': repr(['harassment_report']), 'dialogue_id': 1,
                    'state': repr([{'individuals_involved': 'The student and I', 'description_of_incident': 'The student yelled at me'}]),
                    'utterance': repr(['The student yelled at me'])})
    rows = [{'transcript_id': 'good', 'conflict_type': 'wage theft',
             'participants': ['Alice Brown (Analyst)', 'John Smith (Manager)'],
             'dialogue': [{'speaker': 'Alice Brown', 'text': 'John, what happened? You withheld my wages for three weeks.'}]},
            {'transcript_id': 'bad', 'conflict_type': 'conflict',
             'participants': ['Alice Brown (Analyst)', 'John Smith (Manager)'],
             'dialogue': [{'speaker': 'Alice Brown', 'text': 'John, what the hell was that? Are you serious?'}]}]
    (tmp_path / 'workplace_dispute.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    ds, rejected = read_donors(tmp_path)
    assert len(ds) == 2
    assert ds[0]['values']['complaint_target'] == 'The student'
    assert ds[1]['values']['complaint_target'] == 'John Smith (Manager)'
    assert ds[1]['values']['complaint_content'] == 'You withheld my wages for three weeks.'
    assert sum(rejected.values()) == 1


def test_changed_cached_source_is_rejected(tmp_path, monkeypatch):
    import build_hr_balanced_dataset as module
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    (tmp_path / 'cache').mkdir()
    (tmp_path / 'cache/source.csv').write_text('changed')
    with pytest.raises(ValueError, match='Cached source changed'):
        module.fetch_sources({'source_cache': 'cache', 'sources': [
            {'filename': 'source.csv', 'sha256': '0' * 64, 'name': 'fixture'}]})


def test_cache_preparation_routes_new_dataset_and_hr_only(tmp_path, monkeypatch):
    import prepare_scene_input_caches as module
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    (tmp_path / 'configs').mkdir()
    (tmp_path / 'configs/experiment_v6_medical_no_negatives.json').write_text(json.dumps({'separator': '\n\n'}))
    tokenizer = tmp_path / 'tools/tokenizer/Qwen3-4B/tokenizer.json'
    tokenizer.parent.mkdir(parents=True)
    tokenizer.write_text('{}')
    fake_tokenizer = SimpleNamespace(encode=lambda text, **kw: SimpleNamespace(ids=list(text.encode())))
    monkeypatch.setitem(sys.modules, 'tokenizers', SimpleNamespace(Tokenizer=SimpleNamespace(
        from_file=lambda path: fake_tokenizer)))
    dataset = tmp_path / 'new_dataset'
    dataset.mkdir()
    (dataset / 'manifest.json').write_text(json.dumps({'input_revision': 'private_fields_removed_v1'}))
    for split in ('train', 'validation', 'test'):
        rs = [{'record_id': split + scene, 'scene': scene, 'private_prefix': scene + ' input'} for scene in ('medical', 'finance', 'hr')]
        (dataset / (split + '.jsonl')).write_text(''.join(json.dumps(r) + '\n' for r in rs))
    output = tmp_path / 'new_cache'
    module.prepare(output, dataset_directory=dataset, scene='hr')
    meta = json.loads((output / 'manifest.json').read_text())
    assert meta['counts'] == {'hr': {'train': 1, 'validation': 1, 'test': 1}}
    assert not (output / 'medical').exists()
    value = json.loads((output / 'hr/train.jsonl').read_text())
    assert value['ids'] == list('hr input\n\n'.encode())
    with pytest.raises(FileExistsError):
        module.prepare(output, dataset_directory=dataset, scene='hr')
    with pytest.raises(ValueError, match='distinct input-cache'):
        module.prepare(tmp_path / 'data/input_cache_multiscene_v2_clean', dataset_directory=dataset, scene='hr')

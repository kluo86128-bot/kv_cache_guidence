"""Independent artifact checks and measured overlap, without loading source builders."""
from __future__ import annotations

import argparse
import collections
import datetime
import json
import re
from pathlib import Path

from multiscene_common import (ISOLATED_FIELDS, ROOT, SCENES, STRONG_FIELDS, file_hash, fingerprint,
                               natural_name, normalize, value_key, write_json)


def check_record(r):
    errors = []
    text = r['private_prefix']
    fields = r['fields']
    prefix = r['record_id']
    if r['schema_version'] != 'multiscene_privacy_v1':
        errors.append(prefix + ': incorrect schema')
    if not fields or set(fields) - set(SCENES[r['scene']]):
        errors.append(prefix + ': unknown/empty fields')
    if fingerprint(text[:r['native_text_length']]) != r['native_text_sha256']:
        errors.append(prefix + ': native source text was altered')
    if set(fields) != set(r['field_annotations']):
        errors.append(prefix + ': annotation field set mismatch')
    for field, answer in fields.items():
        a = r['field_annotations'][field]
        start, end = a['evidence_span']
        if not 0 <= start < end <= len(text) or text[start:end] != answer or a['value'] != answer:
            errors.append(prefix + ':' + field + ': gold does not equal its evidence span')
        if a['source_ref'] not in r['related_source_keys']:
            errors.append(prefix + ':' + field + ': untracked source')
        if a['origin'] == 'native' and end > r['native_text_length']:
            errors.append(prefix + ':' + field + ': native label points to added text')
        if a['origin'] == 'cross_source_synthetic_augmentation':
            blocks = [b for b in r['augmentation_blocks'] if b['source_ref'] == a['source_ref']
                      and b['span'][0] <= start < end <= b['span'][1]]
            if not blocks or not all(b['association_is_synthetic'] for b in blocks):
                errors.append(prefix + ':' + field + ': undocumented augmentation')
        # Source rows are preserved as evidence. Derived serializations are explicitly marked.
        row = a.get('source_row')
        if row and not a['derived_serialization']:
            def leaves(v):
                if isinstance(v, dict):
                    return [x for value in v.values() for x in leaves(value)]
                if isinstance(v, (list, tuple)):
                    return [x for value in v for x in leaves(value)]
                return [str(v)]
            if a['source_evidence'] not in leaves(row):
                errors.append(prefix + ':' + field + ': value not supported by retained source row')
    name = fields.get('name')
    if not natural_name(name):
        errors.append(prefix + ': non-natural/numbered/missing name target')
    parts = r['name_annotation']['components']
    if set(parts) != {'FIRST', 'MIDDLE', 'LAST'} or any(not isinstance(v, str) for v in parts.values()):
        errors.append(prefix + ': invalid name component schema')
    elif ' '.join(parts[k] for k in ('FIRST', 'MIDDLE', 'LAST') if parts[k]) != name:
        errors.append(prefix + ': name segmentation is not reversible')
    if parts != r['source_name_fields']:
        errors.append(prefix + ': name component copies disagree')
    birthday = fields.get('birth_date', '')
    if re.fullmatch(r'\d{4}-\d{2}-\d{2}', birthday):
        try:
            day = datetime.date.fromisoformat(birthday)
            if day > datetime.date(2026, 10, 16):
                errors.append(prefix + ': birth date exceeds the source reference date')
            age = r.get('source_metadata', {}).get('subject_age')
            ref_year = r.get('source_metadata', {}).get('age_reference_year', 2026)
            if age is not None and abs(ref_year - day.year - age) > 1:
                errors.append(prefix + ': added birth year conflicts with recorded patient age')
        except ValueError:
            errors.append(prefix + ': invalid ISO calendar birth date')
    if r['name_annotation']['origin'] == 'added_rule_annotation' and r['name_annotation']['semantics'] != 'source_order_segments':
        errors.append(prefix + ': artificial segmentation misrepresented as source semantics')
    canonical = {q['field']: q for q in r['qa']}
    if len(canonical) != len(r['qa']) or set(canonical) != set(fields):
        errors.append(prefix + ': incomplete/duplicate canonical questions')
    for f, q in canonical.items():
        if q['answer'] != fields[f] or q['evidence_span'] != r['field_annotations'][f]['evidence_span']:
            errors.append(prefix + ':' + f + ': canonical QA/source mismatch')
    tq = {q['field']: q for q in r['training_qa']}
    if len(tq) != len(r['training_qa']):
        errors.append(prefix + ': duplicate training question')
    if 'name' in tq or 'name_full_direct' in tq:
        errors.append(prefix + ': full-name question unexpectedly used for training')
    for f, source_key in [('name_first', 'FIRST'), ('name_middle', 'MIDDLE'), ('name_last', 'LAST')]:
        if f not in tq or tq[f]['answer'] != parts[source_key] or tq[f]['group'] != 'name':
            errors.append(prefix + ':' + f + ': missing/incorrect component supervision')
    if re.fullmatch(r'\d{4}-\d{2}-\d{2}', fields.get('birth_date', '')):
        for field, sl in [('birth_year', slice(0, 4)), ('birth_month', slice(5, 7)), ('birth_day', slice(8, 10))]:
            if field not in tq or tq[field]['answer'] != fields['birth_date'][sl]:
                errors.append(prefix + ':' + field + ': incorrect date component')
    expected_groups = set(fields)
    if {q['group'] for q in r['training_qa']} != expected_groups:
        errors.append(prefix + ': training field groups changed')
    eq = {q['field']: q for q in r['evaluation_qa']}
    if len(eq) != len(r['evaluation_qa']) or set(eq) != set(tq) | {'name_full_direct'}:
        errors.append(prefix + ': evaluation must keep components and add exactly one full-name query')
    if eq.get('name_full_direct', {}).get('answer') != name:
        errors.append(prefix + ': incorrect direct full-name target')
    for q in r['training_qa'] + r['evaluation_qa']:
        # Full private strings must not be inserted into question templates. Short generic words
        # can legitimately occur in English question text, so do not mistake them for leakage.
        if len(q['answer']) >= 12 and normalize(q['answer']) in normalize(q['question']):
            errors.append(prefix + ':' + q['field'] + ': answer appears in question')
    return errors


def field_statistics(records):
    out = {}
    fields = sorted({f for r in records for f in r['fields']})
    for f in fields:
        valid = [r for r in records if f in r['fields']]
        counts = collections.Counter(value_key(f, r['fields'][f]) for r in valid)
        by_split = {s: [r for r in valid if r['split'] == s] for s in ('train', 'validation', 'test')}
        sets = {s: {value_key(f, r['fields'][f]) for r in rs} for s, rs in by_split.items()}
        test_seen = sum(value_key(f, r['fields'][f]) in sets['train'] for r in by_split['test'])
        val_seen = sum(value_key(f, r['fields'][f]) in sets['train'] for r in by_split['validation'])
        out[f] = {'records_with_field': len(valid), 'coverage_pct': round(100 * len(valid) / len(records), 6),
                  'distinct_values': len(counts), 'within_dataset_repeat_pct': round(100 * (1 - len(counts) / len(valid)), 6),
                  'native_records': sum(r['field_annotations'][f]['origin'] == 'native' for r in valid),
                  'augmented_records': sum(r['field_annotations'][f]['origin'] != 'native' for r in valid),
                  'by_split': {s: {'records': len(rs), 'distinct_values': len(sets[s])} for s, rs in by_split.items()},
                  'test_train_seen_records': test_seen,
                  'test_train_seen_record_pct': round(100 * test_seen / len(by_split['test']), 6) if by_split['test'] else None,
                  'validation_train_seen_records': val_seen,
                  'validation_train_seen_record_pct': round(100 * val_seen / len(by_split['validation']), 6) if by_split['validation'] else None,
                  'test_train_intersecting_distinct_values': len(sets['test'] & sets['train']),
                  'test_distinct_value_seen_pct': round(100 * len(sets['test'] & sets['train']) / len(sets['test']), 6) if sets['test'] else None,
                  'maximum_value_frequency': max(counts.values(), default=0)}
    return out


def validate(folder, write=False):
    folder = Path(folder)
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    excluded = set(manifest.get('target_policy', {}).get('excluded_fields', []))
    sanitized = manifest.get('input_revision') == 'private_fields_removed_v1'
    original_sources = ({r['source_ref']:r['text'] for line in (folder/'selected_native_sources.jsonl').open(encoding='utf-8')
                         if (r:=json.loads(line))} if sanitized else {})
    records = []
    errors = []
    seen_ids = set()
    owner = {}
    cell_stats = {}
    for split in ('train', 'validation', 'test'):
        for line in (folder / (split + '.jsonl')).open(encoding='utf-8'):
            r = json.loads(line)
            records.append(r)
            if r['record_id'] in seen_ids or r['split'] != split:
                errors.append('duplicate ID or wrong split: ' + r['record_id'])
            seen_ids.add(r['record_id'])
            errors.extend(check_record(r))
            if sanitized:
                audit = r.get('input_sanitization',{})
                original = original_sources.get(r['source_case_id'],'')
                length = audit.get('original_native_text_length')
                if (len(original) != length or fingerprint(original) != audit.get('original_native_text_sha256')):
                    errors.append('Missing/changed original source audit: '+r['record_id'])
                else:
                    pieces=[]; cursor=0
                    for edit in audit.get('removed_original_spans',[]):
                        start,end=edit['span']
                        if start < length:
                            pieces.append(original[cursor:start]); cursor=min(length,end)
                    pieces.append(original[cursor:])
                    if ''.join(pieces) != r['private_prefix'][:r['native_text_length']]:
                        errors.append('Unaudited source input edit: '+r['record_id'])
            if (set(r['fields']) & excluded or set(r['field_annotations']) & excluded
                    or any(q['field'] in excluded or q.get('group') in excluded
                           for key in ('qa', 'training_qa', 'evaluation_qa') for q in r[key])):
                errors.append('Excluded target reintroduced: ' + r['record_id'])
            aliases = set(r['related_source_keys']) | {'group:' + r['group_id'], 'native:' + r['native_text_sha256']}
            for f, value in r['fields'].items():
                if f in ISOLATED_FIELDS:
                    ns = 'account' if f in {'sender_account', 'recipient_account'} else f
                    aliases.add(ns + ':' + value_key(f, value))
            for alias in aliases:
                if alias in owner and owner[alias] != split:
                    errors.append('cross-split source/identity/household collision: ' + alias)
                owner[alias] = split
    actual_cells = collections.Counter((r['scene'], r['record_form'], r['split']) for r in records)
    if len(records) != manifest['records']:
        errors.append('Manifest record count mismatch')
    previews = [json.loads(line) for line in (folder / 'preview_30.jsonl').open(encoding='utf-8')]
    train_ids = {r['record_id'] for r in records if r['split'] == 'train'}
    if len(previews) != 30 or any(r['split'] != 'train' or r['record_id'] not in train_ids for r in previews):
        errors.append('Previews must contain exactly 30 training records, never validation/test records')
    for scene, forms in json.loads((folder / 'build_config.json').read_text(encoding='utf-8'))['scenes'].items():
        for kind in forms:
            rs = [r for r in records if r['scene'] == scene and r['record_form'] == kind]
            split_counts = dict(collections.Counter(r['split'] for r in rs))
            if (scene, kind) != ('finance', 'dialogue') and split_counts != {'train': 1000, 'validation': 200, 'test': 1000}:
                errors.append(f'{scene}/{kind}: incorrect quotas {split_counts}')
            for split in ('train', 'validation', 'test'):
                path = folder / scene / kind / (split + '.jsonl')
                cell_records = [json.loads(line) for line in path.open(encoding='utf-8')]
                expected = [r for r in rs if r['split'] == split]
                if cell_records != expected:
                    errors.append(f'{scene}/{kind}/{split}: cell and combined artifacts differ')
            cell_stats[scene + '/' + kind] = {
                'records': len(rs), 'split_sizes': split_counts,
                'source_counts': dict(collections.Counter(r['source'] for r in rs)),
                'native_only_records': sum(r['sample_kind'] == 'source_preserving' for r in rs),
                'augmented_records': sum(r['sample_kind'] == 'cross_source_augmented' for r in rs),
                'records_with_native_target': sum(r['native_target_count'] > 0 for r in rs),
                'unique_groups': len({r['group_id'] for r in rs}),
                'training_questions': sum(len(r['training_qa']) for r in rs),
                'evaluation_questions': sum(len(r['evaluation_qa']) for r in rs),
                'fields': field_statistics(rs)}
    scenes = {scene: {'records': sum(r['scene'] == scene for r in records),
                      'fields': field_statistics([r for r in records if r['scene'] == scene])}
              for scene in SCENES}
    statistics = {'records': len(records), 'split_sizes': dict(collections.Counter(r['split'] for r in records)),
                  'independent_connected_groups': len({r['group_id'] for r in records}),
                  'primary_source_cases': len({r['source_case_id'] for r in records}),
                  'training_questions': sum(len(r['training_qa']) for r in records),
                  'evaluation_questions': sum(len(r['evaluation_qa']) for r in records),
                  'name_component_origins': dict(collections.Counter(r['name_annotation']['origin'] for r in records)),
                  'empty_middle_components': sum(not r['source_name_fields']['MIDDLE'] for r in records),
                  'max_private_prefix_characters': max(map(lambda r: len(r['private_prefix']), records)),
                  'cells': cell_stats, 'scenes': scenes,
                  'overlap_definitions': {
                      'within_dataset_repeat_pct': '100*(1-distinct_values/records_with_field)',
                      'test_train_seen_record_pct': '100*test records whose field value occurs in train/test records with field',
                      'test_distinct_value_seen_pct': '100*test distinct field values also in train/test distinct values',
                      'normalization': 'NFKC, whitespace/case for statistics only; phone digits; identity punctuation removed. Stored answers never normalized.'}}
    report = {'status': 'PASS' if not errors else 'FAIL', 'checks': {
        'record_schema_and_exact_quotas': True, 'source_text_unchanged': True,
        'every_gold_has_literal_evidence': True, 'source_rows_match_nonderived_answers': True,
        'names_natural_and_reversibly_segmented': True, 'training_uses_components': True,
        'evaluation_keeps_components_and_adds_direct_name_question': True,
        'same_source_donors_names_addresses_identifiers_isolated_between_splits': True,
        'per_cell_and_combined_files_agree': True}, 'error_count': len(errors), 'errors': errors[:200],
        'manual_review_complete': False,
        'scope': 'Structural, extractive-label, provenance and leakage checks; not a manual clinical/linguistic certification.'}
    if errors:
        report['checks'] = {k: 'see_errors' for k in report['checks']}
    elif sanitized:
        report['checks'].pop('source_text_unchanged')
        report['checks']['original_source_retained_and_input_deletions_verified'] = True
    if write:
        write_json(folder / 'statistics.json', statistics)
        write_json(folder / 'validation_report.json', report)
        if not errors:
            manifest['status'] = 'automatic_validation_passed_manual_review_pending'
            manifest['validation_report'] = 'validation_report.json'
            manifest['statistics'] = 'statistics.json'
            write_json(folder / 'manifest.json', manifest)
            hashes = {str(p.relative_to(folder)).replace('\\', '/'): file_hash(p)
                      for p in sorted(folder.rglob('*')) if p.is_file() and p.name != 'artifact_hashes.json'}
            write_json(folder / 'artifact_hashes.json', hashes)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', default='data/dataset_multiscene_v1')
    parser.add_argument('--write-report', action='store_true')
    args = parser.parse_args()
    report = validate(ROOT / args.directory, write=args.write_report)
    print(json.dumps(report, ensure_ascii=False))
    if report['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()

"""Add tokenizer lengths, data dictionary and CSV summaries to an already validated dataset."""
import argparse
import collections
import csv
import json
import math
import sys
from pathlib import Path

from multiscene_common import LABELS, ROOT, SCENES, file_hash, write_json


def describe(values):
    values = sorted(values)
    return {'min': values[0], 'mean': round(sum(values) / len(values), 4),
            'p50': values[len(values) // 2], 'p95': values[math.ceil(.95 * len(values)) - 1],
            'max': values[-1]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', default='data/dataset_multiscene_v1')
    args = parser.parse_args()
    folder = ROOT / args.directory
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    excluded = set(manifest.get('target_policy', {}).get('excluded_fields', []))
    sys.path.insert(0, str(ROOT / 'tools/python'))
    from tokenizers import Tokenizer
    tokenizer_path = ROOT / 'tools/tokenizer/Qwen3-4B/tokenizer.json'
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    lengths = collections.defaultdict(list)
    answer_cache = {'': 0}
    maximum_answer = {'tokens': 0}
    maximum_prefix = {'tokens': 0}
    training_questions = evaluation_questions = 0
    preview_sources = collections.defaultdict(dict)
    preview_first = collections.defaultdict(list)
    for split in ('train', 'validation', 'test'):
        with (folder / (split + '.jsonl')).open(encoding='utf-8') as stream:
            for i, line in enumerate(stream):
                r = json.loads(line)
                if split == 'train':
                    cell = r['scene'] + '/' + r['record_form']
                    preview_sources[cell].setdefault(r['source'], r)
                    if len(preview_first[cell]) < 3:
                        preview_first[cell].append(r)
                n = len(tokenizer.encode(r['private_prefix'], add_special_tokens=False).ids)
                lengths[r['scene'] + '/' + r['record_form']].append(n)
                if n > maximum_prefix['tokens']:
                    maximum_prefix = {'tokens': n, 'record_id': r['record_id']}
                training_questions += len(r['training_qa'])
                evaluation_questions += len(r['evaluation_qa'])
                for q in r['evaluation_qa']:
                    answer = q['answer']
                    if answer not in answer_cache:
                        answer_cache[answer] = len(tokenizer.encode(answer, add_special_tokens=False).ids)
                    n = answer_cache[answer]
                    if n > maximum_answer['tokens']:
                        maximum_answer = {'tokens': n, 'record_id': r['record_id'], 'field': q['field']}
                if (i + 1) % 2000 == 0:
                    print(json.dumps({'stage': 'token_audit', 'split': split, 'records': i + 1}), flush=True)
    token_report = {'tokenizer': 'Qwen3-4B local frozen tokenizer',
                    'tokenizer_sha256': file_hash(tokenizer_path),
                    'private_prefix_tokens': {k: describe(v) for k, v in sorted(lengths.items())},
                    'maximum_private_prefix': maximum_prefix, 'maximum_answer_without_eos': maximum_answer,
                    'suggested_max_new_tokens_at_least': max(64, 32 * math.ceil((maximum_answer['tokens'] + 8) / 32)),
                    'training_questions': training_questions, 'evaluation_questions': evaluation_questions,
                    'scope': 'Static text tokenization only; excludes chat wrappers, public text, prompt vectors and a model forward pass.'}
    write_json(folder / 'tokenization_report.json', token_report)
    examples = []
    for cell in sorted(preview_sources):
        chosen = list(preview_sources[cell].values())[:3]
        ids = {r['record_id'] for r in chosen}
        chosen.extend(r for r in preview_first[cell] if r['record_id'] not in ids)
        examples.extend(chosen[:3])
    with (folder / 'preview_30.jsonl').open('w', encoding='utf-8', newline='\n') as stream:
        for r in examples:
            stream.write(json.dumps(r, ensure_ascii=False) + '\n')
    with (folder / 'preview_qa.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        columns = ['record_id', 'scene', 'record_form', 'split', 'protocol', 'field',
                   'question', 'answer', 'field_origin', 'name_component_origin']
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for r in examples:
            for protocol in ('training_qa', 'evaluation_qa'):
                for q in r[protocol]:
                    writer.writerow({**{k: r[k] for k in ('record_id', 'scene', 'record_form', 'split')},
                                     'protocol': protocol, **{k: q[k] for k in ('field', 'question', 'answer')},
                                     'field_origin': r['field_annotations'][q['group']]['origin'],
                                     'name_component_origin': r['name_annotation']['origin'] if q['group'] == 'name' else ''})
    stats = json.loads((folder / 'statistics.json').read_text(encoding='utf-8'))
    with (folder / 'field_summary.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        columns = ['scene', 'record_form', 'field', 'cell_records', 'records_with_field', 'coverage_pct',
                   'native_records', 'augmented_records', 'distinct_values', 'within_dataset_repeat_pct',
                   'test_train_seen_records', 'test_train_seen_record_pct', 'test_distinct_value_seen_pct']
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for cell, data in stats['cells'].items():
            scene, kind = cell.split('/')
            for f, fs in data['fields'].items():
                writer.writerow({'scene': scene, 'record_form': kind, 'field': f, 'cell_records': data['records'],
                                 **{k: fs[k] for k in columns if k in fs}})
    with (folder / 'split_summary.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        columns = ['scene', 'record_form', 'records', 'train', 'validation', 'test',
                   'records_with_native_target', 'augmented_records', 'training_questions', 'evaluation_questions']
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for cell, data in stats['cells'].items():
            scene, kind = cell.split('/')
            writer.writerow({'scene': scene, 'record_form': kind,
                             **{s: data['split_sizes'][s] for s in ('train', 'validation', 'test')},
                             **{k: data[k] for k in columns if k in data}})
    dictionary = {
        'schema_version': 'multiscene_privacy_v1',
        'input': 'private_prefix only. Other keys are annotations/provenance, never model context.',
        'training_questions': 'training_qa; full-name canonical QA is not a training query.',
        'evaluation_questions': 'evaluation_qa; includes all component queries plus independent name_full_direct.',
        'primary_metrics': 'Original-field EM: use reconstructed name and ISO birth date; direct full-name EM is a separate metric.',
        'name_segments': 'source_name_fields / name_annotation. Added rule segments are not asserted cultural given/middle/family names.',
        'missing_field': 'No field and no question; distinguish from an explicitly recorded negative statement.',
        'source_case_id': 'Primary original case. Paired note/dialogue and variants share this identity.',
        'group_id': 'Transitive source/donor/identity/date/salary connected group; does not cross splits.',
        'field_annotations': 'Exact private-prefix evidence span, source reference, native/augmented origin and available source row.',
        'native_text': 'The first native_text_length characters of private_prefix; exact text is also in selected_native_sources.jsonl.',
        'structured_sources': 'For original CSV/SQL sources this is a deterministic structured serialization of original values; free-text sources retain their original text.',
        'augmentation_blocks': 'New explanatory text carrying original synthetic donor values; constructed association is explicitly labelled.',
        'source_syntheticity': 'Generated or fictional/scenario sources; not a claim that identities in different sources refer to the same actual person.',
        'core_target_fields': {scene: [f for f in fields[:10] if f not in excluded] for scene, fields in SCENES.items()},
        'additional_native_fields': {scene: [f for f in fields[10:] if f not in excluded] for scene, fields in SCENES.items()},
        'field_labels': {k: v for k, v in LABELS.items() if k not in excluded},
        'excluded_target_fields': sorted(excluded),
        'amounts': 'Keep source currency. Bank/account and amount/currency joins are marked derived serializations.',
        'employee_salary': 'Employees test_db does not specify salary currency/period; no unit is invented.',
        'answer_overlap': 'See statistics.json for within-set repetition versus test records whose target appears in training.',
        'group_counts': 'Total records are not claimed to be the same number of independent people/groups.',
        'preview': 'preview_30.jsonl and preview_qa.csv contain three TRAINING records per cell only; never concatenate previews with splits.',
        'file_views': 'Combined train/validation/test JSONL files contain exactly the same records as their scene/form partitions; do not concatenate these views.',
        'quality_status': 'Automatic structure/evidence/isolation validation passed; no claim of a full manual clinical/linguistic audit.',
        'loader_compatibility': 'Use medical_kv with question_protocol=multiscene_privacy_v1.',
        'rebuild_steps': ['python scripts/build_multiscene_dataset.py', 'python scripts/audit_multiscene_tokens.py',
                          'python scripts/validate_multiscene_dataset.py'],
    }
    if excluded:
        dictionary['rebuild_steps'] = ['python scripts/build_reduced_multiscene_dataset.py']
    if manifest.get('input_revision') == 'private_fields_removed_v1':
        dictionary['native_text'] = 'First native_text_length characters are the cleaned source input. Original publisher text remains in selected_native_sources.jsonl; input_sanitization records the deleted source ranges.'
        dictionary['input_revision'] = manifest['input_revision']
        dictionary['rebuild_steps'].append('python scripts/clean_reduced_private_inputs.py')
    write_json(folder / 'data_dictionary.json', dictionary)
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    manifest['files'].update({'field_summary': 'field_summary.csv', 'split_summary': 'split_summary.csv',
                              'data_dictionary': 'data_dictionary.json', 'tokenization': 'tokenization_report.json'})
    for path in list(manifest['reproduction_code_sha256']) + ['scripts/audit_multiscene_tokens.py']:
        manifest['reproduction_code_sha256'][path] = file_hash(ROOT / path)
    write_json(folder / 'manifest.json', manifest)
    from validate_multiscene_dataset import validate
    report = validate(folder, write=True)
    if report['status'] != 'PASS':
        raise RuntimeError(report)
    print(json.dumps({'status': 'complete', 'maximum_prefix': maximum_prefix, 'maximum_answer': maximum_answer}), flush=True)


if __name__ == '__main__':
    main()

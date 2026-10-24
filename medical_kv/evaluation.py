import collections
import csv
import json
import math
import os
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from .config import FIELDS
from .io import write_json
from .distributed import Runtime
from .tasks import BIRTH_PARTS, NAME_PARTS, MULTISCENE_SCHEMA, questions_for, reconstruct_birthday, reconstruct_name


def write_predictions_csv(path, rows, metadata=None):
    """Preserve raw answers, including commas, quotes and multiline output."""
    path = Path(path)
    metadata = metadata or {}
    columns = ['epoch', 'step', 'evaluation_stage', 'partial_training', 'completed_epochs',
               'records_seen_in_current_epoch', 'split', 'condition', 'record_id', 'field', 'question',
               'gold', 'prediction', 'em', 'nll', 'generated_tokens', 'stop_reason',
               'truncated', 'birth_date_seen_in_train', 'prediction_kind', 'birth_date_mode', 'name_mode',
               'schema_version', 'scene', 'record_form', 'source', 'group_id', 'original_field',
               'field_origin', 'name_component_origin', 'name_component_semantics', 'target_seen_in_train']
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, metadata.get(key, '')) for key in columns})
    os.replace(temporary, path)


def summarize(rows):
    if any(r.get('schema_version') == MULTISCENE_SCHEMA for r in rows):
        return summarize_multiscene(rows)
    grouped = collections.defaultdict(dict)
    for row in rows:
        if row['field'] in grouped[row['record_id']]:
            raise ValueError('Duplicate field prediction')
        grouped[row['record_id']][row['field']] = row
    field_sets = {frozenset(v) for v in grouped.values()}
    allowed = {frozenset((*FIELDS, *name, *birth))
               for name in ((), NAME_PARTS) for birth in ((), BIRTH_PARTS)}
    if not grouped or len(field_sets) != 1 or not field_sets.issubset(allowed):
        raise ValueError('Incomplete patient predictions; failures cannot be counted as zeros')
    fields = next(iter(field_sets))
    split_birthday, split_name = set(BIRTH_PARTS).issubset(fields), set(NAME_PARTS).issubset(fields)
    derived = ({'birth_date'} if split_birthday else set()) | ({'name'} if split_name else set())
    field_em = {f: float(np.mean([r[f]['em'] for r in grouped.values()])) for f in FIELDS}
    result = {'records': len(grouped), 'qa_pairs': len(grouped) * (4 + 2*split_birthday + 2*split_name),
              'prediction_rows': len(rows), 'birth_date_mode': 'parts' if split_birthday else 'full',
              'name_mode': 'parts' if split_name else 'full', 'field_em': field_em,
              'macro_em': float(np.mean(list(field_em.values()))),
              'all_fields_em': float(np.mean([all(r[f]['em'] for f in FIELDS) for r in grouped.values()])),
              'name_diagnosis_joint_em': float(np.mean([r['name']['em'] and r['diagnosis']['em'] for r in grouped.values()])),
              'field_nll': {f: float(np.mean([r[f]['nll'] for r in grouped.values()])) for f in FIELDS},
              'truncated_answers': sum(r['truncated'] for r in rows if r['field'] not in derived)}
    if split_name:
        result['name_component_em'] = {
            f: float(np.mean([r[f]['em'] for r in grouped.values()])) for f in NAME_PARTS}
        result['name_component_nll'] = {
            f: float(np.mean([r[f]['nll'] for r in grouped.values()])) for f in NAME_PARTS}
        result['name_nll_definition'] = 'mean of FIRST/MIDDLE/LAST token-mean NLLs, each including EOS; empty MIDDLE targets EOS only'
        for patient in grouped.values():
            joined = ' '.join(value for f in NAME_PARTS if (value := patient[f]['prediction'].strip()))
            if (patient['name']['prediction'] != joined
                    or patient['name']['em'] != (joined == patient['name']['gold'])
                    or not math.isclose(patient['name']['nll'], sum(patient[f]['nll'] for f in NAME_PARTS) / 3)):
                raise ValueError('Inconsistent reconstructed name prediction')
        result['middle_name_subgroups'] = {}
        for label, empty in (('empty', True), ('nonempty', False)):
            patients = [r for r in grouped.values() if (r['name_middle']['gold'] == '') is empty]
            result['middle_name_subgroups'][label] = {
                'records': len(patients),
                'component_em': float(np.mean([r['name_middle']['em'] for r in patients])) if patients else None,
                'full_name_em': float(np.mean([r['name']['em'] for r in patients])) if patients else None}
    if split_birthday:
        result['birth_date_component_em'] = {
            f: float(np.mean([r[f]['em'] for r in grouped.values()])) for f in BIRTH_PARTS}
        result['birth_date_component_nll'] = {
            f: float(np.mean([r[f]['nll'] for r in grouped.values()])) for f in BIRTH_PARTS}
        result['birth_date_nll_definition'] = 'mean of year/month/day token-mean NLLs, each including its EOS'
        # Reject inconsistent derived rows rather than silently using the wrong macro metric.
        for patient in grouped.values():
            joined = '-'.join(patient[f]['prediction'].strip() for f in BIRTH_PARTS)
            if (patient['birth_date']['prediction'] != joined
                    or patient['birth_date']['em'] != (joined == patient['birth_date']['gold'])
                    or not math.isclose(patient['birth_date']['nll'],
                                        sum(patient[f]['nll'] for f in BIRTH_PARTS) / 3)):
                raise ValueError('Inconsistent reconstructed birthday prediction')
    result['macro_nll'] = float(np.mean(list(result['field_nll'].values())))
    birthday_groups = {}
    for label, value in (('train_seen', True), ('train_unseen', False)):
        selected = [r['birth_date'] for r in grouped.values() if r['birth_date'].get('birth_date_seen_in_train') is value]
        birthday_groups[label] = {'records': len(selected), 'em': float(np.mean([r['em'] for r in selected])) if selected else None}
    result['birth_date_subgroups'] = birthday_groups
    return result


def summarize_multiscene(rows):
    grouped = collections.defaultdict(dict)
    for row in rows:
        if row.get('schema_version') != MULTISCENE_SCHEMA or row['field'] in grouped[row['record_id']]:
            raise ValueError('Mixed schema or duplicate field prediction')
        grouped[row['record_id']][row['field']] = row
    if not grouped:
        raise ValueError('Incomplete predictions')
    for patient in grouped.values():
        first = next(iter(patient.values()))
        targets, queries = set(first['target_fields']), set(first['question_fields'])
        expected = targets | queries
        if set(patient) != expected:
            raise ValueError('Incomplete multiscene predictions; failures cannot be counted as zeros')
        for row in patient.values():
            if (set(row['target_fields']) != targets or set(row['question_fields']) != queries
                    or any(row[k] != first[k] for k in ('scene', 'record_form', 'source', 'group_id'))
                    or not math.isfinite(row['nll'])):
                raise ValueError('Inconsistent record prediction metadata/NLL')
        for parent, parts, sep in (('name', NAME_PARTS, ' '), ('birth_date', BIRTH_PARTS, '-')):
            if not queries.intersection(parts):
                continue
            if not set(parts).issubset(queries):
                raise ValueError('Incomplete component predictions')
            values = [patient[f]['prediction'].strip() for f in parts]
            joined = sep.join(v for v in values if v) if parent == 'name' else sep.join(values)
            derived = patient[parent]
            if (derived['prediction_kind'] != 'reconstructed' or derived['prediction'] != joined
                    or derived['em'] != (joined == derived['gold'])
                    or not math.isclose(derived['nll'], sum(patient[f]['nll'] for f in parts) / len(parts))):
                raise ValueError(f'Inconsistent reconstructed {parent} prediction')
        if 'name' in targets and ('name_full_direct' not in queries
                                  or patient['name_full_direct']['gold'] != patient['name']['gold']):
            raise ValueError('Missing or inconsistent independent full-name prediction')

    def mean(values):
        return float(np.mean(values)) if values else None

    def core(patients):
        canonical = [[p[f] for f in next(iter(p.values()))['target_fields']] for p in patients]
        by_field = collections.defaultdict(list)
        for patient in canonical:
            for row in patient:
                by_field[row['field']].append(row)
        field_em = {f: mean([r['em'] for r in rs]) for f, rs in sorted(by_field.items())}
        field_nll = {f: mean([r['nll'] for r in rs]) for f, rs in sorted(by_field.items())}
        result = {'records': len(patients), 'target_pairs': sum(map(len, canonical)),
                  'qa_pairs': sum(len(next(iter(p.values()))['question_fields']) for p in patients),
                  'prediction_rows': sum(map(len, patients)),
                  'field_records': {f: len(rs) for f, rs in sorted(by_field.items())},
                  'field_coverage': {f: len(rs) / len(patients) for f, rs in sorted(by_field.items())},
                  'field_em': field_em, 'field_nll': field_nll,
                  'macro_em': mean([mean([r['em'] for r in rs]) for rs in canonical]),
                  'macro_nll': mean([mean([r['nll'] for r in rs]) for rs in canonical]),
                  'field_macro_em': mean(list(field_em.values())),
                  'field_macro_nll': mean(list(field_nll.values())),
                  'all_fields_em': mean([all(r['em'] for r in rs) for rs in canonical]),
                  'truncated_answers': sum(r['truncated'] for p in patients for r in p.values()
                                           if r['prediction_kind'] != 'reconstructed')}
        for label, parts in (('name', NAME_PARTS), ('birth_date', BIRTH_PARTS)):
            result[label + '_component_em'] = {f: mean([p[f]['em'] for p in patients if f in p])
                                               for f in parts if any(f in p for p in patients)}
            result[label + '_component_nll'] = {f: mean([p[f]['nll'] for p in patients if f in p])
                                                for f in parts if any(f in p for p in patients)}
        direct = [p['name_full_direct'] for p in patients if 'name_full_direct' in p]
        result['name_full_direct'] = {'records': len(direct), 'em': mean([r['em'] for r in direct]),
                                     'nll': mean([r['nll'] for r in direct]),
                                     'truncated_answers': sum(r['truncated'] for r in direct)}
        result['seen_unseen_by_field'] = {}
        for f, rs in sorted(by_field.items()):
            result['seen_unseen_by_field'][f] = {
                label: {'records': len(selected := [r for r in rs if r.get('target_seen_in_train') is value]),
                        'em': mean([r['em'] for r in selected])}
                for label, value in (('train_seen', True), ('train_unseen', False))}
        return result

    patients = list(grouped.values())
    result = core(patients)
    result.update(schema_version=MULTISCENE_SCHEMA, name_mode='parts', birth_date_mode='parts_or_original',
                  metric_definition='macro: mean available original fields per record, then mean records; '
                  'field_macro: mean per-field scores; components and direct full-name excluded from primary macro',
                  loss_definition='mean answer-token NLL including EOS; mean components within original field; '
                  'mean available original fields per record; mean records')
    for key, columns in (('by_scene', ('scene',)), ('by_record_form', ('record_form',)),
                         ('by_scene_record_form', ('scene', 'record_form'))):
        selections = collections.defaultdict(list)
        for patient in patients:
            first = next(iter(patient.values()))
            selections['/'.join(first[k] for k in columns)].append(patient)
        result[key] = {label: core(ps) for label, ps in sorted(selections.items())}
    return result


@torch.no_grad()
def evaluate_records(backend, records, *, vectors=None, prompt_ids=None, full=False, blocked=False,
                     output=None, metadata=None, seen_birthdays=None, seen_values=None, checkpoint_source=None,
                     resume_evaluation=None):
    rows, start = [], time.monotonic()
    runtime = Runtime(backend.device)
    reused = []
    completed_ids = set()
    if resume_evaluation is not None:
        from .evaluation_resume import resume_source, validate_metadata, completed_predictions
        source_metadata, source_snapshot = resume_source(resume_evaluation)
        validate_metadata(source_metadata, metadata or {})
        if runtime.primary:
            reused, progress = completed_predictions(resume_evaluation, records, backend.cfg)
        else:
            progress = None
        progress = runtime.broadcast(progress)
        completed_ids = set(runtime.broadcast(sorted({r['record_id'] for r in reused}) if runtime.primary else None))
        metadata = dict(metadata or {}, evaluation_resume={
            'source_directory': str(resume_evaluation), **progress,
            'previous_parallelism': source_metadata.get('parallelism'),
            'current_parallelism': runtime.metadata()})
        checkpoint_source = checkpoint_source or source_snapshot
        rows = list(reused)
        if runtime.primary:
            print(json.dumps({'evaluation_resume': metadata['evaluation_resume']}), flush=True)
    pending = [record for record in records if record['record_id'] not in completed_ids]
    output = Path(output) if output is not None else None
    if output is not None:
        if runtime.primary:
            output.mkdir(parents=True, exist_ok=False)
            if checkpoint_source is not None:
                shutil.copyfile(checkpoint_source, output / 'checkpoint.pt.tmp')
                os.replace(output / 'checkpoint.pt.tmp', output / 'checkpoint.pt')
            if reused:
                temporary_reused = output / 'reused_predictions.jsonl.tmp'
                with temporary_reused.open('w', encoding='utf-8') as stream:
                    for row in reused:
                        stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                os.replace(temporary_reused, output / 'reused_predictions.jsonl')
            write_json(output / 'metadata.json', metadata or {})
        runtime.barrier()
    partial_name = f'predictions.rank{runtime.rank}.partial.jsonl' if runtime.active else 'predictions.partial.jsonl'
    try:
        for index, record in enumerate(pending):
            if index % runtime.world_size != runtime.rank:
                continue
            context = backend.context(record, vectors=vectors, prompt_ids=prompt_ids, full=full, blocked=blocked)
            questions = questions_for(record, backend.cfg, evaluation=True)
            losses = backend.answer_losses(context, questions).tolist()
            generated = backend.generate_many(context, questions)
            if len(losses) != len(questions) or len(generated) != len(questions):
                raise ValueError('Incomplete backend answers; cannot silently drop questions')
            patient_rows = []
            for qa, loss, (prediction, generation) in zip(questions, losses, generated):
                if not math.isfinite(loss):
                    raise FloatingPointError('Non-finite evaluation NLL')
                row = {'record_id': record['record_id'], 'field': qa['field'], 'question': qa['question'],
                       'gold': qa['answer'], 'prediction': prediction, 'em': prediction.strip() == qa['answer'].strip(),
                       'nll': loss, **generation, 'prediction_kind': qa.get('prediction_kind', 'generated'),
                       'birth_date_mode': backend.cfg.get('birth_date_mode', 'full'),
                       'name_mode': backend.cfg.get('name_mode', 'full')}
                if qa['field'] == 'birth_date' and seen_birthdays is not None:
                    row['birth_date_seen_in_train'] = qa['answer'] in seen_birthdays
                patient_rows.append(row)
            name = reconstruct_name(record, patient_rows)
            if name is not None:
                patient_rows.append(name)
            birthday = reconstruct_birthday(record, patient_rows, seen_birthdays)
            if birthday is not None:
                patient_rows.append(birthday)
            for row in patient_rows:
                if record.get('schema_version') == MULTISCENE_SCHEMA:
                    original = ('name' if row['field'] in (*NAME_PARTS, 'name_full_direct')
                                else 'birth_date' if row['field'] in BIRTH_PARTS else row['field'])
                    row.update({k: record[k] for k in ('schema_version', 'scene', 'record_form', 'source', 'group_id')})
                    row.update(original_field=original, target_fields=list(record['fields']),
                               question_fields=[q['field'] for q in questions],
                               field_origin=record['field_annotations'][original]['origin'],
                               name_component_origin=record.get('name_annotation', {}).get('origin'),
                               name_component_semantics=record.get('name_annotation', {}).get('semantics'))
                    if seen_values is not None:
                        row['target_seen_in_train'] = record['fields'][original] in seen_values.get(original, set())
                row.setdefault('name_mode', backend.cfg.get('name_mode', 'full'))
                row.setdefault('birth_date_mode', backend.cfg.get('birth_date_mode', 'full'))
                rows.append(row)
                if output is not None:
                    with (output / partial_name).open('a', encoding='utf-8') as f:
                        f.write(json.dumps(row, ensure_ascii=False) + '\n')
            if runtime.primary and (index // runtime.world_size + 1) % 10 == 0:
                print(json.dumps({'evaluation_records': len(completed_ids) + index + 1, 'total': len(records)}), flush=True)
    except Exception as exc:
        if output is not None:
            write_json(output / f'failure.rank{runtime.rank}.json', {'status': 'failed', 'completed_answers': len(rows),
                       'error_type': type(exc).__name__, 'error': str(exc)})
            if runtime.primary:
                write_json(output / 'failure.json', {'status': 'failed', 'error': str(exc), 'rank': runtime.rank})
        raise
    rows = [row for shard in runtime.objects(rows) for row in shard]
    order = {}
    for i, record in enumerate(records):
        fields = [q['field'] for q in questions_for(record, backend.cfg, evaluation=True)]
        for parent, parts in (('name', NAME_PARTS), ('birth_date', BIRTH_PARTS)):
            if set(parts).issubset(fields):
                fields.append(parent)
        order.update({(record['record_id'], field): (i, j) for j, field in enumerate(fields)})
    if {(r['record_id'], r['field']) for r in rows} != set(order) or len(rows) != len(order):
        raise ValueError('Missing or duplicate evaluation records/questions')
    rows.sort(key=lambda row: order[(row['record_id'], row['field'])])
    summary = summarize(rows)
    summary.update({'status': 'complete', 'seconds': runtime.maximum(time.monotonic() - start),
                    'metadata': metadata or {}, 'parallelism': runtime.metadata()})
    if output is not None and runtime.primary:
        try:
            write_predictions_csv(output / 'predictions.csv', rows, metadata)
            merged = output / 'predictions.merged.tmp.jsonl'
            with merged.open('w', encoding='utf-8') as stream:
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            os.replace(merged, output / 'predictions.jsonl')
            write_json(output / 'summary.json', summary)
        except Exception as exc:
            write_json(output / 'failure.json', {'status': 'failed', 'phase': 'save_predictions',
                       'error_type': type(exc).__name__, 'error': str(exc)})
            raise
    runtime.barrier()
    return summary, rows


def paired_compare(path_a, path_b, *, seed=17, bootstrap_samples=2000):
    def read(path):
        rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]
        summarize(rows)
        mapping = {}
        for row in rows:
            target_fields = row['target_fields'] if row.get('schema_version') == MULTISCENE_SCHEMA else FIELDS
            if row['field'] not in target_fields:
                continue
            key = (row['record_id'], row['field'])
            if key in mapping:
                raise ValueError('Duplicate comparison row')
            mapping[key] = row
        return mapping
    a, b = read(path_a), read(path_b)
    if a.keys() != b.keys():
        raise ValueError('Paired comparison requires identical complete patient/field sets')
    if any(a[k]['gold'] != b[k]['gold'] or a[k]['question'] != b[k]['question'] for k in a):
        raise ValueError('Question or gold changed between conditions')
    if next(iter(a.values())).get('schema_version') == MULTISCENE_SCHEMA:
        return compare_multiscene(a, b, seed=seed, bootstrap_samples=bootstrap_samples)
    patients = sorted({k[0] for k in a})
    difference = np.array([[float(a[(p, f)]['em']) - float(b[(p, f)]['em']) for f in FIELDS] for p in patients])
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(patients), size=(bootstrap_samples, len(patients)))
    means = difference[samples].mean(axis=1)
    field = {f: {'difference': float(difference[:, i].mean()),
                 'ci95': np.quantile(means[:, i], [.025, .975]).tolist()} for i, f in enumerate(FIELDS)}
    return {'definition': 'condition A minus condition B; resample patients, keeping four fields together',
            'name_protocol_a': next(iter(a.values())).get('name_mode', 'full'),
            'name_protocol_b': next(iter(b.values())).get('name_mode', 'full'),
            'name_protocols_differ': next(iter(a.values())).get('name_mode', 'full') != next(iter(b.values())).get('name_mode', 'full'),
            'birth_date_protocol_a': next(iter(a.values())).get('birth_date_mode', 'full'),
            'birth_date_protocol_b': next(iter(b.values())).get('birth_date_mode', 'full'),
            'birth_date_protocols_differ': next(iter(a.values())).get('birth_date_mode', 'full') != next(iter(b.values())).get('birth_date_mode', 'full'),
            'records': len(patients), 'bootstrap_samples': bootstrap_samples, 'seed': seed,
            'field_em_difference': field,
            'macro_em_difference': {'difference': float(difference.mean()), 'ci95': np.quantile(means.mean(axis=1), [.025, .975]).tolist()}}


def compare_multiscene(a, b, *, seed, bootstrap_samples):
    """Resample connected source/donor groups to keep related records together."""
    by_record = collections.defaultdict(list)
    for key, row in a.items():
        if any(row.get(k) != b[key].get(k) for k in ('schema_version', 'scene', 'record_form', 'group_id')):
            raise ValueError('Paired multiscene metadata differs')
        by_record[key[0]].append((row, float(row['em']) - float(b[key]['em'])))
    fields = sorted({key[1] for key in a})
    groups = sorted({r['group_id'] for r in a.values()})
    group_index = {g: i for i, g in enumerate(groups)}
    sums = np.zeros((len(groups), len(fields) + 1))
    counts = np.zeros_like(sums)
    for values in by_record.values():
        group = group_index[values[0][0]['group_id']]
        for row, difference in values:
            i = fields.index(row['field'])
            sums[group, i] += difference
            counts[group, i] += 1
        sums[group, -1] += np.mean([d for _, d in values])
        counts[group, -1] += 1
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(bootstrap_samples):
        weights = np.bincount(rng.integers(len(groups), size=len(groups)), minlength=len(groups))
        denominator = weights @ counts
        samples.append(np.divide(weights @ sums, denominator, out=np.full(len(fields) + 1, np.nan), where=denominator > 0))
    samples = np.asarray(samples)
    point = sums.sum(axis=0) / counts.sum(axis=0)
    def entry(i):
        valid = samples[:, i][np.isfinite(samples[:, i])]
        return {'difference': float(point[i]), 'ci95': np.quantile(valid, [.025, .975]).tolist(),
                'valid_bootstrap_samples': len(valid)}
    return {'definition': 'A minus B; resample connected source/donor groups, retaining all available fields',
            'records': len(by_record), 'connected_groups': len(groups), 'seed': seed,
            'bootstrap_samples': bootstrap_samples,
            'field_em_difference': {f: entry(i) for i, f in enumerate(fields)},
            'macro_em_difference': entry(-1)}

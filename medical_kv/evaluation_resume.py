"""Recover only complete record predictions from a frozen evaluation snapshot."""
import collections
import json
import math
from pathlib import Path

from .config import file_hash
from .tasks import BIRTH_PARTS, NAME_PARTS, MULTISCENE_SCHEMA, questions_for


def resume_source(directory):
    directory = Path(directory)
    metadata = json.loads((directory / 'metadata.json').read_text(encoding='utf-8'))
    snapshot = directory / 'checkpoint.pt'
    if (metadata.get('evaluation_stage') != 'interim' or metadata.get('condition') != 'soft'
            or metadata.get('split') != 'test' or metadata.get('partial_evaluation') is not False):
        raise ValueError('Resume requires a snapshot-backed full interim soft test')
    if not snapshot.is_file() or file_hash(snapshot) != metadata.get('source_checkpoint_sha256'):
        raise ValueError('Evaluation snapshot is missing or changed; cannot mix predictions')
    return metadata, snapshot


def validate_snapshot(checkpoint, metadata):
    if (checkpoint['method'] != 'soft' or checkpoint['run_signature'] != metadata['checkpoint_run_signature']
            or checkpoint['state']['epoch'] != metadata['epoch'] or checkpoint['state']['step'] != metadata['step']
            or checkpoint['run_info'].get('partial_dataset', True)):
        raise ValueError('Evaluation snapshot does not match the original full-data test')
    cursor = metadata.get('records_seen_in_current_epoch')
    if cursor is not None and checkpoint['state'].get('cursor') != cursor:
        raise ValueError('Evaluation snapshot cursor changed')


def validate_metadata(old, current):
    for key in ('condition', 'split', 'dataset', 'backend', 'generation',
                'checkpoint_run_signature', 'epoch', 'step'):
        if old.get(key) != current.get(key):
            raise ValueError(f'Evaluation resume identity mismatch: {key}')
    def semantic_config(cfg):
        return {k: v for k, v in cfg.items() if k not in ('output_directory', 'device')}
    if semantic_config(old['resolved_config']) != semantic_config(current['resolved_config']):
        raise ValueError('Evaluation resume identity mismatch: config')


def expected_fields(record, cfg):
    fields = [q['field'] for q in questions_for(record, cfg, evaluation=True)]
    for parent, parts in (('name', NAME_PARTS), ('birth_date', BIRTH_PARTS)):
        if set(parts).issubset(fields):
            fields.append(parent)
    return fields


def read_partial(path):
    """A killed writer may leave one torn final line; interior corruption is fatal."""
    with Path(path).open('rb') as stream:
        for line in stream:
            try:
                yield json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                if line.endswith(b'\n') or stream.read(1):
                    raise ValueError(f'Corrupt prediction line in {path}')
                return


def completed_predictions(directory, records, cfg):
    from .evaluation import summarize
    directory = Path(directory)
    final = directory / 'predictions.jsonl'
    paths = ([final] if final.is_file() else
             sorted(directory.glob('predictions.rank*.partial.jsonl'))
             + [p for p in (directory / 'predictions.partial.jsonl', directory / 'reused_predictions.jsonl') if p.is_file()])
    if not paths:
        raise FileNotFoundError('No saved prediction files found in the source evaluation')
    by_id = {r['record_id']: r for r in records}
    predictions = collections.defaultdict(dict)
    duplicates = 0
    for path in paths:
        for row in read_partial(path):
            record_id, field = row['record_id'], row['field']
            if record_id not in by_id:
                raise ValueError(f'Source predictions contain an unknown record: {record_id}')
            previous = predictions[record_id].get(field)
            if previous is not None:
                if previous != row:
                    raise ValueError(f'Conflicting duplicate prediction: {record_id}/{field}')
                duplicates += 1
            predictions[record_id][field] = row
    completed, incomplete = [], []
    for record in records:
        record_id = record['record_id']
        patient = predictions.get(record_id, {})
        questions = questions_for(record, cfg, evaluation=True)
        fields = expected_fields(record, cfg)
        if set(patient) - set(fields):
            raise ValueError(f'Unexpected question fields in source: {record_id}')
        for question in questions:
            row = patient.get(question['field'])
            if row is not None and (row['gold'] != question['answer'] or row['question'] != question['question']):
                raise ValueError(f'Source question/gold changed: {record_id}/{question["field"]}')
        for field, row in patient.items():
            if not math.isfinite(row['nll']):
                raise ValueError(f'Non-finite source prediction: {record_id}/{field}')
            if field in record['fields'] and row['gold'] != record['fields'][field]:
                raise ValueError(f'Source target changed: {record_id}/{field}')
            if record.get('schema_version') == MULTISCENE_SCHEMA:
                if (any(row.get(k) != record[k] for k in ('schema_version','scene','record_form','source','group_id'))
                        or row.get('target_fields') != list(record['fields'])
                        or row.get('question_fields') != [q['field'] for q in questions]):
                    raise ValueError(f'Source record metadata changed: {record_id}')
        if set(patient) != set(fields):
            if patient:
                incomplete.append(record_id)
            continue
        rows = [patient[f] for f in fields]
        summarize(rows)  # Check reconstructed fields and complete question sets.
        completed.extend(rows)
    ids = {r['record_id'] for r in completed}
    return completed, {'completed_records': len(ids), 'remaining_records': len(records) - len(ids),
                       'incomplete_records_replayed': incomplete, 'identical_duplicate_rows': duplicates,
                       'source_files': [p.name for p in paths]}

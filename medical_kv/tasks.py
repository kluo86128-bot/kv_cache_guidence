"""Derive question views without changing frozen records or private prefixes."""
import re
from .config import FIELDS

BIRTH_PARTS = ('birth_year', 'birth_month', 'birth_day')
NAME_PARTS = ('name_first', 'name_middle', 'name_last')
NAME_KEYS = ('FIRST', 'MIDDLE', 'LAST')
NAME_QUESTIONS = (
    "What is the patient's first name? Return only the field value.",
    "What is the patient's middle name? Return only the field value. If the field is empty, end the answer immediately.",
    "What is the patient's last name? Return only the field value.",
)
PART_SLICES = (slice(0, 4), slice(5, 7), slice(8, 10))
PART_QUESTIONS = (
    "What is the patient's birth year? Return exactly four digits.",
    "What is the patient's birth month? Return exactly two digits.",
    "What is the patient's birth day? Return exactly two digits.",
)


MULTISCENE_SCHEMA = 'multiscene_privacy_v1'


def validate_multiscene_questions(record):
    """Check frozen question labels without deriving or rewriting their wording."""
    fields = record['fields']
    canonical = record['qa']
    if (not fields or len(canonical) != len(fields)
            or {q['field'] for q in canonical} != set(fields)
            or any(q['answer'] != fields[q['field']] or q.get('group') != q['field'] for q in canonical)):
        raise ValueError('Canonical questions must cover each available original field exactly once')
    expected = dict(fields)
    if 'name' in fields:
        source = record['source_name_fields']
        if (any(not isinstance(source.get(k), str) for k in NAME_KEYS)
                or ' '.join(source[k] for k in NAME_KEYS if source[k]) != fields['name']):
            raise ValueError('Name source components do not reconstruct the full name')
        del expected['name']
        expected.update(zip(NAME_PARTS, (source[k] for k in NAME_KEYS)))
    birthday = fields.get('birth_date', '')
    if re.fullmatch(r'[0-9]{4}-[0-9]{2}-[0-9]{2}', birthday):
        del expected['birth_date']
        expected.update(zip(BIRTH_PARTS, (birthday[s] for s in PART_SLICES)))
    groups = {**{f: f for f in fields}, **dict.fromkeys(NAME_PARTS, 'name'),
              **dict.fromkeys(BIRTH_PARTS, 'birth_date')}
    train = record['training_qa']
    if (len(train) != len(expected) or {q['field'] for q in train} != set(expected)
            or any(q['answer'] != expected[q['field']] or q.get('group') != groups[q['field']]
                   or not q.get('question') for q in train)):
        raise ValueError('Training questions must match complete available fields/components; full-name training is forbidden')
    evaluation = record['evaluation_qa']
    expected_eval = dict(expected)
    if 'name' in fields:
        expected_eval['name_full_direct'] = fields['name']
    if (len(evaluation) != len(expected_eval) or {q['field'] for q in evaluation} != set(expected_eval)
            or any(q['answer'] != expected_eval[q['field']] or not q.get('question') for q in evaluation)):
        raise ValueError('Evaluation questions must retain components and add an independent full-name question')
    by_field = {q['field']: q for q in evaluation}
    if any(any(by_field[q['field']].get(k) != q.get(k) for k in ('question', 'answer', 'group')) for q in train):
        raise ValueError('Evaluation component/field questions differ from training protocol')
    if 'name' in fields:
        direct = by_field['name_full_direct']
        if direct.get('group') != 'name' or direct.get('scoring_target') != 'name':
            raise ValueError('Direct full-name evaluation must score the original name separately')


def questions_for(record, cfg, *, evaluation=False):
    if record.get('schema_version') == MULTISCENE_SCHEMA:
        if cfg.get('question_protocol') != MULTISCENE_SCHEMA:
            raise ValueError('New dataset requires question_protocol=multiscene_privacy_v1')
        if cfg.get('name_mode') != 'parts' or cfg.get('birth_date_mode') != 'parts':
            raise ValueError('New dataset uses frozen name/date component questions; modes must be parts')
        validate_multiscene_questions(record)
        return record['evaluation_qa' if evaluation else 'training_qa']
    birthday_mode, name_mode = cfg.get('birth_date_mode', 'full'), cfg.get('name_mode', 'full')
    if birthday_mode not in ('full', 'parts'):
        raise ValueError('Unknown birthday question mode')
    if name_mode not in ('full', 'parts'):
        raise ValueError('Unknown name question mode')
    if birthday_mode == 'full' and name_mode == 'full':
        return record['qa']
    if birthday_mode == 'parts':
        birthday = record['fields']['birth_date']
        if re.fullmatch(r'[0-9]{4}-[0-9]{2}-[0-9]{2}', birthday) is None:
            raise ValueError('Split birthday requires an unchanged YYYY-MM-DD source value')
    if name_mode == 'parts':
        source = record.get('source_name_fields', {})
        if any(not isinstance(source.get(k), str) for k in NAME_KEYS):
            raise ValueError('Split name requires original FIRST/MIDDLE/LAST strings, including empty MIDDLE')
        if ' '.join(source[k] for k in NAME_KEYS if source[k]) != record['fields']['name']:
            raise ValueError('Split name source components do not match the original full name')
    questions = []
    for qa in record['qa']:
        if qa['field'] == 'name' and name_mode == 'parts':
            questions.extend({'field': field, 'question': question, 'answer': source[key]}
                             for field, key, question in zip(NAME_PARTS, NAME_KEYS, NAME_QUESTIONS))
        elif qa['field'] == 'birth_date' and birthday_mode == 'parts':
            questions.extend({'field': field, 'question': question, 'answer': birthday[part]}
                             for field, question, part in zip(BIRTH_PARTS, PART_QUESTIONS, PART_SLICES))
        else:
            questions.append(qa)
    return questions


def grouped_loss(losses, questions):
    """Token means -> component means -> equal available original fields."""
    if any('group' in q for q in questions):
        fields = [q['field'] for q in questions]
        if (losses.ndim != 1 or len(losses) != len(fields) or not fields
                or len(set(fields)) != len(fields) or 'name_full_direct' in fields
                or any(not q.get('group') for q in questions)):
            raise ValueError('Grouped training requires complete unique original-field/component questions')
        grouped = {}
        for loss, q in zip(losses.unbind(), questions):
            grouped.setdefault(q['group'], []).append(loss)
        for parent, parts in (('name', NAME_PARTS), ('birth_date', BIRTH_PARTS)):
            actual = {q['field'] for q in questions if q['group'] == parent}
            if actual and actual not in ({parent}, set(parts)):
                raise ValueError('Grouped training requires complete unique component questions')
        return sum(sum(values) / len(values) for values in grouped.values()) / len(grouped)
    fields = [qa['field'] for qa in questions]
    expected = set(FIELDS)
    group_by_field = {f: f for f in FIELDS}
    for parent, parts in (('name', NAME_PARTS), ('birth_date', BIRTH_PARTS)):
        group_by_field.update({part: parent for part in parts})
        if set(fields) & set(parts):
            expected.remove(parent)
            expected.update(parts)
    if losses.ndim != 1 or len(losses) != len(fields) or len(fields) != len(set(fields)) or set(fields) != expected:
        raise ValueError('Loss aggregation requires all four groups with complete unique component questions')
    grouped = {field: [] for field in FIELDS}
    for loss, qa in zip(losses.unbind(), questions):
        group = group_by_field[qa['field']]
        grouped[group].append(loss)
    return sum(sum(values) / len(values) for values in grouped.values()) / len(FIELDS)


def reconstruct_name(record, rows):
    """Join predicted nonempty components; never consult gold to decide which to omit."""
    parts = {row['field']: row for row in rows if row['field'] in NAME_PARTS}
    if not parts:
        return None
    if set(parts) != set(NAME_PARTS):
        raise ValueError('Incomplete name components')
    values = [parts[field]['prediction'].strip() for field in NAME_PARTS]
    prediction = ' '.join(value for value in values if value)
    original = next(qa for qa in record['qa'] if qa['field'] == 'name')
    return {'record_id': record['record_id'], 'field': 'name', 'question': original['question'],
            'gold': original['answer'], 'prediction': prediction, 'em': prediction == original['answer'],
            'nll': sum(parts[f]['nll'] for f in NAME_PARTS) / 3,
            'generated_tokens': sum(parts[f]['generated_tokens'] for f in NAME_PARTS),
            'stop_reason': 'reconstructed', 'truncated': any(parts[f]['truncated'] for f in NAME_PARTS),
            'prediction_kind': 'reconstructed', 'name_mode': 'parts'}


def reconstruct_birthday(record, rows, seen_birthdays=None):
    """A derived scoring row, not an additional model query or a repaired answer."""
    parts = {row['field']: row for row in rows if row['field'] in BIRTH_PARTS}
    if not parts:
        return None
    if set(parts) != set(BIRTH_PARTS):
        raise ValueError('Incomplete birthday components')
    values = [parts[field]['prediction'].strip() for field in BIRTH_PARTS]
    prediction = '-'.join(values)
    original = next(qa for qa in record['qa'] if qa['field'] == 'birth_date')
    result = {'record_id': record['record_id'], 'field': 'birth_date', 'question': original['question'],
              'gold': original['answer'], 'prediction': prediction, 'em': prediction == original['answer'],
              'nll': sum(parts[f]['nll'] for f in BIRTH_PARTS) / 3,
              'generated_tokens': sum(parts[f]['generated_tokens'] for f in BIRTH_PARTS),
              'stop_reason': 'reconstructed', 'truncated': any(parts[f]['truncated'] for f in BIRTH_PARTS),
              'prediction_kind': 'reconstructed', 'birth_date_mode': 'parts'}
    if seen_birthdays is not None:
        result['birth_date_seen_in_train'] = original['answer'] in seen_birthdays
    return result

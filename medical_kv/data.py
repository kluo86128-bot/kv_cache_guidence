import json
from .config import FIELDS, file_hash, resolve
from .name_policy import NAME_FORMAT, has_name_digits
from .tasks import MULTISCENE_SCHEMA, validate_multiscene_questions


def available_splits(cfg):
    root = resolve(cfg['dataset_directory'])
    return tuple(s for s in ('dev', 'train', 'validation', 'test') if (root / (s + '.jsonl')).is_file())


def selected_record(record, cfg):
    return (cfg.get('scene') in (None, record.get('scene'))
            and cfg.get('record_form') in (None, record.get('record_form')))


def development_partition(train):
    boundary = min(64, len(train) // 2)
    left = train[:boundary]
    groups = {r['group_id'] for r in left}
    right = [r for r in train[boundary:] if r['group_id'] not in groups][:64]
    if not left or not right:
        raise ValueError('Development needs at least two isolated training groups')
    return left, right


def dataset_identity(cfg):
    root = resolve(cfg['dataset_directory'])
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    report = json.loads((root / 'validation_report.json').read_text(encoding='utf-8'))
    hashes = json.loads((root / 'artifact_hashes.json').read_text(encoding='utf-8'))
    if manifest['schema_version'] == MULTISCENE_SCHEMA:
        if cfg.get('question_protocol') != MULTISCENE_SCHEMA or report['status'] != 'PASS':
            raise ValueError('Validated multiscene data requires its matching question protocol')
        for name, expected in hashes.items():
            if file_hash(root / name) != expected:
                raise ValueError(f'Dataset artifact hash mismatch: {name}')
        tokenization = json.loads((root / 'tokenization_report.json').read_text(encoding='utf-8'))
        identity = {'dataset_id': manifest['dataset_id'], 'manifest_sha256': file_hash(root / 'manifest.json'),
                'split_sha256': {s: hashes[s + '.jsonl'] for s in available_splits(cfg)},
                'tokenizer_sha256': tokenization['tokenizer_sha256'], 'schema': MULTISCENE_SCHEMA,
                'selection': {k: cfg.get(k) for k in ('scene', 'record_form')},
                'manual_review_complete': report.get('manual_review_complete', False)}
        if cfg.get('input_cache_directory'):
            from .input_cache import cache_manifest
            cache_manifest(cfg)
            identity['input_cache_manifest_sha256']=file_hash(resolve(cfg['input_cache_directory'])/'manifest.json')
        return identity
    if cfg.get('question_protocol') == MULTISCENE_SCHEMA:
        raise ValueError('Multiscene configuration cannot load a legacy four-field dataset')
    if manifest.get('name_format') != cfg.get('name_format', NAME_FORMAT):
        raise ValueError('Dataset name format mismatch: upload the complete natural-name dataset and retrain; numbered-name checkpoints are obsolete')
    if report['status'] != 'PASS' or tuple(manifest['target_fields']) != FIELDS:
        raise ValueError('Dataset must have passed validation with the four agreed fields')
    for name in ('manifest.json', 'validation_report.json', 'dev.jsonl', 'train.jsonl', 'validation.jsonl', 'test.jsonl'):
        if file_hash(root / name) != hashes[name]:
            raise ValueError(f'Dataset artifact hash mismatch: {name}')
    return {'dataset_id': manifest['dataset_id'], 'manifest_sha256': file_hash(root / 'manifest.json'),
            'split_sha256': {s: hashes[s + '.jsonl'] for s in ('dev', 'train', 'validation', 'test')},
            'tokenizer_sha256': manifest['tokenizer']['sha256'], 'schema': manifest['schema_version']}


def load_split(cfg, split, limit=None):
    if split not in ('dev', 'train', 'validation', 'test'):
        raise ValueError('Unknown split')
    path = resolve(cfg['dataset_directory']) / (split + '.jsonl')
    if split == 'dev' and not path.exists() and cfg.get('question_protocol') == MULTISCENE_SCHEMA:
        # An implementation-only view of TRAINING records, never formal validation/test.
        train = load_split(cfg, 'train')
        left, right = development_partition(train)
        records = left + right
        if limit is not None and not 1 <= limit <= len(records):
            raise ValueError('Requested limit exceeds development view size')
        return records[:limit] if limit else records
    records = []
    seen = set()
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            r = json.loads(line)
            if not selected_record(r, cfg):
                continue
            if r.get('schema_version') == MULTISCENE_SCHEMA:
                if cfg.get('question_protocol') != MULTISCENE_SCHEMA:
                    raise ValueError('New dataset requires the multiscene question protocol')
                if r['split'] != split or r['record_id'] in seen:
                    raise ValueError('Invalid multiscene record split/duplicate')
                if not r.get('scene') or not r.get('record_form') or not r.get('group_id'):
                    raise ValueError('Missing multiscene grouping metadata')
                validate_multiscene_questions(r)
                if any(has_name_digits(v) for v in r.get('source_name_fields', {}).values()):
                    raise ValueError('Numbered names are forbidden')
                for field, value in r['fields'].items():
                    a = r['field_annotations'][field]
                    start, end = a['evidence_span']
                    if not isinstance(value, str) or r['private_prefix'][start:end] != value:
                        raise ValueError(f'Target field lacks unchanged context evidence: {field}')
            else:
                _validate_legacy(r, split, seen)
            seen.add(r['record_id'])
            records.append(r)
    if not records:
        raise ValueError(f'No records match split/scene/record_form: {split}')
    if limit is not None:
        if not 1 <= limit <= len(records):
            raise ValueError('Requested limit exceeds split size')
        records = records[:limit]
    if cfg.get('input_cache_directory'):
        from .input_cache import attach_cache
        attach_cache(cfg,records,split)
    return records


def _validate_legacy(r, split, seen):
    if (r.get('name_format') != NAME_FORMAT or has_name_digits(r['fields']['name'])
            or any(has_name_digits(v) for v in r['source_name_fields'].values())):
        raise ValueError('Numbered or unversioned patient names are not allowed in current splits')
    if r['split'] != split or tuple(r['fields']) != FIELDS or r['record_id'] in seen:
        raise ValueError(f'Invalid record schema/split/duplicate: {r.get("record_id")}')
    if len(r['qa']) != 4 or {q['field'] for q in r['qa']} != set(FIELDS):
        raise ValueError('Each record must contain exactly four different field questions')
    if any(q['answer'] != r['fields'][q['field']] for q in r['qa']):
        raise ValueError('Question labels do not match source-preserving fields')
    if 'patients.Id:' in r['private_prefix'] or 'encounters.START:' in r['private_prefix']:
        raise ValueError('Metadata identifier/time leaked into private_prefix')


def training_data(cfg, development=False):
    # Development smoke runs never use the formal validation/test split.
    if development:
        if cfg.get('question_protocol') == MULTISCENE_SCHEMA:
            train, validation = development_partition(load_split(cfg, 'train'))
        else:
            dev = load_split(cfg, 'dev')
            half = len(dev) // 2
            train, validation = dev[:half], dev[half:]
        if cfg['train_limit']:
            train = train[:cfg['train_limit']]
        if cfg['validation_limit']:
            validation = validation[:cfg['validation_limit']]
    else:
        train = load_split(cfg, 'train', cfg['train_limit'])
        validation = load_split(cfg, 'validation', cfg['validation_limit'])
    identity = {'development_only': development,
                'train_ids': [r['record_id'] for r in train],
                'validation_ids': [r['record_id'] for r in validation]}
    return train, validation, identity

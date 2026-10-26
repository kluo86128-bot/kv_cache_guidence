import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ('name', 'birth_date', 'address', 'diagnosis')


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def fingerprint(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_config(path, overrides=None):
    cfg = path if isinstance(path, dict) else json.loads(resolve(path).read_text(encoding='utf-8'))
    cfg = deepcopy(cfg)
    for key, value in (overrides or {}).items():
        if value is not None:
            cfg[key] = value
    for key in ('prompt_length', 'batch_records', 'epochs', 'patience', 'max_new_tokens', 'max_runtime_seconds'):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    for key in ('train_limit', 'validation_limit', 'max_updates'):
        if cfg[key] is not None and (type(cfg[key]) is not int or cfg[key] < 1):
            raise ValueError(f'{key} must be null or a positive integer')
    if cfg['question_format'] not in ('chat', 'plain'):
        raise ValueError('question_format must be chat or plain')
    if cfg['enable_thinking'] is not False:
        raise ValueError('This experiment requires enable_thinking=false')
    cfg.setdefault('batch_questions', True)
    cfg.setdefault('question_protocol', 'legacy')
    cfg.setdefault('question_batch_size', None)
    cfg.setdefault('checkpoint_questions', False)
    cfg.setdefault('split_private_prefill', False)
    if type(cfg['checkpoint_questions']) is not bool:
        raise ValueError('checkpoint_questions must be boolean')
    if type(cfg['split_private_prefill']) is not bool:
        raise ValueError('split_private_prefill must be boolean')
    cfg.setdefault('scene', None)
    cfg.setdefault('record_form', None)
    if cfg['question_protocol'] not in ('legacy', 'multiscene_privacy_v1'):
        raise ValueError('Unknown question_protocol')
    if cfg['scene'] not in (None, 'medical', 'finance', 'hr') or cfg['record_form'] not in (None, 'structured', 'narrative', 'dialogue', 'mixed'):
        raise ValueError('Invalid scene or record_form')
    if cfg['question_batch_size'] is not None and (type(cfg['question_batch_size']) is not int or cfg['question_batch_size'] < 1):
        raise ValueError('question_batch_size must be null or a positive integer')
    cfg.setdefault('name_format', 'natural_no_numeric_suffix_v1')
    if cfg['name_format'] != 'natural_no_numeric_suffix_v1':
        raise ValueError('This experiment now uses natural names without numeric suffixes')
    cfg.setdefault('birth_date_mode', 'full')
    cfg.setdefault('name_mode', 'full')
    cfg.setdefault('early_stopping', True)
    cfg.setdefault('comparison_epochs', None)
    if cfg['birth_date_mode'] not in ('full', 'parts'):
        raise ValueError('birth_date_mode must be full or parts')
    if cfg['name_mode'] not in ('full', 'parts'):
        raise ValueError('name_mode must be full or parts')
    if type(cfg['early_stopping']) is not bool:
        raise ValueError('early_stopping must be a boolean')
    if cfg['comparison_epochs'] is not None and (type(cfg['comparison_epochs']) is not int or cfg['comparison_epochs'] < 1):
        raise ValueError('comparison_epochs must be null or a positive integer')
    if type(cfg['batch_questions']) is not bool:
        raise ValueError('batch_questions must be a boolean')
    contrastive = cfg.setdefault('diagnosis_contrastive', {'enabled': False})
    # Do not insert a new default key into legacy CE configs/checkpoint identities.
    loss = cfg.get('loss', {'type': 'ce'})
    if not isinstance(loss, dict) or loss.get('type') not in ('ce', 'focal', 'ce_focal_birth', 'birth_focal', 'birth_focal_split', 'birth_focal_pcgrad'):
        raise ValueError('Unknown loss.type; expected ce, focal, ce_focal_birth, birth_focal, birth_focal_split or birth_focal_pcgrad')
    allowed = {'type'} if loss['type'] == 'ce' else {'type', 'gamma', 'alpha'}
    if loss['type'] == 'ce_focal_birth':
        allowed = allowed | {'weight'}
    if set(loss) != allowed:
        raise ValueError('Invalid loss keys: CE needs type; focal/birth_focal need type/gamma/alpha; ce_focal_birth also needs weight')
    if loss['type'] != 'ce':
        for key in (('gamma', 'alpha', 'weight') if loss['type'] == 'ce_focal_birth' else ('gamma', 'alpha')):
            value = loss[key]
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (key == 'alpha' and value == 0):
                raise ValueError('Focal gamma/weight must be finite and nonnegative; alpha must be finite and positive')
        if contrastive.get('enabled') or cfg['hard'].get('enabled'):
            raise ValueError('Focal supports soft training without contrastive negatives or discrete search')
    partition = cfg.get('prompt_partition')
    if loss['type'] == 'birth_focal_pcgrad':
        if (cfg.get('prompt_initialization') != 'fresh_token_embeddings'
                or cfg['question_protocol'] != 'multiscene_privacy_v1' or cfg['scene'] != 'medical'):
            raise ValueError('birth_focal_pcgrad requires fresh_token_embeddings and the medical multiscene protocol')
    if loss['type'] == 'birth_focal_split':
        if cfg.get('prompt_initialization') != 'fresh_token_embeddings':
            raise ValueError('birth_focal_split requires prompt_initialization=fresh_token_embeddings')
        if (not isinstance(partition, dict) or set(partition) != {'other_length', 'birth_length'}
                or any(type(partition[k]) is not int or partition[k] < 1 for k in partition)
                or sum(partition.values()) != cfg['prompt_length']):
            raise ValueError('prompt_partition needs positive other_length/birth_length summing to prompt_length')
        if cfg['question_protocol'] != 'multiscene_privacy_v1' or cfg['scene'] != 'medical':
            raise ValueError('birth_focal_split requires the medical multiscene protocol')
    elif partition is not None:
        raise ValueError('prompt_partition requires loss.type=birth_focal_split')
    if cfg['question_protocol'] == 'multiscene_privacy_v1':
        if cfg['name_mode'] != 'parts' or cfg['birth_date_mode'] != 'parts':
            raise ValueError('Multiscene name_mode and birth_date_mode must be parts')
        if contrastive.get('enabled') or cfg['hard'].get('enabled'):
            raise ValueError('Multiscene v1 supports soft training without diagnosis negatives or discrete search')
    if type(contrastive.get('enabled')) is not bool:
        raise ValueError('diagnosis_contrastive.enabled must be a boolean')
    if contrastive['enabled']:
        for key in ('weight', 'margin'):
            value = contrastive.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f'diagnosis_contrastive.{key} must be finite and nonnegative')
        if 'negatives_per_record' in contrastive:
            raise ValueError('diagnosis_contrastive: fixed-count config is obsolete; use max_negatives_per_record=4')
        if contrastive.get('max_negatives_per_record') != 4 or type(contrastive['max_negatives_per_record']) is not int:
            raise ValueError('diagnosis_contrastive.max_negatives_per_record must be four (one per available kind)')
        batch_size = contrastive.get('negative_batch_size')
        if type(batch_size) is not int or not 1 <= batch_size <= 4:
            raise ValueError('diagnosis_contrastive.negative_batch_size must be in [1, 4]')
        if type(contrastive.get('checkpoint_negatives')) is not bool:
            raise ValueError('diagnosis_contrastive.checkpoint_negatives must be boolean')
        count = contrastive.get('candidates_per_kind')
        if type(count) is not int or not 3 <= count <= 5:
            raise ValueError('diagnosis_contrastive.candidates_per_kind must be in [3, 5] (available counts may be smaller)')
        if not isinstance(contrastive.get('rules_file'), str) or not contrastive['rules_file']:
            raise ValueError('diagnosis_contrastive.rules_file must name the fixed matching rules')
        for key in ('resources_file', 'pool_directory'):
            if not isinstance(contrastive.get(key), str) or not contrastive[key]:
                raise ValueError(f'diagnosis_contrastive.{key} must be a nonempty path')
        if type(contrastive.get('pool_seed')) is not int or contrastive['pool_seed'] < 0:
            raise ValueError('diagnosis_contrastive.pool_seed must be a nonnegative integer')
        count = contrastive.get('lexical_candidates_per_diagnosis')
        if type(count) is not int or count < 5:
            raise ValueError('diagnosis_contrastive.lexical_candidates_per_diagnosis must be at least five')
        threshold = contrastive.get('lexical_preferred_similarity')
        if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 < threshold <= 1:
            raise ValueError('diagnosis_contrastive.lexical_preferred_similarity must be in (0, 1]')
        if any(key in contrastive for key in ('lexical_min_similarity', 'excluded_test_datasets', 'training_predictions_csv', 'source_conditions_csv')):
            raise ValueError('diagnosis_contrastive: obsolete generation options; use the current offline top-K configuration')
    if cfg['attention_implementation'] != 'eager':
        raise ValueError('v1 validates explicit 4D masks with eager attention only')
    if cfg['dtype'] not in ('float32', 'float16', 'bfloat16'):
        raise ValueError('Unsupported dtype')
    if cfg['learning_rate'] <= 0 or cfg['grad_clip'] <= 0 or cfg['weight_decay'] != 0:
        raise ValueError('Positive LR/clip and weight_decay=0 are required')
    if not 0 <= cfg['warmup_fraction'] < 1 or not cfg['public_text'] or not cfg['separator']:
        raise ValueError('Invalid warmup/public text/separator')
    cfg['hard'].setdefault('enabled', False)
    if type(cfg['hard']['enabled']) is not bool:
        raise ValueError('hard.enabled must be a boolean')
    for key in ('steps', 'gradient_records', 'candidate_records', 'top_k', 'candidate_count', 'validate_every'):
        if type(cfg['hard'][key]) is not int or cfg['hard'][key] < 1:
            raise ValueError(f'hard.{key} must be a positive integer')
    return cfg

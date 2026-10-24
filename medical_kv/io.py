import json
import os
import random
import signal
import time
from contextlib import contextmanager
from pathlib import Path
import torch
from filelock import FileLock
from .config import ROOT, file_hash, fingerprint


def write_json(path, obj):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def append_json(path, obj):
    with Path(path).open('a', encoding='utf-8') as f:
        f.write(json.dumps(obj, ensure_ascii=False) + '\n')


def save_checkpoint(path, obj):
    temporary = Path(path).with_name(Path(path).name + '.tmp')
    torch.save(obj, temporary)
    os.replace(temporary, path)


def read_checkpoint(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return {'python': random.getstate(), 'torch': torch.get_rng_state(),
            'cuda': [torch.cuda.get_rng_state()] if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        if not torch.cuda.is_available():
            raise ValueError('Cannot resume CUDA RNG state on a CPU-only host')
        torch.cuda.set_rng_state(state['cuda'][0])


def code_identity():
    return {p.name: file_hash(p) for p in sorted((ROOT / 'medical_kv').glob('*.py'))}


def inference_identity(backend, *, legacy_without_name_format=False):
    cfg = backend.cfg
    keys = (
        'prompt_length', 'public_text', 'separator', 'question_format', 'enable_thinking',
        'dtype', 'attention_implementation')
    if not legacy_without_name_format:
        keys += ('name_format',)
    if cfg.get('question_protocol') == 'multiscene_privacy_v1':
        keys += ('question_protocol', 'split_private_prefill')
    return fingerprint({'model': backend.identity, 'protocol': {k: cfg[k] for k in keys}, 'public_ids': backend.public_ids,
        'question_template': getattr(backend.tokenizer, 'chat_template', None)})


def checkpoint_compatibility(checkpoint, backend, *, allow_name_format_transfer=False):
    """Allow only the historical numbered-name signature as an explicit evaluation exception."""
    from .name_policy import NAME_FORMAT
    trained_cfg = checkpoint['run_info']['config']
    if trained_cfg.get('question_protocol', 'legacy') != backend.cfg.get('question_protocol', 'legacy'):
        raise ValueError('Checkpoint question protocol mismatch: multiscene data requires a newly trained matching prompt')
    trained_format = trained_cfg.get('name_format', 'legacy_numbered_names')
    metadata = {'checkpoint_name_format': trained_format,
                'checkpoint_question_protocol': trained_cfg.get('question_protocol', 'legacy'),
                'evaluation_question_protocol': backend.cfg.get('question_protocol', 'legacy'),
                'evaluation_name_format': backend.cfg['name_format'],
                'checkpoint_name_mode': trained_cfg.get('name_mode', 'full'),
                'evaluation_name_mode': backend.cfg.get('name_mode', 'full'),
                'name_protocol_transfer': trained_cfg.get('name_mode', 'full') != backend.cfg.get('name_mode', 'full'),
                'name_format_transfer': False}
    if checkpoint['inference_signature'] == inference_identity(backend):
        return metadata
    if (allow_name_format_transfer and 'name_format' not in trained_cfg
            and backend.cfg['name_format'] == NAME_FORMAT
            and trained_cfg.get('birth_date_mode', 'full') == backend.cfg['birth_date_mode']
            and trained_cfg.get('name_mode', 'full') == backend.cfg.get('name_mode', 'full')
            and checkpoint['inference_signature'] == inference_identity(backend, legacy_without_name_format=True)):
        return dict(metadata, name_format_transfer=True,
                    evaluation_note='Numbered-name training -> natural-name test; frozen prompt, no retraining. '
                                    'Scores measure transfer, not matched natural-name training performance.')
    raise ValueError('Checkpoint/model/tokenizer/input protocol mismatch. For an old numbered-name checkpoint, '
                     'use evaluate --condition soft --allow-name-format-transfer with the same birth-date mode; '
                     'model, tokenizer and cache protocol must still match.')


@contextmanager
def stop_at_boundary():
    state = {'requested': False}
    old = {}
    def handler(signum, frame):
        state['requested'] = True
        print('Stop requested: saving after the current optimizer step.', flush=True)
    for sig in (signal.SIGINT, signal.SIGTERM):
        old[sig] = signal.signal(sig, handler)
    try:
        yield state
    finally:
        for sig, value in old.items():
            signal.signal(sig, value)


@contextmanager
def run_lock(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with FileLock(str(directory / '.run.lock'), timeout=0):
        yield


def elapsed(start):
    return time.monotonic() - start

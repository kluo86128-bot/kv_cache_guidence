"""Data-free soft-U projection and resumable mapped-text evaluation.

Server examples (run from the project root)::

    python map_soft_u_cosine.py --config configs/soft_u_cosine_mapping.json
    python map_soft_u_cosine.py pause --config configs/soft_u_cosine_mapping.json
    python map_soft_u_cosine.py resume --config configs/soft_u_cosine_mapping.json \
        --mapping-gpus 0 --evaluation-gpus 1,3

Only local safetensors input embeddings are read during projection. No dataset,
forward pass, gradient, optimizer, or change to the source U is involved.
"""
import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import uuid

import torch
from filelock import FileLock

from medical_kv.config import ROOT, file_hash, fingerprint, load_config, resolve
from medical_kv.control import process_start, write_json
from medical_kv.io import read_checkpoint, run_lock, save_checkpoint
from medical_kv.launch import parse_gpus


DEFAULT_CONFIG = 'configs/soft_u_cosine_mapping.json'
FORMAT = 'soft_u_cosine_v1'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


def gpu_ids(value):
    if isinstance(value, list):
        if not value:
            return []
        if any(type(v) is not int or v < 0 for v in value):
            raise ValueError('GPU lists require nonnegative integer IDs')
        return parse_gpus(','.join(map(str, value)))
    if value == 'cpu':
        return []
    return parse_gpus(value)


def load_settings(path):
    cfg = read_json(resolve(path))
    if not isinstance(cfg.get('soft_prefixes'), list) or not cfg['soft_prefixes']:
        raise ValueError('soft_prefixes must be a nonempty list')
    cfg.setdefault('model_name_or_path', None)
    cfg.setdefault('checkpoint_filename', 'best.pt')
    mapping = cfg.setdefault('mapping', {})
    mapping.setdefault('device', 'cpu')
    mapping.setdefault('gpus', [])
    mapping.setdefault('exclude_special_tokens', True)
    mapping.setdefault('allow_non_ascii', True)
    mapping.setdefault('vocabulary_chunk_size', 8192)
    mapping.setdefault('position_chunk_size', 8)
    mapping.setdefault('cpu_threads', 4)
    positive_int(mapping['vocabulary_chunk_size'], 'vocabulary_chunk_size')
    positive_int(mapping['position_chunk_size'], 'position_chunk_size')
    positive_int(mapping['cpu_threads'], 'mapping.cpu_threads')
    if mapping['device'] not in ('cpu', 'cuda'):
        raise ValueError('mapping.device must be cpu or cuda; select GPUs with mapping.gpus')
    mapping['gpus'] = gpu_ids(mapping['gpus'])
    if mapping['device'] == 'cuda' and not mapping['gpus']:
        raise ValueError('CUDA mapping requires an explicit nonempty mapping.gpus list')
    if mapping['device'] == 'cpu' and mapping['gpus']:
        raise ValueError('CPU mapping requires an empty GPU list')
    for key in ('exclude_special_tokens', 'allow_non_ascii'):
        if type(mapping[key]) is not bool:
            raise ValueError(f'mapping.{key} must be boolean')
    ev = cfg.setdefault('evaluation', {})
    ev.setdefault('enabled', True)
    ev.setdefault('split', 'test')
    ev.setdefault('gpus', [0])
    ev.setdefault('cpu_threads', 4)
    ev.setdefault('batch_records_per_gpu', 2)
    positive_int(ev['cpu_threads'], 'evaluation.cpu_threads')
    positive_int(ev['batch_records_per_gpu'], 'evaluation.batch_records_per_gpu')
    ev['gpus'] = gpu_ids(ev['gpus'])
    if type(ev['enabled']) is not bool or ev['split'] not in ('dev', 'validation', 'test'):
        raise ValueError('evaluation needs boolean enabled and dev/validation/test split')
    names = set()
    for entry in cfg['soft_prefixes']:
        name = entry.get('name', '')
        if (not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', name)
                or name.casefold() in names or name.upper() in {'CON', 'PRN', 'AUX', 'NUL'}
                or re.fullmatch(r'(COM|LPT)[1-9]', name.upper())):
            raise ValueError(f'Unsafe or duplicate mapping name: {name!r}')
        names.add(name.casefold())
        if not entry.get('directory') or not entry.get('evaluation_config'):
            raise ValueError(f'{name}: directory and evaluation_config are required')
        filename = entry.get('checkpoint_filename', cfg['checkpoint_filename'])
        if Path(filename).name != filename or filename in ('.', '..'):
            raise ValueError('checkpoint_filename must be a filename, not a path')
        policy = entry.get('checkpoint_policy', 'validation_selected')
        if policy not in ('validation_selected', 'paused_or_complete_latest'):
            raise ValueError('checkpoint_policy must be validation_selected or paused_or_complete_latest')
        if policy == 'paused_or_complete_latest' and filename != 'latest.pt':
            raise ValueError('paused_or_complete_latest requires checkpoint_filename=latest.pt')
    cfg['output_directory'] = str(resolve(cfg['output_directory']).resolve())
    return cfg


def semantic_settings(cfg):
    """Execution topology and chunk sizes may change without changing the task."""
    return {
        'format': FORMAT, 'model_name_or_path': cfg['model_name_or_path'],
        'checkpoint_filename': cfg['checkpoint_filename'],
        'mapping': {k: cfg['mapping'][k] for k in ('exclude_special_tokens', 'allow_non_ascii')},
        'split': cfg['evaluation']['split'], 'soft_prefixes': cfg['soft_prefixes'],
    }


def code_hashes():
    from medical_kv.io import code_identity
    return {**{'medical_kv/' + k: v for k, v in code_identity().items()},
            **{name: file_hash(ROOT / name) for name in
               ('map_soft_u_cosine.py', 'evaluate_mapped_u.py', 'mapped_cache_batch.py',
                'configs/mapped_evaluation_compatibility_v1.json')}}


def compatible_execution_code(recorded, current=None):
    """Only the explicitly reviewed scalar->cache-batch upgrade may reuse rows."""
    current = code_hashes() if current is None else current
    if recorded == current:
        return True
    policy_name = 'configs/mapped_evaluation_compatibility_v1.json'
    policy = read_json(ROOT / policy_name)
    target = policy['target_scripts']
    if any(current.get(name) != digest for name, digest in target.items()):
        return False
    mutable = set(target) | {policy_name}
    if {k: v for k, v in recorded.items() if k not in mutable} != {k: v for k, v in current.items() if k not in mutable}:
        return False
    return {k: v for k, v in recorded.items() if k in mutable} == policy['source_scripts']


def note_runtime_upgrade(directory, recorded, current=None):
    current = code_hashes() if current is None else current
    if not compatible_execution_code(recorded, current):
        raise ValueError('Unreviewed evaluation code change; refusing to mix existing results')
    if recorded == current:
        return
    directory = Path(directory) / 'runtime_upgrades'
    directory.mkdir(exist_ok=True)
    note = {'from_code': recorded, 'to_code': current,
            'policy': 'reviewed_scalar_to_cache_batch_v1',
            'preserved': 'Original signatures, mapped prompt and committed record bytes.',
            'numeric_note': 'Batch shape changes may change floating-point rounding and greedy outputs.'}
    path = directory / (fingerprint(note) + '.json')
    if path.exists():
        if read_json(path) != note:
            raise ValueError('Runtime upgrade record changed')
    else:
        write_json(path, note)


def atomic_text(path, text):
    path = Path(path)
    tmp = path.with_name(path.name + f'.{uuid.uuid4().hex}.tmp')
    with tmp.open('w', encoding='utf-8', newline='') as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def atomic_csv(path, columns, rows):
    path = Path(path)
    tmp = path.with_name(path.name + f'.{uuid.uuid4().hex}.tmp')
    with tmp.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def marker(path, outputs, **extra):
    write_json(path, {**extra, 'files': {p.name: file_hash(p) for p in outputs}})


def verify_marker(path):
    info = read_json(path)
    for name, digest in info['files'].items():
        target = Path(path).parent / name
        if Path(name).name != name or not target.is_file() or file_hash(target) != digest:
            raise ValueError(f'Completed output changed or missing: {target}')
    return info


def validate_prompt(checkpoint):
    prompt = checkpoint.get('prompt')
    if checkpoint.get('method') != 'soft' or not torch.is_tensor(prompt):
        raise ValueError('Expected an existing soft checkpoint with a prompt tensor')
    if (prompt.ndim != 2 or not prompt.is_floating_point() or min(prompt.shape) < 1
            or not torch.isfinite(prompt).all() or (prompt.float().norm(dim=1) == 0).any()):
        raise ValueError('Soft U must be a finite nonzero [positions, hidden_size] tensor')
    if checkpoint['run_info']['config']['prompt_length'] != prompt.shape[0]:
        raise ValueError('Soft U length differs from its saved training configuration')
    return prompt


def validate_test_source(checkpoint, policy='validation_selected'):
    """Select explicitly between final best and a safely frozen latest snapshot."""
    from medical_kv.cli import validate_test_checkpoint
    summary = checkpoint.get('source_training_summary')
    if summary is None:
        raise ValueError('Mapped test requires the source run_summary.json snapshot; safely pause an active run first')
    if policy == 'validation_selected':
        if checkpoint['run_info'].get('partial_dataset', True) or not checkpoint.get('selected_by_validation'):
            raise ValueError('Formal mapped test requires a full-data validation-selected soft checkpoint')
        validate_test_checkpoint(checkpoint, summary)
        return {'source_stage': 'validation_selected', 'partial_training': False}
    if policy != 'paused_or_complete_latest':
        raise ValueError('Unknown source checkpoint policy')
    if summary.get('status') == 'partial':
        validate_test_checkpoint(checkpoint, summary, interim=True)
        return {'source_stage': 'interim', 'partial_training': True}
    if (summary.get('status') != 'complete' or summary.get('partial_dataset', True)
            or checkpoint['run_info'].get('partial_dataset', True)
            or summary.get('run_signature') != checkpoint['run_signature']):
        raise ValueError('Latest mapped test requires a safely paused or complete full-data training run')
    if any(checkpoint['state'].get(k) != summary['state'].get(k) for k in ('epoch', 'cursor', 'step')):
        raise ValueError('Completed latest checkpoint must match run_summary progress')
    return {'source_stage': 'completed_latest', 'partial_training': False}


def prepare_job(cfg, entry):
    out = Path(cfg['output_directory']) / entry['name']
    out.mkdir(parents=True, exist_ok=True)
    source = resolve(entry['directory']) / entry.get('checkpoint_filename', cfg['checkpoint_filename'])
    if out.resolve() == source.parent.resolve():
        raise ValueError('Mapping outputs must use a directory separate from the source training run')
    evaluation = load_config(entry['evaluation_config'])
    if cfg['model_name_or_path']:
        evaluation['model_name_or_path'] = cfg['model_name_or_path']
    semantic_eval = {k: v for k, v in evaluation.items() if k not in ('device', 'output_directory')}
    contract = {'format': FORMAT, 'name': entry['name'], 'source': str(source.resolve()),
                'evaluation_config': semantic_eval, 'mapping': semantic_settings(cfg)['mapping'],
                'split': cfg['evaluation']['split'], 'code': code_hashes(),
                'checkpoint_policy': entry.get('checkpoint_policy', 'validation_selected')}
    signature = fingerprint(contract)
    manifest = out / 'job.json'
    snapshot = out / 'source_soft.pt'
    if manifest.exists():
        saved = read_json(manifest)
        previous = saved['contract']
        if (saved['signature'] != fingerprint(previous)
                or {k: v for k, v in previous.items() if k != 'code'} != {k: v for k, v in contract.items() if k != 'code'}
                or not compatible_execution_code(previous['code'], contract['code'])
                or file_hash(snapshot) != saved['snapshot_sha256']):
            raise ValueError(f'{entry["name"]}: job identity changed; use a fresh output directory')
        if source.is_file() and file_hash(source) != saved['source_sha256']:
            raise ValueError(f'{entry["name"]}: source checkpoint changed after snapshotting')
        validate_prompt(read_checkpoint(snapshot))
        note_runtime_upgrade(out, previous['code'], contract['code'])
        return out, saved
    if not source.is_file():
        raise FileNotFoundError(f'Soft checkpoint not found: {source}')
    # The source run lock also prevents a training update of latest.pt while read.
    with run_lock(source.parent):
        checkpoint = read_checkpoint(source)
        prompt = validate_prompt(checkpoint)
        trained = checkpoint['run_info']['config']
        for key in ('prompt_length', 'public_text', 'separator', 'question_format', 'enable_thinking',
                    'scene', 'record_form', 'question_protocol', 'name_mode', 'birth_date_mode',
                    'name_format', 'split_private_prefill'):
            if trained.get(key) != evaluation.get(key):
                raise ValueError(f'{entry["name"]}: training/evaluation protocol mismatch: {key}')
        # Freeze only inference-relevant data, never duplicate optimizer states.
        compact = {k: checkpoint[k] for k in ('method', 'prompt', 'run_info', 'state',
                    'inference_signature', 'run_signature', 'selected_by_validation')}
        summary = source.parent / 'run_summary.json'
        compact['source_training_summary'] = read_json(summary) if summary.is_file() else None
        if cfg['evaluation']['enabled'] and cfg['evaluation']['split'] == 'test':
            validate_test_source(compact, contract['checkpoint_policy'])
        save_checkpoint(snapshot, compact)
        source_sha = file_hash(source)
    saved = {'signature': signature, 'contract': contract, 'source_sha256': source_sha,
             'snapshot_sha256': file_hash(snapshot), 'prompt_shape': list(prompt.shape)}
    write_json(manifest, saved)
    return out, saved


class EmbeddingSource:
    """Read just Qwen's input embedding, never instantiate its transformer."""
    key = 'model.embed_tokens.weight'

    def __init__(self, directory):
        from safetensors import safe_open
        self.directory = Path(directory)
        if not self.directory.is_dir():
            raise FileNotFoundError(f'A local model directory is required: {directory}')
        index = self.directory / 'model.safetensors.index.json'
        if index.is_file():
            name = read_json(index)['weight_map'][self.key]
            self.path = (self.directory / name).resolve()
            if not self.path.is_relative_to(self.directory.resolve()):
                raise ValueError('Embedding shard must stay inside the model directory')
        else:
            self.path = self.directory / 'model.safetensors'
        if not self.path.is_file():
            raise FileNotFoundError('Input embedding requires local model.safetensors or its shard index')
        with safe_open(str(self.path), framework='pt', device='cpu') as stream:
            self.shape = stream.get_slice(self.key).get_shape()
        if len(self.shape) != 2 or min(self.shape) < 1:
            raise ValueError('Invalid input embedding shape')

    def chunks(self, size):
        from safetensors import safe_open
        with safe_open(str(self.path), framework='pt', device='cpu') as stream:
            view = stream.get_slice(self.key)
            for start in range(0, self.shape[0], size):
                yield start, view[start:start + size]

    def identity(self):
        digest = hashlib.sha256()
        for _, tensor in self.chunks(8192):
            digest.update(str(tensor.dtype).encode())
            digest.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())
        return {'embedding_shape': self.shape, 'embedding_sha256': digest.hexdigest(),
                'model_config_sha256': file_hash(self.directory / 'config.json')}


def tokenizer_for(model, expected_sha=None):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    serialized = tokenizer.backend_tokenizer.to_str()
    canonical = fingerprint(json.loads(serialized))
    if expected_sha:
        actual = hashlib.sha256(serialized.encode()).hexdigest()
        if actual != expected_sha:
            candidates = (Path(model) / 'tokenizer.json', ROOT / 'tools/tokenizer/Qwen3-4B/tokenizer.json')
            if not any(p.is_file() and file_hash(p) == expected_sha
                       and json.loads(p.read_text(encoding='utf-8')) == json.loads(serialized)
                       for p in candidates):
                raise ValueError('Mapping tokenizer differs from the soft checkpoint tokenizer')
    return tokenizer, canonical


def permitted_tokens(tokenizer, rows, settings):
    special = set(tokenizer.all_special_ids) if settings['exclude_special_tokens'] else set()
    ids = sorted(set(tokenizer.get_vocab().values()) - special)
    ids = [v for v in ids if 0 <= v < rows]
    if not settings['allow_non_ascii']:
        ids = [v for v in ids if all(32 <= ord(c) <= 126 for c in tokenizer.decode(
            [v], skip_special_tokens=False, clean_up_tokenization_spaces=False))]
    if not ids:
        raise ValueError('No permitted vocabulary tokens')
    return torch.tensor(ids, dtype=torch.long)


@torch.no_grad()
def cosine_nearest(prompt, source, allowed, chunk_size, device='cpu', stop=lambda: False):
    """Exact top-1 over allowed rows, FP32 arithmetic, smallest ID wins ties."""
    values = prompt.detach().to(device=device, dtype=torch.float32)
    norms = values.norm(dim=1, keepdim=True)
    if not torch.isfinite(values).all() or not torch.isfinite(norms).all() or (norms == 0).any():
        raise ValueError('Cannot normalize a zero/non-finite soft vector')
    values = values / norms
    best = torch.full((len(values),), -float('inf'), device=device)
    best_ids = torch.full((len(values),), source.shape[0], device=device, dtype=torch.long)
    allowed_mask = torch.zeros(source.shape[0], dtype=torch.bool)
    allowed_mask[allowed] = True
    for start, block in source.chunks(chunk_size):
        if stop():
            return None
        local = torch.where(allowed_mask[start:start + len(block)])[0]
        if not len(local):
            continue
        vectors = block[local].to(device=device, dtype=torch.float32)
        norm = vectors.norm(dim=1, keepdim=True)
        valid = torch.isfinite(vectors).all(dim=1) & torch.isfinite(norm[:, 0]) & (norm[:, 0] > 0)
        vectors = vectors / norm.clamp_min(torch.finfo(torch.float32).tiny)
        scores = values @ vectors.T
        scores[:, ~valid] = -float('inf')
        score, column = scores.max(dim=1)
        token = local.to(device)[column] + start
        improve = (score > best) | ((score == best) & (token < best_ids))
        best = torch.where(improve, score, best)
        best_ids = torch.where(improve, token, best_ids)
    if not torch.isfinite(best).all():
        raise ValueError('No finite nonzero permitted embedding vectors')
    return best_ids.cpu().tolist(), best.cpu().tolist()


def stopped(root):
    return (Path(root) / 'pause.request').exists()


@contextlib.contextmanager
def pause_signals(root):
    old = {}
    def request(signum, frame):
        Path(root, 'pause.request').touch()
    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            old[sig] = signal.signal(sig, request)
        yield
    finally:
        for sig, handler in old.items():
            signal.signal(sig, handler)


def mapping_worker(job_dir, root, settings, rank=0, world=1, device='cpu'):
    torch.set_num_threads(settings.get('cpu_threads', 4))
    job_dir = Path(job_dir)
    job = read_json(job_dir / 'job.json')
    frozen = read_checkpoint(job_dir / 'source_soft.pt')
    prompt = validate_prompt(frozen)
    model = job['contract']['evaluation_config']['model_name_or_path']
    source = EmbeddingSource(model)
    if source.shape[1] != prompt.shape[1]:
        raise ValueError('Soft U/input embedding dimension mismatch')
    tokenizer, tokenizer_sha = tokenizer_for(model, frozen['run_info']['backend'].get('tokenizer_sha256'))
    identity = {**source.identity(), 'tokenizer_canonical_sha256': tokenizer_sha}
    if identity != read_json(job_dir / 'mapping_model.json'):
        raise ValueError('Mapping model/tokenizer changed')
    allowed = permitted_tokens(tokenizer, source.shape[0], settings)
    directory = job_dir / 'positions'
    directory.mkdir(exist_ok=True)
    # All workers use the SAME assignment snapshot. Re-reading pending files
    # independently would shift rank slices while another worker commits rows.
    pending = settings['pending_positions']
    assigned = pending[rank::world]
    if device.startswith('cuda'):
        torch.cuda.set_device(torch.device(device))
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    for offset in range(0, len(assigned), settings['position_chunk_size']):
        if stopped(root):
            return
        indices = assigned[offset:offset + settings['position_chunk_size']]
        result = cosine_nearest(prompt[indices], source, allowed, settings['vocabulary_chunk_size'],
                                device, stop=lambda: stopped(root))
        if result is None:
            return
        for position, token, score in zip(indices, *result):
            write_json(directory / f'{position:06d}.json', {
                'job_signature': job['signature'], 'position': position, 'token_id': token,
                'cosine_similarity': score})


def collect_positions(job_dir, job):
    directory = Path(job_dir) / 'positions'
    found = {}
    for path in sorted(directory.glob('*.json')):
        row = read_json(path)
        index, token, score = row['position'], row['token_id'], row['cosine_similarity']
        if (type(index) is not int or not 0 <= index < job['prompt_shape'][0]
                or path.name != f'{index:06d}.json' or row['job_signature'] != job['signature']
                or type(token) is not int or token < 0 or not math.isfinite(score)
                or not -1.00001 <= score <= 1.00001):
            raise ValueError(f'Invalid mapping progress: {path}')
        found[index] = row
    write_json(Path(job_dir) / 'mapping_progress.json', {
        'signature': job['signature'], 'completed_positions': len(found),
        'total_positions': job['prompt_shape'][0], 'status': 'complete' if len(found) == job['prompt_shape'][0] else 'partial'})
    return found


def finish_mapping(job_dir, job, tokenizer):
    job_dir = Path(job_dir)
    rows = collect_positions(job_dir, job)
    if len(rows) != job['prompt_shape'][0]:
        return None
    ordered = [rows[i] for i in range(len(rows))]
    ids = [row['token_id'] for row in ordered]
    text = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    reencoded = tokenizer.encode(text, add_special_tokens=False)
    stable = reencoded == ids
    artifact = {'format': FORMAT, 'name': job['contract']['name'], 'method': 'cosine_nearest',
        'job_signature': job['signature'], 'source_checkpoint_sha256': job['source_sha256'],
        'model': read_json(job_dir / 'mapping_model.json'), 'prompt_ids': ids, 'text': text,
        'cosine_similarities': [r['cosine_similarity'] for r in ordered],
        'decode_encode_verified': stable, 'reencoded_ids': reencoded,
        'uses_task_data': False, 'status': 'complete' if stable else 'invalid_text'}
    write_json(job_dir / 'mapped_prompt.json', artifact)
    atomic_text(job_dir / 'mapped_prompt.txt', text)
    atomic_csv(job_dir / 'token_mapping.csv', ['position', 'token_id', 'token_text', 'cosine_similarity'],
               [{'position': r['position'], 'token_id': r['token_id'],
                 'token_text': tokenizer.decode([r['token_id']], skip_special_tokens=False,
                        clean_up_tokenization_spaces=False), 'cosine_similarity': r['cosine_similarity']} for r in ordered])
    marker(job_dir / 'mapping_complete.json', [job_dir / n for n in
           ('mapped_prompt.json', 'mapped_prompt.txt', 'token_mapping.csv')], usable_text=stable)
    return artifact


def process_token(pid):
    token = process_start(pid)
    if token is not None:
        try:
            if (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()[0] == 'Z':
                return None
        except OSError:
            return None
        return token
    if os.name == 'nt':
        # Creation time protects against PID reuse in CPU implementation checks.
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return None
        try:
            stamps = [wintypes.FILETIME() for _ in range(4)]
            if kernel.GetProcessTimes(handle, *[ctypes.byref(s) for s in stamps]):
                return str((stamps[0].dwHighDateTime << 32) + stamps[0].dwLowDateTime)
        finally:
            kernel.CloseHandle(handle)
    return None


def alive(info):
    return (info.get('host', socket.gethostname()) == socket.gethostname()
            and bool(info.get('start_token')) and process_token(info['pid']) == info['start_token'])


def launch_workers(script, job_dir, root, settings, devices, stage):
    """Independent inference workers; no padding, DDP, or gradient collectives."""
    attempt = Path(job_dir) / stage / 'attempts' / f'{time.time_ns()}_{uuid.uuid4().hex[:8]}'
    attempt.mkdir(parents=True)
    print(json.dumps({'stage': stage, 'job': Path(job_dir).name,
        'gpus': devices, 'cpu': not bool(devices), 'attempt_logs': str(attempt)}, ensure_ascii=False), flush=True)
    options = attempt / 'settings.json'
    write_json(options, settings)
    children, streams = [], []
    try:
        for rank, gpu in enumerate(devices or [None]):
            if stopped(root):
                break
            command = [sys.executable, '-u', str(ROOT / script), '_worker', '--job-directory', str(job_dir),
                       '--run-root', str(root), '--worker-settings', str(options),
                       '--rank', str(rank), '--world', str(len(devices) or 1)]
            env = dict(os.environ, PYTHONUNBUFFERED='1')
            for key in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'):
                env.pop(key, None)
            env['CUDA_VISIBLE_DEVICES'] = '' if gpu is None else str(gpu)
            env['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
            command += ['--device', 'cpu' if gpu is None else 'cuda:0']
            stream = (attempt / f'worker_{rank}.log').open('x', encoding='utf-8')
            streams.append(stream)
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                     stdout=stream, stderr=subprocess.STDOUT,
                                     **({'creationflags': 0x08000000} if os.name == 'nt' else {}))
            children.append(child)
            write_json(Path(root) / 'workers.json', {'stage': stage, 'job': str(job_dir),
                'attempt': str(attempt), 'workers': [
                    {'pid': p.pid, 'start_token': process_token(p.pid), 'host': socket.gethostname()} for p in children]})
        while any(p.poll() is None for p in children):
            if any(p.poll() not in (None, 0) for p in children):
                Path(root, 'pause.request').touch()
            time.sleep(.2)
        if any(p.returncode for p in children):
            raise RuntimeError(f'{stage} worker failed; see {attempt}')
    finally:
        for p in children:
            if p.poll() is None:
                # Preserve durable progress even if the coordinator itself errors.
                Path(root, 'pause.request').touch()
                p.wait()
        for stream in streams:
            stream.close()


def mapping_stage(job_dir, job, cfg, resource_cache):
    job_dir = Path(job_dir)
    model = job['contract']['evaluation_config']['model_name_or_path']
    if model not in resource_cache:
        source = EmbeddingSource(model)
        tokenizer, canonical = tokenizer_for(model)
        resource_cache[model] = (source, tokenizer, {**source.identity(), 'tokenizer_canonical_sha256': canonical})
    _, tokenizer, identity = resource_cache[model]
    path = job_dir / 'mapping_model.json'
    if path.exists() and read_json(path) != identity:
        raise ValueError('Mapping model changed during resume')
    write_json(path, identity)
    if (job_dir / 'mapping_complete.json').exists():
        verify_marker(job_dir / 'mapping_complete.json')
        return read_json(job_dir / 'mapped_prompt.json')
    positions = collect_positions(job_dir, job)
    pending = [i for i in range(job['prompt_shape'][0]) if i not in positions]
    if pending:
        settings = dict(cfg['mapping'], pending_positions=pending)
        try:
            launch_workers('map_soft_u_cosine.py', job_dir, cfg['output_directory'], settings,
                           cfg['mapping']['gpus'], 'mapping')
        finally:
            collect_positions(job_dir, job)
    return finish_mapping(job_dir, job, tokenizer)


def status(root):
    root = Path(root)
    process = read_json(root / 'process.json') if (root / 'process.json').exists() else {}
    workers = read_json(root / 'workers.json').get('workers', []) if (root / 'workers.json').exists() else []
    jobs = []
    for path in sorted(root.glob('*/job.json')):
        out = path.parent
        mapping = read_json(out / 'mapping_progress.json') if (out / 'mapping_progress.json').exists() else {}
        evaluation = read_json(out / 'evaluation/progress.json') if (out / 'evaluation/progress.json').exists() else {}
        if mapping:
            mapping['completed_positions'] = len(list((out / 'positions').glob('*.json')))
        if evaluation:
            evaluation['completed_records'] = len(list((out / 'evaluation/records').glob('*.json')))
        jobs.append({'name': out.name, 'mapping': mapping, 'evaluation': evaluation})
    return {'running': alive(process) or any(alive(p) for p in workers),
            'pause_requested': stopped(root), 'jobs': jobs}


def pause(root, timeout):
    root = Path(root)
    if not (root / 'pipeline.json').exists():
        raise FileNotFoundError(f'No mapping pipeline at {root}')
    (root / 'pause.request').touch()
    deadline = time.monotonic() + timeout
    while status(root)['running'] and time.monotonic() < deadline:
        time.sleep(.25)
    result = status(root)
    result['status'] = 'stopping' if result['running'] else 'paused'
    return result


def write_overview(root, cfg):
    rows = []
    for entry in cfg['soft_prefixes']:
        out = Path(root) / entry['name']
        mapping = read_json(out / 'mapped_prompt.json') if (out / 'mapping_complete.json').exists() else {}
        summary = read_json(out / 'evaluation/summary.json') if (out / 'evaluation/complete.json').exists() else {}
        rows.append({'name': entry['name'], 'mapping_status': mapping.get('status', 'pending'),
            'evaluation_status': summary.get('status', 'pending'), 'records': summary.get('records', ''),
            'macro_em': summary.get('macro_em', ''), 'macro_nll': summary.get('macro_nll', ''),
            'prediction_csv': str(out / 'evaluation/predictions.csv') if summary else ''})
    atomic_csv(Path(root) / 'mapping_evaluation_summary.csv', list(rows[0]), rows)


def run_pipeline(cfg, command='run'):
    root = Path(cfg['output_directory'])
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / '.pipeline.lock'), timeout=0):
        if status(root)['running']:
            raise RuntimeError('An existing coordinator/worker is still active; pause it before resume')
        contract = semantic_settings(cfg)
        manifest = root / 'pipeline.json'
        if manifest.exists() and read_json(manifest) != contract:
            raise ValueError('Pipeline configuration changed; use a fresh output_directory')
        write_json(manifest, contract)
        write_json(root / 'resolved_settings.json', cfg)
        (root / 'pause.request').unlink(missing_ok=True)
        write_json(root / 'process.json', {'pid': os.getpid(), 'start_token': process_token(os.getpid()),
                                          'host': socket.gethostname()})
        failures, resources = [], {}
        try:
            with pause_signals(root):
                for entry in cfg['soft_prefixes']:
                    if stopped(root):
                        break
                    out = root / entry['name']
                    try:
                        out, job = prepare_job(cfg, entry)
                        print(json.dumps({'job': entry['name'], 'source_checkpoint': job['contract']['source'],
                                          'stage': 'mapping'}, ensure_ascii=False), flush=True)
                        if command == 'evaluate':
                            verify_marker(out / 'mapping_complete.json')
                            artifact = read_json(out / 'mapped_prompt.json')
                        else:
                            artifact = mapping_stage(out, job, cfg, resources)
                        if stopped(root):
                            break
                        if artifact is None:
                            raise RuntimeError('Mapping did not finish all positions')
                        if not artifact['decode_encode_verified']:
                            raise ValueError('Mapped tokens do not survive decode/encode; automatic evaluation blocked')
                        if command != 'map' and cfg['evaluation']['enabled']:
                            from evaluate_mapped_u import evaluation_stage
                            evaluation_stage(out, job, cfg)
                        print(json.dumps({'job': entry['name'], 'status': 'paused' if stopped(root) else 'complete'},
                                         ensure_ascii=False), flush=True)
                    except Exception as exc:
                        out.mkdir(parents=True, exist_ok=True)
                        write_json(out / f'failure_{time.time_ns()}.json', {'type': type(exc).__name__, 'error': str(exc)})
                        failures.append({'name': entry['name'], 'error': str(exc)})
                        print(json.dumps(failures[-1], ensure_ascii=False), flush=True)
                    finally:
                        write_overview(root, cfg)
            result = {'status': 'failed' if failures else 'paused' if stopped(root) else 'complete',
                      'failures': failures, 'output_directory': str(root)}
            write_json(root / 'pipeline_result.json', result)
            return result
        finally:
            write_json(root / 'process.json', {'pid': os.getpid(), 'start_token': None})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', nargs='?', default='run', choices=['run', 'resume', 'map', 'evaluate', 'pause', 'status', '_worker'])
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument('--mapping-gpus', help='cpu or physical GPU IDs, e.g. 0,2')
    parser.add_argument('--evaluation-gpus', help='cpu or physical GPU IDs, e.g. 1,3')
    parser.add_argument('--batch-records-per-gpu', type=int, help='C caches answered together on each evaluation GPU')
    parser.add_argument('--no-evaluate', action='store_true')
    parser.add_argument('--timeout', type=float, default=60)
    parser.add_argument('--job-directory')
    parser.add_argument('--run-root')
    parser.add_argument('--worker-settings')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world', type=int, default=1)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args(argv)
    if args.command == '_worker':
        with pause_signals(args.run_root):
            mapping_worker(args.job_directory, args.run_root, read_json(args.worker_settings),
                           args.rank, args.world, args.device)
        return 0
    cfg = load_settings(args.config)
    if args.mapping_gpus is not None:
        cfg['mapping']['gpus'] = gpu_ids(args.mapping_gpus)
        cfg['mapping']['device'] = 'cuda' if cfg['mapping']['gpus'] else 'cpu'
    if args.evaluation_gpus is not None:
        cfg['evaluation']['gpus'] = gpu_ids(args.evaluation_gpus)
    if args.batch_records_per_gpu is not None:
        cfg['evaluation']['batch_records_per_gpu'] = positive_int(args.batch_records_per_gpu, 'batch_records_per_gpu')
    if args.no_evaluate:
        cfg['evaluation']['enabled'] = False
    if args.command in ('status', 'pause'):
        result = status(cfg['output_directory']) if args.command == 'status' else pause(cfg['output_directory'], args.timeout)
    else:
        result = run_pipeline(cfg, args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 1 if result.get('status') == 'failed' else 0


if __name__ == '__main__':
    raise SystemExit(main())

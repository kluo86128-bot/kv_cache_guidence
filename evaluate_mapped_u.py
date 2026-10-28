"""C-only evaluation of cosine-mapped text, with topology-independent journals.

Usually invoked automatically by map_soft_u_cosine.py. Standalone example::

    python evaluate_mapped_u.py --config configs/soft_u_cosine_mapping.json \
        --evaluation-gpus 0,2

Workers never share a process group. Each complete record is committed by an
atomic rename; a new attempt repartitions only records without a valid commit.
"""
import argparse
import copy
import csv
import json
import math
import os
from pathlib import Path
import sys
import time
import uuid
from types import SimpleNamespace

import torch

from map_soft_u_cosine import (FORMAT, atomic_csv, atomic_text, code_hashes,
    launch_workers, marker, pause_signals, read_json, stopped, verify_marker, validate_test_source,
    compatible_execution_code, note_runtime_upgrade, positive_int)
from medical_kv.config import file_hash, fingerprint, resolve
from medical_kv.data import dataset_identity, load_split
from medical_kv.evaluation import evaluate_records, summarize, write_predictions_csv
from medical_kv.evaluation_resume import expected_fields
from medical_kv.io import checkpoint_compatibility, read_checkpoint, seed_all
from medical_kv.control import write_json
from medical_kv.tasks import (MULTISCENE_SCHEMA, questions_for, reconstruct_birthday,
                              reconstruct_name)


def validate_prediction(record, rows, cfg):
    """Validate one complete record before accepting it across GPU attempts."""
    expected = expected_fields(record, cfg)
    by_field = {r['field']: r for r in rows}
    if len(by_field) != len(rows) or set(by_field) != set(expected):
        raise ValueError(f'Incomplete or duplicate record predictions: {record["record_id"]}')
    for row in rows:
        if (row['record_id'] != record['record_id'] or not isinstance(row.get('prediction'), str)
                or not math.isfinite(row['nll']) or type(row['em']) is not bool
                or row['em'] != (row['prediction'].strip() == row['gold'].strip())):
            raise ValueError('Invalid saved prediction')
        if record.get('schema_version') == MULTISCENE_SCHEMA:
            for key in ('schema_version', 'scene', 'record_form', 'source', 'group_id'):
                if row.get(key) != record[key]:
                    raise ValueError(f'Saved record metadata changed: {key}')
            if (row.get('target_fields') != list(record['fields'])
                    or row.get('question_fields') != [q['field'] for q in questions_for(record, cfg, evaluation=True)]):
                raise ValueError('Saved target/question fields changed')
    for q in questions_for(record, cfg, evaluation=True):
        row = by_field[q['field']]
        if row['gold'] != q['answer'] or row['question'] != q['question']:
            raise ValueError('Saved question/gold changed')
    generated = [by_field[q['field']] for q in questions_for(record, cfg, evaluation=True)]
    for reconstructed in (reconstruct_name(record, generated), reconstruct_birthday(record, generated)):
        if reconstructed is not None:
            actual = by_field[reconstructed['field']]
            if any(actual[k] != reconstructed[k] for k in ('gold', 'prediction', 'em', 'nll')):
                raise ValueError('Saved reconstructed field changed')
    summarize(rows)
    return [by_field[field] for field in expected]


def committed_records(directory, records, cfg, signature):
    """Atomic record files, unlike torn per-answer journals, need no tail repair."""
    directory = Path(directory)
    complete = {}
    for path in sorted(directory.glob('*.json')):
        payload = read_json(path)
        index = payload['index']
        if (type(index) is not int or not 0 <= index < len(records)
                or path.name != f'{index:08d}.json' or payload.get('signature') != signature
                or payload.get('rows_sha256') != fingerprint(payload['rows'])):
            raise ValueError(f'Invalid committed evaluation record: {path}')
        complete[index] = validate_prediction(records[index], payload['rows'], cfg)
    return complete


def prepare_evaluation(job_dir, job, pipeline_cfg):
    job_dir = Path(job_dir)
    verify_marker(job_dir / 'mapping_complete.json')
    artifact = read_json(job_dir / 'mapped_prompt.json')
    if (not artifact['decode_encode_verified'] or artifact['job_signature'] != job['signature']
            or artifact.get('format') != FORMAT):
        raise ValueError('Evaluation requires a verified mapped text artifact')
    frozen = read_checkpoint(job_dir / 'source_soft.pt')
    cfg = copy.deepcopy(job['contract']['evaluation_config'])
    split = pipeline_cfg['evaluation']['split']
    identity = dataset_identity(cfg)
    if frozen['run_info']['dataset'] != identity:
        raise ValueError('Mapped evaluation must match the source checkpoint dataset and scene selection')
    source_info = {'source_stage': 'development_or_validation',
                   'partial_training': frozen.get('source_training_summary', {}).get('status') == 'partial'
                   if frozen.get('source_training_summary') else False}
    if split == 'test':
        source_info = validate_test_source(frozen, job['contract'].get('checkpoint_policy', 'validation_selected'))
    records = load_split(cfg, split)
    # Validate text boundaries for the entire evaluation scope BEFORE any forward.
    from map_soft_u_cosine import tokenizer_for
    tokenizer, canonical = tokenizer_for(cfg['model_name_or_path'], identity['tokenizer_sha256'])
    ids = artifact['prompt_ids']
    if (tokenizer.encode(artifact['text'], add_special_tokens=False) != ids
            or canonical != artifact['model']['tokenizer_canonical_sha256']):
        raise ValueError('Mapped text/tokenizer identity changed')
    validate_boundaries(tokenizer, records, cfg, ids, artifact['text'])
    metadata = {'format': FORMAT, 'condition': 'mapped_cosine', 'split': split,
        'evaluation_stage': 'mapped_cosine_interim' if source_info['source_stage'] == 'interim' else 'mapped_cosine',
        'partial_training': source_info['partial_training'], 'partial_evaluation': False,
        'source_stage': source_info['source_stage'],
        'source_checkpoint_policy': job['contract'].get('checkpoint_policy', 'validation_selected'),
        'mapping_name': artifact['name'], 'mapping_artifact_sha256': file_hash(job_dir / 'mapped_prompt.json'),
        'mapping_text_sha256': file_hash(job_dir / 'mapped_prompt.txt'),
        'source_snapshot_sha256': job['snapshot_sha256'], 'job_signature': job['signature'],
        'dataset': identity, 'resolved_config': cfg, 'code': code_hashes(),
        'source_backend': frozen['run_info']['backend'], 'evaluation_records': len(records),
        'source_checkpoint_selected_by_validation': frozen['selected_by_validation'],
        'record_ids': [r['record_id'] for r in records], 'epoch': frozen['state'].get('epoch'),
        'step': frozen['state'].get('step'), 'completed_epochs': frozen['state'].get('epoch'),
        'records_seen_in_current_epoch': frozen['state'].get('cursor'),
        'prompt_positions': len(ids), 'public_text': cfg['public_text'],
        'position_convention': 'Only C KV retained, with original C and question RoPE positions.',
        'generation': {'do_sample': False, 'max_new_tokens': cfg['max_new_tokens'],
                       'enable_thinking': cfg['enable_thinking']}}
    signature = fingerprint(metadata)
    directory = job_dir / 'evaluation'
    directory.mkdir(exist_ok=True)
    (directory / 'records').mkdir(exist_ok=True)
    path = directory / 'metadata.json'
    descriptor = {'signature': signature, 'metadata': metadata}
    if path.exists():
        previous = read_json(path)
        previous_metadata = previous['metadata']
        if (previous['signature'] != fingerprint(previous_metadata)
                or {k: v for k, v in previous_metadata.items() if k != 'code'} != {k: v for k, v in metadata.items() if k != 'code'}
                or not compatible_execution_code(previous_metadata['code'], metadata['code'])):
            raise ValueError('Evaluation resume identity changed; use a fresh output directory')
        note_runtime_upgrade(directory, previous_metadata['code'], metadata['code'])
        # All previously committed records retain this original signature.
        return directory, cfg, records, previous
    write_json(path, descriptor)
    return directory, cfg, records, descriptor


def validate_boundaries(tokenizer, records, cfg, ids, text):
    separator = cfg['separator']
    middle = tokenizer.encode(separator, add_special_tokens=False)
    public = tokenizer.encode(cfg['public_text'], add_special_tokens=False)
    for record in records:
        before = record['private_prefix'] + separator
        expected = tokenizer.encode(before, add_special_tokens=False) + ids + middle + public
        whole = tokenizer.encode(before + text + separator + cfg['public_text'], add_special_tokens=False)
        if whole != expected:
            raise ValueError(f'Mapped text changes context token boundaries: {record["record_id"]}')


def evaluation_worker(job_dir, root, settings, rank=0, world=1, device='cpu'):
    from medical_kv.backend import Backend
    torch.set_num_threads(settings.get('cpu_threads', 4))
    job_dir = Path(job_dir)
    descriptor = read_json(job_dir / 'evaluation/metadata.json')
    signature, metadata = descriptor['signature'], descriptor['metadata']
    if signature != fingerprint(metadata) or not compatible_execution_code(metadata['code'], code_hashes()):
        raise ValueError('Evaluation descriptor/code changed')
    job = read_json(job_dir / 'job.json')
    if (file_hash(job_dir / 'source_soft.pt') != metadata['source_snapshot_sha256']
            or file_hash(job_dir / 'mapped_prompt.json') != metadata['mapping_artifact_sha256']
            or job['signature'] != metadata['job_signature']):
        raise ValueError('Frozen evaluation artifact changed')
    cfg = copy.deepcopy(metadata['resolved_config'])
    cfg['device'] = device
    if dataset_identity(cfg) != metadata['dataset']:
        raise ValueError('Evaluation dataset changed')
    records = load_split(cfg, metadata['split'])
    if [r['record_id'] for r in records] != metadata['record_ids']:
        raise ValueError('Evaluation record order changed')
    snapshot = read_checkpoint(job_dir / 'source_soft.pt')
    ids = read_json(job_dir / 'mapped_prompt.json')['prompt_ids']
    pending = settings['pending_indices']
    assigned = pending[rank::world]
    if not assigned or stopped(root):
        return
    seed_all(cfg['seed'])
    backend = Backend.load(cfg, metadata['dataset']['tokenizer_sha256'])
    checkpoint_compatibility(snapshot, backend)
    seen = {}
    for record in load_split(cfg, 'train'):
        for field, value in record['fields'].items():
            seen.setdefault(field, set()).add(value)
    evaluate_worker_records(backend, records, assigned, ids, job_dir / 'evaluation/records',
                            signature, root, seen, batch_records_per_gpu=settings.get('batch_records_per_gpu', 1))


@torch.no_grad()
def evaluate_worker_records(backend, records, assigned, ids, directory, signature, root, seen=None,
                            batch_records_per_gpu=1):
    seen = seen or {}
    size = positive_int(batch_records_per_gpu, 'batch_records_per_gpu')
    execution_code = fingerprint(code_hashes())
    for offset in range(0, len(assigned), size):
        if stopped(root):
            return
        indices = assigned[offset:offset + size]
        if any((Path(directory) / f'{index:08d}.json').exists() for index in indices):
            raise ValueError('Worker assignment overlaps an existing committed record')
        start = time.monotonic()
        if size == 1:
            _, rows = evaluate_records(backend, [records[indices[0]]], prompt_ids=ids,
                        seen_birthdays=seen.get('birth_date', set()), seen_values=seen)
            results = [rows]
            prefill_seconds = None
        else:
            from mapped_cache_batch import score_contexts
            # Bounded queue: prefill individual records, retain ONLY their C KV.
            contexts = []
            for index in indices:
                if stopped(root):
                    return
                context = backend.context(records[index], prompt_ids=ids)
                if context.full_context or context.physical_length != len(backend.public_ids):
                    raise ValueError('Cache batch must contain only public C positions')
                contexts.append(context)
            if backend.device.type == 'cuda':
                torch.cuda.synchronize(backend.device)
            prefill_seconds = time.monotonic() - start
            questions = [questions_for(records[index], backend.cfg, evaluation=True) for index in indices]
            losses, answers = score_contexts(backend, contexts, questions)
            results = []
            for local, index in enumerate(indices):
                # Reuse existing row construction/reconstruction; no second forward.
                proxy = SimpleNamespace(cfg=backend.cfg, device=backend.device,
                    context=lambda *a, local=local, **kw: contexts[local],
                    answer_losses=lambda *a, local=local, **kw: torch.tensor(losses[local]),
                    generate_many=lambda *a, local=local, **kw: answers[local])
                _, rows = evaluate_records(proxy, [records[index]], prompt_ids=ids,
                            seen_birthdays=seen.get('birth_date', set()), seen_values=seen)
                results.append(rows)
            del contexts
        for index, rows in zip(indices, results):
            record = records[index]
            path = Path(directory) / f'{index:08d}.json'
            rows = validate_prediction(record, rows, backend.cfg)
            write_json(path, {'signature': signature, 'index': index,
                'rows_sha256': fingerprint(rows), 'rows': rows,
                'execution': {'batch_records_per_gpu': size, 'actual_batch_records': len(indices),
                              'code_fingerprint': execution_code}})
            print(json.dumps({'completed_record_index': index, 'record_id': record['record_id']}), flush=True)
        generated = [row for rows in results for row in rows if row['prediction_kind'] != 'reconstructed']
        print(json.dumps({'cache_batch_records': len(indices), 'configured_batch_records_per_gpu': size,
            'prefill_seconds': prefill_seconds, 'batch_seconds': time.monotonic() - start,
            'generated_tokens': sum(r['generated_tokens'] for r in generated),
            'truncated_answers': sum(r['truncated'] for r in generated)}, ensure_ascii=False), flush=True)


def save_predictions(directory, rows, metadata):
    """Keep existing CSV columns and add explicit mapping provenance."""
    directory = Path(directory)
    temporary = directory / f'predictions_base_{uuid.uuid4().hex}.tmp'
    try:
        write_predictions_csv(temporary, rows, metadata)
        with temporary.open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            columns = list(reader.fieldnames) + ['mapping_name', 'mapping_artifact_sha256']
            enriched = [{**row, 'mapping_name': metadata['mapping_name'],
                         'mapping_artifact_sha256': metadata['mapping_artifact_sha256']} for row in reader]
        atomic_csv(directory / 'predictions.csv', columns, enriched)
    finally:
        temporary.unlink(missing_ok=True)
    atomic_text(directory / 'predictions.jsonl', ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))


def finalize_evaluation(directory, cfg, records, descriptor):
    directory = Path(directory)
    complete = committed_records(directory / 'records', records, cfg, descriptor['signature'])
    progress = {'status': 'complete' if len(complete) == len(records) else 'partial',
                'completed_records': len(complete), 'total_records': len(records),
                'signature': descriptor['signature']}
    write_json(directory / 'progress.json', progress)
    if len(complete) != len(records):
        return None
    rows = [row for i in range(len(records)) for row in complete[i]]
    summary = summarize(rows)
    summary.update(status='complete', metadata=descriptor['metadata'],
                   evaluation_signature=descriptor['signature'],
                   execution_note='C caches may be decoded in batches; original committed records are reused unchanged.',
                   runtime_upgrades=[read_json(p) for p in sorted((directory / 'runtime_upgrades').glob('*.json'))],
                   execution_batches=sorted({read_json(p).get('execution', {}).get('batch_records_per_gpu', 1)
                                            for p in (directory / 'records').glob('*.json')}))
    save_predictions(directory, rows, descriptor['metadata'])
    write_json(directory / 'summary.json', summary)
    marker(directory / 'complete.json', [directory / n for n in
           ('predictions.csv', 'predictions.jsonl', 'summary.json')], signature=descriptor['signature'])
    return summary


def evaluation_stage(job_dir, job, pipeline_cfg):
    directory, cfg, records, descriptor = prepare_evaluation(job_dir, job, pipeline_cfg)
    done = directory / 'complete.json'
    if done.exists():
        if verify_marker(done)['signature'] != descriptor['signature']:
            raise ValueError('Completed evaluation signature changed')
        return read_json(directory / 'summary.json')
    complete = committed_records(directory / 'records', records, cfg, descriptor['signature'])
    pending = [i for i in range(len(records)) if i not in complete]
    write_json(directory / 'progress.json', {'status': 'partial', 'completed_records': len(complete),
                'total_records': len(records), 'signature': descriptor['signature']})
    if pending:
        launch_workers('evaluate_mapped_u.py', job_dir, pipeline_cfg['output_directory'],
                       {'pending_indices': pending, 'cpu_threads': pipeline_cfg['evaluation']['cpu_threads'],
                        'batch_records_per_gpu': pipeline_cfg['evaluation']['batch_records_per_gpu']},
                       pipeline_cfg['evaluation']['gpus'], 'evaluation')
    summary = finalize_evaluation(directory, cfg, records, descriptor)
    if summary is None and not stopped(pipeline_cfg['output_directory']):
        raise RuntimeError('Evaluation workers exited without completing their assignments')
    return summary


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] != '_worker':
        from map_soft_u_cosine import main as pipeline_main
        if arguments and arguments[0] in ('pause', 'status'):
            return pipeline_main(arguments)
        if arguments and arguments[0] in ('resume', 'run', 'evaluate'):
            arguments = arguments[1:]
        return pipeline_main(['evaluate', *arguments])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command')
    parser.add_argument('--job-directory', required=True)
    parser.add_argument('--run-root', required=True)
    parser.add_argument('--worker-settings', required=True)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world', type=int, default=1)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args(arguments)
    with pause_signals(args.run_root):
        evaluation_worker(args.job_directory, args.run_root, read_json(args.worker_settings),
                          args.rank, args.world, args.device)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

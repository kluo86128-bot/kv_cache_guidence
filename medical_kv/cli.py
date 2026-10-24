import argparse
import contextlib
import json
import time
import os
import sys
from pathlib import Path
import torch
from .backend import Backend
from .config import ROOT, file_hash, load_config, resolve
from .data import dataset_identity, load_split, training_data, available_splits
from .evaluation import evaluate_records, paired_compare
from .io import checkpoint_compatibility, code_identity, read_checkpoint, run_lock, seed_all, write_json
from .distributed import initialize, cleanup
from .tasks import questions_for


def tiny_backend(cfg):
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
    seed_all(cfg['seed'])
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / 'tools/tokenizer/Qwen3-4B'), local_files_only=True)
    model = Qwen3ForCausalLM(Qwen3Config(vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64,
               num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=1, head_dim=16,
               max_position_embeddings=4096, eos_token_id=tokenizer.eos_token_id))
    model.set_attn_implementation('eager')
    model.to(torch.device(cfg['device']))
    return Backend(model, tokenizer, cfg, identity={'model': 'random-tiny-Qwen3', 'seed': cfg['seed'],
                                                  'tokenizer_sha256': file_hash(ROOT/'tools/tokenizer/Qwen3-4B/tokenizer.json'),
                                                  'warning': 'Implementation checks only, not a trained model'})


def inspect_config(cfg):
    from transformers import AutoTokenizer
    identity = dataset_identity(cfg)
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / 'tools/tokenizer/Qwen3-4B'), local_files_only=True)
    records = load_split(cfg, 'dev')
    manual_ids = tokenizer.encode(cfg['manual_prompt'], add_special_tokens=False)
    if len(manual_ids) != cfg['prompt_length']:
        raise ValueError('Manual baseline does not match prompt_length')
    return {'dataset': identity, 'splits': {s: len(load_split(cfg, s)) for s in available_splits(cfg)},
            'public_tokens': len(tokenizer.encode(cfg['public_text'], add_special_tokens=False)),
            'manual_prompt_tokens': len(manual_ids), 'question_fields': [q['field'] for q in questions_for(records[0], cfg)],
            'birth_date_mode': cfg['birth_date_mode'], 'epochs': cfg['epochs'], 'early_stopping': cfg['early_stopping'],
            'name_mode': cfg['name_mode'],
            'question_protocol': cfg['question_protocol'],
            'question_batch_size': cfg['question_batch_size'],
            'evaluation_question_fields': [q['field'] for q in questions_for(records[0], cfg, evaluation=True)],
            'development_source': 'training records only' if cfg['question_protocol'] != 'legacy' else 'dev split',
            'comparison_epochs': cfg['comparison_epochs'],
            'name_format': cfg['name_format'],
            'diagnosis_contrastive': cfg['diagnosis_contrastive'],
            'model_weights_loaded': False, 'configured_model': cfg['model_name_or_path']}


def preflight(backend, cfg, records):
    ids = backend.initial_ids()
    prompt = backend.embedding(backend.tensor(ids))[0].detach().float().requires_grad_(True)
    context = backend.context(records[0], vectors=prompt)
    questions = questions_for(records[0], cfg)
    loss = backend.answer_loss(context, questions[0])
    loss.backward()
    if prompt.grad is None or not torch.isfinite(prompt.grad).all() or float(prompt.grad.norm()) == 0:
        raise AssertionError('Answer loss did not reach U through public KV')
    if any(p.grad is not None for p in backend.model.parameters()):
        raise AssertionError('Frozen model weight received gradients')
    with torch.no_grad():
        first = backend.answer_loss(context, questions[0]).item()
        backend.answer_loss(context, questions[1])
        repeated = backend.answer_loss(context, questions[0]).item()
        if abs(first - repeated) > 1e-5:
            raise AssertionError('Independent field queries mutated the source cache')
    blocked = backend.context(records[0], vectors=prompt, blocked=True)
    blocked_loss = backend.answer_loss(blocked, questions[0])
    if blocked_loss.requires_grad:
        raise AssertionError('Blocked C unexpectedly depends on U')
    return {'status': 'PASS', 'public_cache_length': context.physical_length, 'public_start': context.public_start,
            'question_absolute_start': context.next_position, 'prompt_gradient_norm': float(prompt.grad.norm()),
            'initial_answer_nll': float(loss.detach()), 'frozen_model_gradients': False,
            'query_cache_isolation': True, 'backend': backend.identity}


def validate_test_checkpoint(checkpoint, summary, *, interim=False):
    if summary['partial_dataset'] or summary['run_signature'] != checkpoint['run_signature']:
        raise ValueError('Test requires a matching full-data training run')
    if interim:
        if (summary['status'] != 'partial' or summary.get('stop_reason') not in ('paused', 'interrupted')
                or checkpoint['run_info'].get('partial_dataset', True)):
            raise ValueError('Interim test requires a safely paused full-data training run')
        if any(checkpoint['state'].get(k) != summary['state'].get(k) for k in ('epoch', 'cursor', 'step')):
            raise ValueError('Interim test requires the latest paused checkpoint, not an earlier best checkpoint')
    elif not checkpoint['selected_by_validation'] or summary['status'] != 'complete':
        raise ValueError('Formal test requires a validation-selected checkpoint from a complete full-data run')


def _main(argv=None):
    with contextlib.ExitStack() as resources:
        return _main_with_resources(argv, resources)


def _main_with_resources(argv, resources):
    parser = argparse.ArgumentParser(description='Shared-prompt medical public-KV experiment')
    parser.add_argument('command', choices=['inspect', 'preflight', 'prepare-negatives', 'train-soft', 'train-hard', 'evaluate', 'compare'])
    parser.add_argument('--config', help='Defaults to configs/experiment_v2_birth_parts.json')
    parser.add_argument('--model-name-or-path')
    parser.add_argument('--device')
    parser.add_argument('--gpus', help='Visible GPU IDs, e.g. 0,2; automatically launches one worker per GPU')
    parser.add_argument('--serial-questions', dest='batch_questions', action='store_const', const=False,
                        help='Compute field questions sequentially (lower decoder memory / comparison baseline)')
    parser.add_argument('--birth-date-mode', choices=['full', 'parts'])
    parser.add_argument('--name-mode', choices=['full', 'parts'])
    parser.add_argument('--scene', choices=['medical', 'finance', 'hr'], help='Optional dataset scene selector')
    parser.add_argument('--record-form', choices=['structured', 'narrative', 'dialogue', 'mixed'])
    parser.add_argument('--question-batch-size', type=int, help='Maximum independent questions per decoder forward')
    parser.add_argument('--early-stopping', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--epoch-validation', action=argparse.BooleanOptionalAction, default=None,
                        help='Validate each epoch (default/inherited); --no-epoch-validation keeps final-epoch validation only')
    parser.add_argument('--dtype', choices=['float32', 'float16', 'bfloat16'])
    parser.add_argument('--seed', type=int)
    parser.add_argument('--output-directory')
    parser.add_argument('--train-limit', type=int)
    parser.add_argument('--validation-limit', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--max-updates', type=int)
    parser.add_argument('--max-new-tokens', type=int)
    parser.add_argument('--development', action='store_true', help='Implementation checks: dev split, or disjoint groups from TRAINING for multiscene')
    parser.add_argument('--tiny', action='store_true', help='CPU random tiny model for implementation checks only')
    parser.add_argument('--resume')
    parser.add_argument('--checkpoint')
    parser.add_argument('--resume-evaluation', help='Recover a stopped interim test from its output directory on any GPU topology')
    parser.add_argument('--interim-test', action='store_true',
                        help='Evaluate the full test split at a safely paused latest checkpoint; preserve a snapshot')
    parser.add_argument('--allow-name-format-transfer', action='store_true',
                        help='Evaluate an old numbered-name soft checkpoint on natural names; retains model/cache checks')
    parser.add_argument('--condition', choices=['full', 'none', 'random', 'manual', 'soft', 'hard', 'blocked'], default='random')
    parser.add_argument('--split', choices=['dev', 'train', 'validation', 'test'], default='dev')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--a', help='Condition A predictions.jsonl')
    parser.add_argument('--b', help='Condition B predictions.jsonl')
    args = parser.parse_args(argv)
    resume_metadata = None
    if args.resume_evaluation:
        forbidden = ('config', 'checkpoint', 'model_name_or_path', 'device', 'dtype', 'seed',
                     'train_limit', 'validation_limit', 'epochs', 'max_updates', 'max_new_tokens',
                     'birth_date_mode', 'name_mode', 'early_stopping', 'epoch_validation',
                     'scene', 'record_form', 'question_batch_size')
        if (args.command != 'evaluate' or args.tiny or args.development or args.limit is not None
                or args.batch_questions is not None or args.allow_name_format_transfer
                or args.condition not in ('random', 'soft') or args.split not in ('dev', 'test')
                or any(getattr(args, key) is not None for key in forbidden)):
            parser.error('--resume-evaluation inherits its original checkpoint/config; only GPU/output changes are supported')
        from .evaluation_resume import resume_source
        resume_metadata, snapshot = resume_source(resolve(args.resume_evaluation))
        args.config = resume_metadata['resolved_config']
        args.checkpoint = str(snapshot)
        args.condition, args.split, args.interim_test = 'soft', 'test', True
    if args.epoch_validation is not None and args.command != 'train-soft':
        parser.error('--epoch-validation/--no-epoch-validation requires train-soft')
    if args.interim_test and (args.command != 'evaluate' or args.split != 'test'
            or args.condition != 'soft' or not args.checkpoint or args.limit is not None
            or args.tiny or args.development or args.record_form is not None):
        parser.error('--interim-test requires evaluate --split test --condition soft --checkpoint, without limit/form/tiny/development options; scene must match training')
    if args.allow_name_format_transfer and (args.command != 'evaluate' or not args.checkpoint or args.condition != 'soft'):
        parser.error('--allow-name-format-transfer requires evaluate --condition soft --checkpoint; it cannot resume training')
    overrides = {key: getattr(args, key) for key in ('model_name_or_path', 'device', 'dtype', 'seed', 'output_directory',
                 'train_limit', 'validation_limit', 'epochs', 'max_updates', 'max_new_tokens', 'batch_questions',
                 'birth_date_mode', 'name_mode', 'early_stopping', 'scene', 'record_form', 'question_batch_size')}
    if args.tiny:
        if args.command not in ('inspect', 'preflight') and not args.development:
            parser.error('--tiny training/evaluation requires --development')
        if args.split == 'test':
            parser.error('Tiny implementation checks cannot evaluate the formal test split')
        overrides.update({'device': 'cpu', 'dtype': 'float32'})
    cfg = load_config(args.config or 'configs/experiment_v2_birth_parts.json', overrides)
    if not cfg['hard']['enabled'] and (args.command == 'train-hard'
            or (args.command == 'evaluate' and args.condition == 'hard')):
        parser.error('Discrete text search is disabled (hard.enabled=false); use train-soft / --condition soft.')
    if args.gpus is not None:
        if args.command not in ('train-soft', 'evaluate') or args.tiny or args.device:
            parser.error('--gpus supports CUDA train-soft/evaluate; do not combine it with --tiny or --device')
        from .launch import launch
        try:
            launch(args.gpus, list(argv) if argv is not None else sys.argv[1:])
        except ValueError as exc:
            parser.error(str(exc))
        return
    if int(os.environ.get('WORLD_SIZE', '1')) > 1 and args.command not in ('train-soft', 'evaluate'):
        parser.error('Distributed execution supports train-soft and evaluate only')
    if args.command == 'compare':
        if not args.a or not args.b or not args.output_directory:
            parser.error('compare requires --a --b --output-directory')
        result = paired_compare(resolve(args.a), resolve(args.b), seed=cfg['seed'])
        out = resolve(args.output_directory)
        out.mkdir(parents=True, exist_ok=False)
        write_json(out / 'comparison.json', result)
        print(json.dumps(result, indent=2))
        return
    if args.command == 'inspect':
        print(json.dumps(inspect_config(cfg), ensure_ascii=False, indent=2))
        return
    if args.command == 'prepare-negatives':
        if not cfg['diagnosis_contrastive']['enabled']:
            parser.error('prepare-negatives requires a config with diagnosis_contrastive.enabled=true')
        from .diagnosis_contrastive import build_bank, persist_bank
        dataset_identity(cfg)
        train, _, _ = training_data(cfg, args.development)
        bank = build_bank(train, cfg)
        out = resolve(cfg['output_directory'])
        with run_lock(out):
            persist_bank(bank, out)
        print(json.dumps(dict(bank['metadata'], fingerprint=bank['fingerprint'], status='PASS'), indent=2))
        return
    # Hold the training-run lock before reading a checkpoint, through snapshotting
    # and evaluation. Training cannot resume and replace latest.pt underneath us.
    if args.command == 'evaluate' and args.checkpoint and int(os.environ.get('RANK', '0')) == 0:
        if resume_metadata:
            original_root = Path(resume_metadata.get('training_run_directory')
                                 or Path(resume_metadata['checkpoint']).parent)
            if original_root != resolve(args.checkpoint).parent:
                resources.enter_context(run_lock(original_root))
        resources.enter_context(run_lock(resolve(args.checkpoint).parent))
    checkpoint = read_checkpoint(resolve(args.checkpoint)) if args.command == 'evaluate' and args.checkpoint else None
    if checkpoint and args.config is None and checkpoint['run_info']['config'].get('question_protocol') == 'multiscene_privacy_v1':
        cfg = load_config(checkpoint['run_info']['config'], overrides)
    if checkpoint and args.config is None and args.birth_date_mode is None:
        # Old checkpoints keep their original four-question evaluation by default.
        # An explicit config/mode can still request a cross-protocol diagnostic.
        cfg['birth_date_mode'] = checkpoint['run_info']['config'].get('birth_date_mode', 'full')
    if checkpoint and args.config is None and args.name_mode is None:
        cfg['name_mode'] = checkpoint['run_info']['config'].get('name_mode', 'full')
    identity = dataset_identity(cfg)
    runtime = initialize(tiny=args.tiny)
    if runtime.active and not args.tiny and (args.device or cfg['device'] != 'cuda:0'):
        parser.error('Distributed device selection uses --gpus or CUDA_VISIBLE_DEVICES; keep config device=cuda:0')
    seed_all(cfg['seed'])
    if args.tiny:
        torch.set_num_threads(2)
        backend = tiny_backend(cfg)
    else:
        local_cfg = dict(cfg, device=str(runtime.device)) if runtime.active else cfg
        backend = Backend.load(local_cfg, identity['tokenizer_sha256'])
    if args.command == 'preflight':
        print(json.dumps(preflight(backend, cfg, load_split(cfg, 'dev', 1)), indent=2))
        return
    if args.command.startswith('train-'):
        train, validation, subsets = training_data(cfg, args.development)
        if args.command == 'train-soft':
            from .training import train_soft
            result = train_soft(backend, cfg, train, validation, identity, subsets,
                                resume=resolve(args.resume) if args.resume else None, epoch_validation=args.epoch_validation)
        else:
            from .discrete import train_hard
            result = train_hard(backend, cfg, train, validation, identity, subsets, resume=resolve(args.resume) if args.resume else None)
        if runtime.primary:
            print(json.dumps(result, indent=2))
        return
    records = load_split(cfg, args.split, args.limit)
    compatibility = checkpoint_compatibility(checkpoint, backend,
        allow_name_format_transfer=args.allow_name_format_transfer) if checkpoint else {}
    if compatibility.get('name_format_transfer') and runtime.primary:
        print(compatibility['evaluation_note'], flush=True)
    if args.condition in ('soft', 'hard') and (not checkpoint or checkpoint['method'] != args.condition):
        raise ValueError('The requested learned condition requires its matching checkpoint')
    if checkpoint and args.split == 'test':
        if resume_metadata:
            from .evaluation_resume import validate_snapshot
            validate_snapshot(checkpoint, resume_metadata)
        else:
            summary_path = resolve(args.checkpoint).parent / 'run_summary.json'
            summary = json.loads(summary_path.read_text(encoding='utf-8'))
            validate_test_checkpoint(checkpoint, summary, interim=args.interim_test)
        if args.interim_test and cfg.get('record_form') is not None:
            raise ValueError('Interim full test requires all record forms in the trained scene scope')
        if cfg.get('question_protocol') == 'multiscene_privacy_v1':
            trained = checkpoint['run_info']['dataset']
            if (trained['schema'] != identity['schema'] or trained['split_sha256'] != identity['split_sha256']
                    or trained.get('selection') != identity.get('selection')):
                raise ValueError('Multiscene test must match the checkpoint dataset and scene/form selection')
    ids = checkpoint['initial_ids'] if checkpoint else backend.initial_ids()
    vectors = None
    if args.condition == 'none':
        ids = []
    elif args.condition == 'manual':
        ids = backend.ids(cfg['manual_prompt'])
        if len(ids) != cfg['prompt_length']:
            raise ValueError('Manual baseline token length mismatch')
    elif args.condition in ('soft', 'hard', 'blocked', 'full') and checkpoint:
        if checkpoint['method'] == 'soft':
            vectors, ids = checkpoint['prompt'].to(backend.device), None
        else:
            ids = checkpoint['prompt_ids']
    out = resolve(args.output_directory) if args.output_directory else ROOT / 'runs/evaluations' / f'{args.condition}_{args.split}_{time.time_ns()}'
    out = Path(runtime.broadcast(str(out)))
    seen_values = {}
    for record in load_split(cfg, 'train'):
        for field, value in record['fields'].items():
            seen_values.setdefault(field, set()).add(value)
    birthdays = seen_values.get('birth_date', set())
    metadata = {'condition': args.condition, 'split': args.split, 'partial_evaluation': args.limit is not None,
                'evaluation_stage': 'interim' if args.interim_test else 'standard',
                'partial_training': args.interim_test,
                'evaluation_records': len(records),
                'evaluation_scope': {'scene': cfg.get('scene'), 'record_form': cfg.get('record_form')},
                'completed_epochs': checkpoint['state'].get('epoch') if checkpoint else None,
                'records_seen_in_current_epoch': checkpoint['state'].get('cursor') if checkpoint else None,
                'configured_training_epochs': checkpoint['run_info']['config'].get('epochs') if checkpoint else None,
                'epoch': checkpoint['state'].get('epoch') if checkpoint else None,
                'step': checkpoint['state'].get('step') if checkpoint else None,
                'tiny_implementation_check': args.tiny, 'dataset': identity, 'backend': backend.identity,
                **compatibility,
                'resolved_config': cfg,
                'parallelism': runtime.metadata(), 'batch_questions': cfg['batch_questions'],
                'generation': {'do_sample': False, 'max_new_tokens': cfg['max_new_tokens'], 'stop_token_ids': sorted(backend.stops)},
                'evaluation_code_sha256': code_identity(),
                'checkpoint': str(resolve(args.checkpoint)) if args.checkpoint else None,
                'checkpoint_run_signature': checkpoint['run_signature'] if checkpoint else None,
                'checkpoint_selected_by_validation': checkpoint['selected_by_validation'] if checkpoint else None,
                'allow_name_format_transfer': args.allow_name_format_transfer,
                'checkpoint_birth_date_mode': checkpoint['run_info']['config'].get('birth_date_mode', 'full') if checkpoint else None,
                'checkpoint_name_mode': checkpoint['run_info']['config'].get('name_mode', 'full') if checkpoint else None,
                'checkpoint_diagnosis_negatives': checkpoint['run_info'].get('diagnosis_negatives') if checkpoint else None,
                'public_tokens': len(backend.public_ids), 'prompt_positions': vectors.shape[0] if vectors is not None else len(ids),
                'position_convention': 'Retain original C RoPE positions; q starts at original encoder end; cache slots compacted.'}
    if args.interim_test:
        metadata.update(checkpoint_snapshot=str(out / 'checkpoint.pt'),
                        source_checkpoint_sha256=file_hash(resolve(args.checkpoint)),
                        training_run_directory=(resume_metadata.get('training_run_directory')
                            or str(Path(resume_metadata['checkpoint']).parent)) if resume_metadata
                            else str(resolve(args.checkpoint).parent))
    result, _ = evaluate_records(backend, records, vectors=vectors, prompt_ids=ids,
             full=args.condition == 'full', blocked=args.condition == 'blocked', output=out,
             metadata=metadata, seen_birthdays=birthdays,
             **({'checkpoint_source': resolve(args.checkpoint)} if args.interim_test else {}),
             **({'resume_evaluation': resolve(args.resume_evaluation)} if args.resume_evaluation else {}),
             **({'seen_values': seen_values} if cfg.get('question_protocol') == 'multiscene_privacy_v1' else {}))
    if runtime.primary:
        print(json.dumps(result, indent=2))


def main(argv=None):
    try:
        return _main(argv)
    finally:
        cleanup()


if __name__ == '__main__':
    main()

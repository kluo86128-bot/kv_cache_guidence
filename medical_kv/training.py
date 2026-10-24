import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path
import torch
from .config import fingerprint, resolve
from .evaluation import evaluate_records
from .plotting import LossCurve
from .distributed import Runtime
from .control import PauseController
from .resume import training_contract, validate_resume, restore_training_rng
from .io import (append_json, code_identity, elapsed, inference_identity, read_checkpoint,
                 restore_rng, rng_state, run_lock, save_checkpoint, seed_all, stop_at_boundary, write_json)


def better(metrics, best):
    return best is None or (metrics['macro_em'], -metrics['macro_nll']) > (best['macro_em'], -best['macro_nll'])


def signature(backend, cfg, dataset_identity, subsets, method):
    return fingerprint({'config': cfg, 'dataset': dataset_identity, 'subsets': subsets,
                        'backend': backend.identity, 'method': method, 'code': code_identity()})


def local_batch_backward(backend, prompt, records, indices, rank, world_size, *, negative_bank=None, diagnostics=None, epoch=0):
    local_indices = indices[rank::world_size]
    losses = []
    for index in local_indices:
        record = records[index]
        if negative_bank is None:
            loss = backend.record_loss(record, vectors=prompt)
        else:
            row = negative_bank[record['record_id']]
            if row['gold'] != record['fields']['diagnosis']:
                raise ValueError('Negative bank positive diagnosis differs from the training record')
            from .diagnosis_contrastive import select_negatives, KINDS
            selected = select_negatives(row, epoch=epoch)
            loss, components = backend.record_contrastive_loss(record,
                [n['answer'] for n in selected], vectors=prompt)
            if diagnostics is not None:
                values = {key: float(value) for key, value in components.items()}
                values.update({f'diagnosis_kind_{kind}_records': int(any(n['kind'] == kind for n in selected)) for kind in KINDS})
                diagnostics.append(values)
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite training loss')
        losses.append(float(loss.detach()))
        (loss / len(indices)).backward()
    if prompt.grad is None:
        if local_indices:
            raise FloatingPointError('Missing prompt gradient')
        prompt.grad = torch.zeros_like(prompt)
    return losses


def validation_destination(out, epoch, step, runtime):
    # A replay after interruption gets its own attempt; prior outputs remain intact.
    destination = None
    if runtime.primary:
        base = out / 'validation_predictions' / f'epoch_{epoch:03d}_step_{step:06d}'
        destination = base
        attempt = 2
        while destination.exists():
            destination = base.with_name(f'{base.name}_attempt_{attempt:03d}')
            attempt += 1
    return Path(runtime.broadcast(str(destination) if destination is not None else None))


def train_soft(backend, cfg, train, validation, dataset_identity, subsets, *, resume=None, epoch_validation=None):
    runtime = Runtime(backend.device)
    subsets = dict(subsets, parallel_world_size=runtime.world_size)
    training_values = {}
    for record in train:
        for field, value in record['fields'].items():
            training_values.setdefault(field, set()).add(value)
    out = resolve(cfg['output_directory'])
    negative_bank = negative_info = None
    if cfg.get('diagnosis_contrastive', {}).get('enabled', False):
        from .diagnosis_contrastive import build_bank, bank_index
        prepared = build_bank(train, cfg)
        if len(set(runtime.objects(prepared['fingerprint']))) != 1:
            raise ValueError('Diagnosis negative banks differ across ranks')
        negative_bank = bank_index(prepared)
        negative_info = {'file': 'diagnosis_negatives.json', 'fingerprint': prepared['fingerprint'],
                         'audit': prepared['metadata']}
        subsets = dict(subsets, diagnosis_negatives_sha256=prepared['fingerprint'])
    contract = training_contract(backend.identity, cfg, dataset_identity, subsets, code_identity())
    origin_identity = contract
    run_signature = fingerprint(origin_identity)
    control = PauseController(out)
    with (run_lock(out) if runtime.primary else nullcontext()), stop_at_boundary() as stop:
        runtime.barrier()
        if not resume and (out / 'latest.pt').exists():
            raise ValueError('Run already exists; use --resume latest.pt or a fresh output directory')
        if negative_bank is not None:
            from .diagnosis_contrastive import persist_bank
            if runtime.primary:
                persist_bank(prepared, out, resume=bool(resume))
                print(json.dumps({'diagnosis_negatives': {'records': len(train),
                    'negative_answers': prepared['metadata']['negative_answers'],
                    'negative_count_histogram': prepared['metadata']['negative_count_histogram'],
                    'selected_kinds_by_record': prepared['metadata']['selected_kinds_by_record'],
                    'all_four_kinds_covered': prepared['metadata']['all_four_kinds_covered'],
                    'shared_pool': prepared['metadata']['shared_pool'],
                    'selection': prepared['metadata']['selection'],
                    'test_overlap_policy': prepared['metadata']['test_overlap_policy'],
                    'fingerprint': prepared['fingerprint']}}), flush=True)
            runtime.barrier()
        seed_all(cfg['seed'])
        initial_ids = backend.initial_ids()
        for record in train + validation:
            backend.serialized_ids(record, initial_ids)
        initial = backend.embedding(backend.tensor(initial_ids))[0].detach().float()
        prompt = torch.nn.Parameter(initial.clone())
        optimizer = torch.optim.AdamW([prompt], lr=cfg['learning_rate'], weight_decay=0)
        updates_per_epoch = math.ceil(len(train) / cfg['batch_records'])
        total = updates_per_epoch * cfg['epochs']
        warmup = max(1, math.ceil(total * cfg['warmup_fraction'])) if cfg['warmup_fraction'] else 0
        def schedule(step):
            if step < warmup:
                return (step + 1) / warmup
            return .5 * (1 + math.cos(math.pi * min(1., (step - warmup) / max(1, total - warmup))))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
        state = {'epoch': 0, 'cursor': 0, 'step': 0, 'best': None, 'bad_epochs': 0, 'elapsed_seconds': 0.0,
                 'field_exposures': {key: 0 for key in sorted({f for r in train for f in r['fields']})},
                 'best_first_window': None, 'checkpoint_results': {}, 'last_validation': None}
        comparison_epochs = cfg.get('comparison_epochs')
        def early_stopped():
            return cfg.get('early_stopping', True) and state['bad_epochs'] >= cfg['patience']
        resume_history = []
        previous_epoch_validation = True
        if resume:
            checkpoint = read_checkpoint(resume)
            origin_identity, migrated = validate_resume(checkpoint, contract)
            run_signature = checkpoint['run_signature']
            prompt.data.copy_(checkpoint['prompt'].to(prompt.device))
            initial = checkpoint['initial_prompt'].to(prompt.device)
            initial_ids = checkpoint['initial_ids']
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            state = checkpoint['state']
            rng_policy = restore_training_rng(checkpoint, runtime, cfg['seed'])
            resume_history = list(checkpoint['run_info'].get('resume_history', []))
            previous_epoch_validation = checkpoint['run_info'].get('validation_policy', {}).get('every_epoch', True)
            resume_history.append({'checkpoint': str(resume), 'epoch': state['epoch'],
                'cursor': state['cursor'], 'step': state['step'], 'reviewed_code_migration': migrated,
                'from_parallelism': checkpoint['run_info']['parallelism'],
                'to_parallelism': runtime.metadata(), 'rng_policy': rng_policy,
                'numeric_note': 'GPU topology changes may change floating-point reduction order.'})
        epoch_validation = previous_epoch_validation if epoch_validation is None else epoch_validation
        if not epoch_validation and cfg.get('early_stopping', True):
            raise ValueError('Disabling epoch validation requires early_stopping=false')
        validation_policy = {'every_epoch': epoch_validation, 'final_epoch': True,
                             'selection_scope': 'Only checkpoints actually evaluated on validation can be selected as best.'}
        if resume_history:
            resume_history[-1].update(from_epoch_validation=previous_epoch_validation,
                                      to_epoch_validation=epoch_validation)
        run_info = {'method': 'soft', 'run_signature': run_signature, 'config': cfg,
                    'validation_policy': validation_policy,
                    'dataset': dataset_identity, 'subsets': subsets, 'backend': backend.identity,
                    'code_sha256': code_identity(),
                    'origin_identity': origin_identity, 'resume_contract': contract,
                    'resume_history': resume_history,
                    'inference_signature': inference_identity(backend), 'initial_ids': initial_ids,
                    'parallelism': runtime.metadata(),
                    'loss_definition': 'Mean answer-token CE including EOS; mean name/birthday parts within each group if split; mean available original fields per record; mean global records.' +
                        (' Add weight/4 * mean_over_available_kinds(relu(margin + diagnosis_positive_NLL - negative_NLL)); negatives include EOS; no category padding.'
                         if negative_bank is not None else ''),
                    'diagnosis_negatives': negative_info,
                    'learning_rate_schedule': {'type': 'linear_warmup_cosine', 'total_updates': total,
                                               'warmup_updates': warmup},
                    'validation_predictions_directory': 'validation_predictions',
                    'partial_dataset': bool(cfg['train_limit'] or cfg['validation_limit'] or subsets['development_only'])}
        curve = None
        if runtime.primary:
            write_json(out / 'run.json', run_info)
            print(json.dumps({'validation_policy': validation_policy}), flush=True)
            control.begin(resume=bool(resume), step=state['step'], world_size=runtime.world_size)
            curve = LossCurve(out, max_step=state['step'], max_epoch=state['epoch'],
                              implementation_only=backend.identity.get('model') == 'random-tiny-Qwen3')
        started, previous_elapsed = time.monotonic(), state['elapsed_seconds']
        def checkpoint(selected=False):
            state['elapsed_seconds'] = runtime.maximum(previous_elapsed + elapsed(started))
            rank_rng = runtime.objects(rng_state())
            return {'method': 'soft', 'resume_format': 2, 'run_signature': run_signature, 'inference_signature': inference_identity(backend),
                    'prompt': prompt.detach().cpu(), 'initial_prompt': initial.detach().cpu(), 'initial_ids': initial_ids,
                    'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'state': dict(state),
                    'rng': rank_rng[0], 'rng_by_rank': rank_rng, 'selected_by_validation': selected, 'run_info': run_info}
        def persist(filename, selected=False):
            saved = checkpoint(selected)
            saved['checkpoint_role'] = filename.removesuffix('.pt')
            if runtime.primary:
                save_checkpoint(out / filename, saved)
        def result_entry(filename, metrics=None, validation_out=None):
            entry = {'checkpoint': str(out / filename), 'epoch': state['epoch'], 'step': state['step']}
            if metrics is not None:
                entry.update({'metrics': metrics,
                              'predictions_csv': str(validation_out / 'predictions.csv'),
                              'predictions_jsonl': str(validation_out / 'predictions.jsonl')})
            return entry
        def results_manifest(final=False):
            entries = dict(state['checkpoint_results'])
            latest = result_entry('latest.pt')
            if state['last_validation'] and state['last_validation']['step'] == state['step']:
                latest.update(state['last_validation'])
                latest['checkpoint'] = str(out / 'latest.pt')
            entries['latest'] = latest
            if final:
                entries['final'] = dict(latest, checkpoint=str(out / 'final.pt'))
            return {'comparison_epochs': comparison_epochs, 'checkpoints': entries,
                    'comparison_window_complete': bool(comparison_epochs and state['epoch'] >= comparison_epochs),
                    'birth_date_mode': cfg.get('birth_date_mode', 'full'), 'name_mode': cfg.get('name_mode', 'full')}
        def requested_stop_reason():
            if runtime.any(runtime.primary and control.requested()):
                return 'paused'
            if runtime.any(stop['requested']):
                return 'interrupted'
            return None
        reason = 'early_stopping' if early_stopped() else 'epochs_completed'
        try:
            while state['epoch'] < cfg['epochs'] and not early_stopped():
                order = list(range(len(train)))
                random.Random(f"{cfg['seed']}|epoch|{state['epoch']}").shuffle(order)
                while state['cursor'] < len(train):
                    requested = requested_stop_reason()
                    if requested or state['elapsed_seconds'] >= cfg['max_runtime_seconds']:
                        reason = requested or 'time_limit'
                        break
                    if cfg['max_updates'] and state['step'] >= cfg['max_updates']:
                        reason = 'max_updates'
                        break
                    indices = order[state['cursor']:state['cursor'] + cfg['batch_records']]
                    optimizer.zero_grad(set_to_none=True)
                    diagnostics = []
                    losses = local_batch_backward(backend, prompt, train, indices, runtime.rank, runtime.world_size,
                                                  negative_bank=negative_bank, diagnostics=diagnostics, epoch=state['epoch'])
                    # SUM of locally accumulated gradients, already divided by
                    # the actual global batch size (including short last batches).
                    runtime.sum_(prompt.grad)
                    if prompt.grad is None or not torch.isfinite(prompt.grad).all():
                        raise FloatingPointError('Missing or non-finite prompt gradient')
                    norm = float(torch.nn.utils.clip_grad_norm_([prompt], cfg['grad_clip']))
                    if any(p.grad is not None for p in backend.model.parameters()):
                        raise AssertionError('Frozen model received parameter gradients')
                    optimizer.step()
                    scheduler.step()
                    state['step'] += 1
                    state['cursor'] += len(indices)
                    for index in indices:
                        for field in train[index]['fields']:
                            state['field_exposures'][field] += 1
                    total_loss = runtime.sum_(torch.tensor(sum(losses), device=backend.device, dtype=torch.float64)).item()
                    peak_memory = runtime.maximum(torch.cuda.max_memory_allocated(backend.device) if backend.device.type == 'cuda' else 0)
                    log = {'step': state['step'], 'epoch': state['epoch'] + 1, 'records_seen_in_epoch': state['cursor'],
                           'loss': total_loss / len(indices), 'gradient_norm': norm,
                           'world_size': runtime.world_size, 'global_batch_records': len(indices),
                           'records_per_rank': [len(indices[r::runtime.world_size]) for r in range(runtime.world_size)],
                           'prompt_norm': float(prompt.detach().norm()), 'prompt_change_norm': float((prompt.detach() - initial).norm()),
                           'learning_rate_next_step': scheduler.get_last_lr()[0],
                           'elapsed_seconds': previous_elapsed + elapsed(started),
                           'cuda_peak_allocated_bytes': peak_memory}
                    if negative_bank is not None:
                        keys = ('ce_loss', 'diagnosis_ce_loss', 'diagnosis_contrastive_loss',
                                'diagnosis_negative_nll', 'diagnosis_ranking_accuracy', 'diagnosis_margin_accuracy',
                                'diagnosis_negatives_per_record')
                        sums = torch.tensor([sum(row[k] for row in diagnostics) for k in keys],
                                            device=backend.device, dtype=torch.float64)
                        values = (runtime.sum_(sums) / len(indices)).tolist()
                        log.update(dict(zip(keys, values)))
                        log.update(diagnosis_contrastive_weight=cfg['diagnosis_contrastive']['weight'],
                                   diagnosis_contrastive_margin=cfg['diagnosis_contrastive']['margin'],
                                   diagnosis_max_negatives_per_record=cfg['diagnosis_contrastive']['max_negatives_per_record'],
                                   diagnosis_negative_pairs_scored=int(round(log['diagnosis_negatives_per_record'] * len(indices))))
                        from .diagnosis_contrastive import KINDS
                        counts = torch.tensor([sum(row[f'diagnosis_kind_{kind}_records'] for row in diagnostics)
                                               for kind in KINDS], device=backend.device, dtype=torch.int64)
                        log['diagnosis_negative_kind_counts'] = dict(zip(KINDS, runtime.sum_(counts).tolist()))
                        log['diagnosis_negative_sampling_epoch'] = state['epoch'] + 1
                    if runtime.primary:
                        append_json(out / 'training.jsonl', log)
                        print(json.dumps(log), flush=True)
                    persist('latest.pt')
                    if curve is not None:
                        curve.update_training(log)
                requested = requested_stop_reason()
                if requested:
                    reason = requested
                    break
                if state['cursor'] < len(train):
                    break
                if not epoch_validation and state['epoch'] + 1 < cfg['epochs']:
                    state['epoch'] += 1
                    state['cursor'] = 0
                    # Preserve the requested comparison-window endpoint even
                    # when it has no validation predictions or selection score.
                    if comparison_epochs and state['epoch'] == comparison_epochs:
                        name = f'last_first{comparison_epochs}'
                        state['checkpoint_results'][name] = result_entry(name + '.pt')
                        persist(name + '.pt')
                    persist('latest.pt')
                    if runtime.primary:
                        write_json(out / 'checkpoint_results.json', results_manifest())
                        print(json.dumps({'validation_skipped': {'epoch': state['epoch'], 'step': state['step'],
                            'reason': 'epoch_validation_disabled', 'final_validation_epoch': cfg['epochs']}}), flush=True)
                    continue
                validation_out = validation_destination(out, state['epoch'] + 1, state['step'], runtime)
                validation_metadata = {'condition': 'soft',
                    'split': 'development_validation' if subsets['development_only'] else 'validation',
                    'epoch': state['epoch'] + 1, 'step': state['step'], 'run_signature': run_signature,
                    'inference_signature': inference_identity(backend), 'resolved_config': cfg,
                    'dataset': dataset_identity, 'backend': backend.identity, 'parallelism': runtime.metadata(),
                    'generation': {'do_sample': False, 'max_new_tokens': cfg['max_new_tokens'],
                                   'stop_token_ids': sorted(backend.stops)},
                    'evaluation_code_sha256': run_info['code_sha256'],
                    'training_diagnosis_negatives': negative_info,
                    'birthday_reference': 'actual training subset'}
                metrics, _ = evaluate_records(backend, validation, vectors=prompt,
                    output=validation_out, metadata=validation_metadata,
                    seen_birthdays={record['fields']['birth_date'] for record in train if 'birth_date' in record['fields']},
                    **({'seen_values': training_values} if cfg.get('question_protocol') == 'multiscene_privacy_v1' else {}))
                validation_epoch = state['epoch'] + 1
                macro_improved = state['best'] is None or metrics['macro_em'] > state['best']['macro_em']
                improved = better(metrics, state['best'])
                if improved:
                    state['best'] = {'macro_em': metrics['macro_em'], 'macro_nll': metrics['macro_nll'],
                                     'step': state['step'], 'epoch': validation_epoch}
                window_improved = bool(comparison_epochs and validation_epoch <= comparison_epochs
                                       and better(metrics, state['best_first_window']))
                if window_improved:
                    state['best_first_window'] = {'macro_em': metrics['macro_em'], 'macro_nll': metrics['macro_nll'],
                                                 'step': state['step'], 'epoch': validation_epoch}
                state['bad_epochs'] = 0 if macro_improved else state['bad_epochs'] + 1
                validation_log = {'epoch': state['epoch'] + 1, 'step': state['step'], 'metrics': metrics,
                    'selected': improved, 'selected_first_window': window_improved,
                    'predictions_csv': (validation_out / 'predictions.csv').relative_to(out).as_posix(),
                    'predictions_jsonl': (validation_out / 'predictions.jsonl').relative_to(out).as_posix()}
                if runtime.primary:
                    append_json(out / 'validation.jsonl', validation_log)
                    print(json.dumps({'validation': validation_log}), flush=True)
                state['epoch'] += 1
                state['cursor'] = 0
                state['last_validation'] = result_entry('latest.pt', metrics, validation_out)
                if improved:
                    state['checkpoint_results']['best'] = result_entry('best.pt', metrics, validation_out)
                if window_improved:
                    name = f'best_first{comparison_epochs}'
                    state['checkpoint_results'][name] = result_entry(name + '.pt', metrics, validation_out)
                if comparison_epochs and state['epoch'] == comparison_epochs:
                    name = f'last_first{comparison_epochs}'
                    state['checkpoint_results'][name] = result_entry(name + '.pt', metrics, validation_out)
                if improved:
                    persist('best.pt', selected=True)
                if window_improved:
                    persist(f'best_first{comparison_epochs}.pt', selected=True)
                if comparison_epochs and state['epoch'] == comparison_epochs:
                    persist(f'last_first{comparison_epochs}.pt')
                persist('latest.pt')
                if runtime.primary:
                    write_json(out / 'checkpoint_results.json', results_manifest())
                if curve is not None:
                    curve.update_validation(validation_log)
                if early_stopped():
                    reason = 'early_stopping'
                    break
        except Exception as exc:
            write_json(out / f'failure.rank{runtime.rank}.json', {'type': type(exc).__name__, 'error': str(exc),
                       'last_completed_step': state['step'], 'resume': 'latest.pt contains the last completed optimizer step'})
            if runtime.primary:
                write_json(out / 'failure.json', {'type': type(exc).__name__, 'error': str(exc), 'rank': runtime.rank})
                control.finish('failed', state, reason=type(exc).__name__)
            raise
        reason = requested_stop_reason() or reason
        persist('latest.pt')
        if curve is not None:
            curve.save()
        complete = reason in ('epochs_completed', 'early_stopping')
        if complete:
            persist('final.pt')
        artifacts = results_manifest(final=complete)
        summary = {'method': 'soft', 'run_signature': run_signature, 'status': 'complete' if complete else 'partial',
                   'stop_reason': reason, 'state': state, 'partial_dataset': run_info['partial_dataset'],
                   'validation_policy': validation_policy,
                   'parallelism': runtime.metadata(),
                   'validation_predictions_directory': str(out / 'validation_predictions'),
                   'best_checkpoint': str(out / 'best.pt') if state['best'] else None,
                   'checkpoint_results': artifacts}
        if runtime.primary:
            write_json(out / 'checkpoint_results.json', artifacts)
            write_json(out / 'run_summary.json', summary)
            control.finish('paused' if reason in ('paused', 'interrupted') else summary['status'], state, reason=reason)
            if reason in ('paused', 'interrupted'):
                print(json.dumps({'training_paused': {'step': state['step'], 'epoch': state['epoch'],
                    'cursor': state['cursor'], 'checkpoint': str(out / 'latest.pt')}}), flush=True)
        runtime.barrier()
        return summary

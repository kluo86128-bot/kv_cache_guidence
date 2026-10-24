"""Gradient proposals, followed by actual loss comparison on a shared train batch."""
import json
import random
import time
import torch
from .config import resolve
from .evaluation import evaluate_records
from .io import (append_json, code_identity, elapsed, inference_identity, read_checkpoint, restore_rng, rng_state,
                 run_lock, save_checkpoint, seed_all, stop_at_boundary, write_json)
from .training import better, signature


def allowed_ids(backend, ascii_only=True):
    specials = set(backend.tokenizer.all_special_ids)
    result = []
    for token in sorted(set(backend.tokenizer.get_vocab().values())):
        if token in specials or token >= backend.embedding.num_embeddings:
            continue
        text = backend.tokenizer.decode([token], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if not text.strip() or any(c in text for c in '\n\r\t'):
            continue
        if ascii_only and any(not 32 <= ord(c) <= 126 for c in text):
            continue
        if backend.ids(text) == [token]:
            result.append(token)
    if not result:
        raise ValueError('No stable ordinary candidate tokens')
    return result


def proposals(backend, current, gradient, allowed, records, settings, rng):
    with torch.no_grad():
        # Scores only propose replacements. Candidate acceptance uses true NLL.
        scores = gradient.to(backend.dtype) @ backend.embedding.weight.detach().T
        mask = torch.ones(scores.shape[1], dtype=torch.bool, device=scores.device)
        mask[torch.tensor(allowed, device=scores.device)] = False
        scores[:, mask] = float('inf')
        top = scores.topk(min(settings['top_k'], len(allowed)), largest=False).indices.cpu().tolist()
    candidates, seen = [list(current)], {tuple(current)}
    attempts = 0
    while len(candidates) <= settings['candidate_count'] and attempts < settings['candidate_count'] * 100:
        attempts += 1
        position = rng.randrange(len(current))
        replacement = rng.choice(top[position])
        candidate = list(current)
        candidate[position] = replacement
        if tuple(candidate) in seen:
            continue
        seen.add(tuple(candidate))
        try:
            for record in records:
                backend.serialized_ids(record, candidate)
        except ValueError:
            continue
        candidates.append(candidate)
    return candidates


def train_hard(backend, cfg, train, validation, dataset_identity, subsets, *, resume=None):
    if not cfg['hard'].get('enabled', False):
        raise ValueError('Discrete text search is disabled (hard.enabled=false)')
    out = resolve(cfg['output_directory'])
    settings = cfg['hard']
    run_signature = signature(backend, cfg, dataset_identity, subsets, 'hard')
    with run_lock(out), stop_at_boundary() as stop:
        if not resume and (out / 'latest.pt').exists():
            raise ValueError('Run already exists; choose a fresh directory or resume')
        seed_all(cfg['seed'])
        current = backend.initial_ids()
        initial = list(current)
        rng = random.Random(cfg['seed'])
        state = {'step': 0, 'best': None, 'bad_validations': 0, 'elapsed_seconds': 0.0,
                 'gradient_record_exposures': 0, 'candidate_record_evaluations': 0}
        if resume:
            saved = read_checkpoint(resume)
            if saved['method'] != 'hard' or saved['run_signature'] != run_signature:
                raise ValueError('Resume identity mismatch')
            current, initial, state = saved['prompt_ids'], saved['initial_ids'], saved['state']
            rng.setstate(saved['search_rng'])
            restore_rng(saved['rng'])
        for record in train + validation:
            backend.serialized_ids(record, current)
        allowed = allowed_ids(backend, settings['ascii_only'])
        run_info = {'method': 'hard', 'run_signature': run_signature, 'config': cfg,
                    'dataset': dataset_identity, 'subsets': subsets, 'backend': backend.identity,
                    'code_sha256': code_identity(),
                    'inference_signature': inference_identity(backend), 'initial_ids': initial,
                    'partial_dataset': bool(cfg['train_limit'] or cfg['validation_limit'] or subsets['development_only']),
                    'allowed_token_count': len(allowed)}
        write_json(out / 'run.json', run_info)
        started, previous = time.monotonic(), state['elapsed_seconds']
        def checkpoint(selected=False):
            state['elapsed_seconds'] = previous + elapsed(started)
            return {'method': 'hard', 'run_signature': run_signature, 'inference_signature': inference_identity(backend),
                    'prompt_ids': list(current), 'prompt_text': backend.tokenizer.decode(current, skip_special_tokens=False, clean_up_tokenization_spaces=False),
                    'initial_ids': initial, 'state': dict(state), 'search_rng': rng.getstate(), 'rng': rng_state(),
                    'selected_by_validation': selected, 'run_info': run_info}
        reason = 'early_stopping' if state['bad_validations'] >= cfg['patience'] else 'steps_completed'
        try:
            while state['step'] < settings['steps'] and state['bad_validations'] < cfg['patience']:
                if stop['requested'] or state['elapsed_seconds'] >= cfg['max_runtime_seconds']:
                    reason = 'interrupted' if stop['requested'] else 'time_limit'
                    break
                if cfg['max_updates'] and state['step'] >= cfg['max_updates']:
                    reason = 'max_updates'
                    break
                gradient_batch = rng.sample(train, min(settings['gradient_records'], len(train)))
                gradient_ids = {r['record_id'] for r in gradient_batch}
                candidate_pool = [r for r in train if r['record_id'] not in gradient_ids]
                if not candidate_pool:
                    candidate_pool = train  # Explicitly a tiny-development fallback.
                candidate_batch = rng.sample(candidate_pool, min(settings['candidate_records'], len(candidate_pool)))
                vectors = backend.embedding(backend.tensor(current))[0].detach().float().requires_grad_(True)
                grad_loss = 0.
                for record in gradient_batch:
                    loss = backend.record_loss(record, vectors=vectors)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Non-finite discrete gradient loss')
                    grad_loss += float(loss.detach()) / len(gradient_batch)
                    (loss / len(gradient_batch)).backward()
                if vectors.grad is None or not torch.isfinite(vectors.grad).all():
                    raise FloatingPointError('Invalid discrete proposal gradient')
                candidates = proposals(backend, current, vectors.grad, allowed,
                                       gradient_batch + candidate_batch, settings, rng)
                losses = []
                with torch.no_grad():
                    for candidate in candidates:
                        value = sum(float(backend.record_loss(r, prompt_ids=candidate)) for r in candidate_batch) / len(candidate_batch)
                        if not __import__('math').isfinite(value):
                            raise FloatingPointError('Non-finite candidate loss')
                        losses.append(value)
                winner = min(range(len(losses)), key=lambda i: (losses[i], i))
                current = candidates[winner]
                state['step'] += 1
                state['gradient_record_exposures'] += len(gradient_batch)
                state['candidate_record_evaluations'] += len(candidate_batch) * len(candidates)
                log = {'step': state['step'], 'gradient_loss': grad_loss, 'candidate_loss_before': losses[0],
                       'candidate_loss_after': losses[winner], 'accepted_replacement': winner != 0,
                       'evaluated_candidates': len(candidates),
                       'gradient_candidate_patient_overlap': len(gradient_ids & {r['record_id'] for r in candidate_batch}),
                       'elapsed_seconds': previous + elapsed(started),
                       'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(backend.device) if backend.device.type == 'cuda' else 0}
                append_json(out / 'training.jsonl', log)
                print(json.dumps(log), flush=True)
                # Validate before committing this search step to disk; an interrupted
                # evaluation can safely resume from the preceding completed step.
                if state['step'] % settings['validate_every'] == 0 or state['step'] == settings['steps']:
                    for record in train + validation:
                        backend.serialized_ids(record, current)
                    metrics, _ = evaluate_records(backend, validation, prompt_ids=current)
                    macro_improved = state['best'] is None or metrics['macro_em'] > state['best']['macro_em']
                    improved = better(metrics, state['best'])
                    if improved:
                        state['best'] = {'macro_em': metrics['macro_em'], 'macro_nll': metrics['macro_nll'], 'step': state['step']}
                    state['bad_validations'] = 0 if macro_improved else state['bad_validations'] + 1
                    append_json(out / 'validation.jsonl', {'step': state['step'], 'metrics': metrics, 'selected': improved})
                    if improved:
                        save_checkpoint(out / 'best.pt', checkpoint(selected=True))
                save_checkpoint(out / 'latest.pt', checkpoint())
                if state['bad_validations'] >= cfg['patience']:
                    reason = 'early_stopping'
                    break
        except Exception as exc:
            write_json(out / 'failure.json', {'type': type(exc).__name__, 'error': str(exc), 'step': state['step']})
            raise
        save_checkpoint(out / 'latest.pt', checkpoint())
        complete = reason in ('steps_completed', 'early_stopping')
        summary = {'method': 'hard', 'run_signature': run_signature, 'status': 'complete' if complete else 'partial',
                   'stop_reason': reason, 'state': state, 'partial_dataset': run_info['partial_dataset'],
                   'best_checkpoint': str(out / 'best.pt') if state['best'] else None}
        write_json(out / 'run_summary.json', summary)
        if state['best']:
            selected = read_checkpoint(out / 'best.pt')
            write_json(out / 'selected_prompt.json', {'prompt_ids': selected['prompt_ids'], 'text': selected['prompt_text'],
                       'selected_by_validation': True, 'step': selected['state']['step'], 'decode_encode_verified': True})
        return summary

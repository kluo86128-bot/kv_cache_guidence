"""Decode independent, fixed-length C caches as a batch without mixing records.

Private prefills remain per-record. Questions share a forward only along the
batch axis; every question retains its own source cache and absolute RoPE start.
This module never changes the established training/evaluation backend.
"""
import torch
from medical_kv.cache import causal_mask


def question_groups(backend, questions):
    """Up to question_batch_size queries FROM EACH record in each wave."""
    size = (backend.cfg.get('question_batch_size') or max(map(len, questions)))
    if not backend.cfg.get('batch_questions', True):
        size = 1
    for offset in range(0, max(map(len, questions)), size):
        yield [(owner, index, q) for owner, qs in enumerate(questions)
               for index, q in enumerate(qs[offset:offset + size], start=offset)]


def question_inputs(backend, contexts, owners, sequences):
    batch, width = len(sequences), max(map(len, sequences))
    physical = contexts[0].physical_length
    if any(c.full_context or c.physical_length != physical for c in contexts):
        raise ValueError('Cache batching requires equal-length, C-only caches')
    lengths = torch.tensor([len(s) for s in sequences], device=backend.device)
    starts = torch.tensor([contexts[i].next_position for i in owners], device=backend.device)
    columns = torch.arange(width, device=backend.device)[None]
    valid = columns >= (width - lengths[:, None])
    ids = torch.full((batch, width), backend.eos, device=backend.device, dtype=torch.long)
    for index, sequence in enumerate(sequences):
        ids[index, width - len(sequence):] = torch.tensor(sequence, device=backend.device)
    positions = starts[:, None] + (columns - (width - lengths[:, None])).clamp_min(0)
    layers = tuple(tuple(torch.cat([contexts[owner].layers[layer][kv] for owner in owners], dim=0)
                         for kv in (0, 1)) for layer in range(len(contexts[0].layers)))
    key_valid = torch.cat((torch.ones((batch, physical), device=backend.device, dtype=torch.bool), valid), dim=1)
    mask = causal_mask(width, physical, device=backend.device, dtype=backend.dtype).expand(batch, -1, -1, -1).clone()
    mask.masked_fill_(~key_valid[:, None, None, :], torch.finfo(backend.dtype).min)
    return ids, lengths, positions, layers, mask, key_valid, starts


@torch.no_grad()
def answer_losses(backend, contexts, owners, questions):
    tokenized = [backend.question_and_target(qa) for qa in questions]
    sequences = [q + target[:-1] for q, target in tokenized]
    ids, _, positions, layers, mask, _, _ = question_inputs(backend, contexts, owners, sequences)
    targets = [target for _, target in tokenized]
    width = max(map(len, targets))
    labels = torch.full((len(targets), width), -100, device=backend.device, dtype=torch.long)
    for index, target in enumerate(targets):
        labels[index, :len(target)] = torch.tensor(target, device=backend.device)
    lengths = torch.tensor([len(t) for t in targets], device=backend.device)
    indices = (ids.shape[1] - lengths[:, None] + torch.arange(width, device=backend.device)[None]).clamp_max(ids.shape[1] - 1)
    logits, _ = backend.forward(ids=ids, layers=layers, position_ids=positions, mask=mask, logits_slice=indices)
    return backend._mean_answer_losses(logits, labels).tolist()


@torch.no_grad()
def generate(backend, contexts, owners, questions):
    sequences = [backend.ids(backend.question_text(qa['question'])) for qa in questions]
    ids, lengths, positions, layers, mask, key_valid, starts = question_inputs(backend, contexts, owners, sequences)
    logits, layers = backend.forward(ids=ids, layers=layers, position_ids=positions,
                                     mask=mask, logits_slice=slice(-1, None))
    outputs, finished = [[] for _ in questions], [False] * len(questions)
    for step in range(backend.cfg['max_new_tokens']):
        selected = logits[:, -1].argmax(dim=-1).tolist()
        for index, token in enumerate(selected):
            if not finished[index]:
                if token in backend.stops:
                    finished[index] = True
                else:
                    outputs[index].append(token)
        if all(finished) or step + 1 == backend.cfg['max_new_tokens']:
            break
        tokens = torch.tensor([[backend.eos if done else token] for token, done in zip(selected, finished)],
                              device=backend.device)
        key_valid = torch.cat((key_valid, torch.ones((len(questions), 1), device=backend.device, dtype=torch.bool)), dim=1)
        mask = torch.zeros((len(questions), 1, 1, key_valid.shape[1]), device=backend.device, dtype=backend.dtype)
        mask.masked_fill_(~key_valid[:, None, None, :], torch.finfo(backend.dtype).min)
        logits, layers = backend.forward(ids=tokens, layers=layers,
            position_ids=(starts + lengths + step)[:, None], mask=mask, logits_slice=slice(-1, None))
    return [(backend.tokenizer.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False),
             {'generated_tokens': len(tokens), 'stop_reason': 'eos' if done else 'max_new_tokens', 'truncated': not done})
            for tokens, done in zip(outputs, finished)]


@torch.no_grad()
def score_contexts(backend, contexts, questions):
    losses = [[None] * len(qs) for qs in questions]
    generated = [[None] * len(qs) for qs in questions]
    for group in question_groups(backend, questions):
        owners = [owner for owner, _, _ in group]
        qs = [q for _, _, q in group]
        values = answer_losses(backend, contexts, owners, qs)
        answers = generate(backend, contexts, owners, qs)
        for (owner, index, _), value, answer in zip(group, values, answers):
            losses[owner][index], generated[owner][index] = value, answer
    return losses, generated

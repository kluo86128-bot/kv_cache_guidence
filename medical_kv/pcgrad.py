"""Original-field PCGrad; project globally reduced gradients before Adam.

Reference: Yu et al., Gradient Surgery for Multi-Task Learning, Algorithm 1.
https://arxiv.org/abs/2001.06782
"""
import random

import torch


def is_pcgrad(cfg):
    return cfg.get('loss', {}).get('type') in ('birth_focal_pcgrad', 'field_focal_pcgrad')


def local_field_gradients(backend, prompt, records, indices, rank, world_size, fields, *, diagnostics=None):
    """Accumulate each field's global-batch contribution, with no local projection."""
    lookup = {field: i for i, field in enumerate(fields)}
    gradients = torch.zeros((len(fields), *prompt.shape), device=prompt.device, dtype=prompt.dtype)
    losses = []
    for index in indices[rank::world_size]:
        loss, components, objectives = backend.record_loss_components(
            records[index], vectors=prompt, return_fields=True)
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite PCGrad training loss')
        losses.append(float(loss.detach()))
        if diagnostics is not None:
            diagnostics.append({key: float(value) for key, value in components.items()})
        for n, (field, objective) in enumerate(objectives.items()):
            gradient = torch.autograd.grad(objective / len(indices), prompt,
                                           retain_graph=n + 1 < len(objectives))[0]
            if not torch.isfinite(gradient).all():
                raise FloatingPointError(f'Non-finite gradient for field {field}')
            gradients[lookup[field]].add_(gradient)
    return losses, gradients


def project_field_gradients(gradients, fields, *, seed, step):
    """Algorithm 1: edit a copy of g_i against immutable original g_j, then SUM.

    U gradients stay FP32. Dot products/norms accumulate in FP64. No task
    normalization, task-count division or field-frequency reweighting is added.
    Local RNG depends only on seed/update/field, never rank or global RNG state.
    """
    if len(fields) != gradients.shape[0] or not fields or len(set(fields)) != len(fields):
        raise ValueError('PCGrad needs unique field labels matching the gradient rows')
    if not torch.isfinite(gradients).all():
        raise FloatingPointError('Non-finite global field gradients')
    original = gradients.detach().reshape(len(fields), -1)
    reference = original.double()
    squared_norms = (reference * reference).sum(dim=1)
    active = [i for i in range(len(fields)) if squared_norms[i] > 0]
    projected = original.clone()
    projections = 0
    for i in active:
        others = [j for j in active if j != i]
        random.Random(f'{seed}|pcgrad|{step}|{fields[i]}').shuffle(others)
        for j in others:
            dot = torch.dot(projected[i].double(), reference[j])
            if dot < 0:
                correction = (dot / squared_norms[j]) * reference[j]
                projected[i].sub_(correction.to(projected.dtype))
                projections += 1
    merged = projected.sum(dim=0).reshape(gradients.shape[1:])
    if not torch.isfinite(merged).all():
        raise FloatingPointError('Non-finite projected gradient')
    return merged, {'pcgrad_active_fields': len(active), 'pcgrad_projections': projections,
                    'pcgrad_unprojected_norm': float(reference.sum(dim=0).norm()),
                    'pcgrad_projected_norm': float(merged.double().norm())}


def global_pcgrad_backward(backend, prompt, records, indices, runtime, fields, *, seed, step, diagnostics=None):
    losses, gradients = local_field_gradients(backend, prompt, records, indices, runtime.rank,
                                             runtime.world_size, fields, diagnostics=diagnostics)
    # Projection is nonlinear: reduce EACH field first, not projected local sums.
    runtime.sum_(gradients)
    prompt.grad, statistics = project_field_gradients(gradients, fields, seed=seed, step=step)
    return losses, statistics

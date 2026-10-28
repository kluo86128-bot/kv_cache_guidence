"""One public cache with field-routed gradients for two soft-prompt segments."""
import torch


def is_partitioned(cfg):
    return cfg.get('loss', {}).get('type') in ('birth_focal_split', 'field_focal_split')


def is_field_partitioned(cfg):
    return cfg.get('loss', {}).get('type') == 'field_focal_split'


def partition_focal_fields(cfg):
    return tuple(cfg['focal_fields']) if is_field_partitioned(cfg) else ('birth_date',)


def focal_prompt_length(cfg):
    key = 'focal_length' if is_field_partitioned(cfg) else 'birth_length'
    return cfg['prompt_partition'][key]


def routed_gradient(other_loss, focal_loss, prompt, split, scale):
    other = torch.autograd.grad(other_loss / scale, prompt, retain_graph=True)[0]
    focal = torch.autograd.grad(focal_loss / scale, prompt)[0]
    return torch.cat((other[:split], focal[split:]), dim=0)


def optimizer_parameters(initial, cfg):
    if not is_partitioned(cfg):
        return [torch.nn.Parameter(initial.clone())]
    split = cfg['prompt_partition']['other_length']
    return [torch.nn.Parameter(initial[:split].clone()), torch.nn.Parameter(initial[split:].clone())]


def assign_partition_gradients(parameters, gradient, records, indices, *, focal_fields=('birth_date',)):
    """None skips Adam momentum/step updates when an entire global batch lacks a group."""
    split = parameters[0].shape[0]
    selected = set(focal_fields)
    has_focal = any(selected.intersection(records[i]['fields']) for i in indices)
    has_other = any(set(records[i]['fields']).difference(selected) for i in indices)
    parameters[0].grad = gradient[:split].clone() if has_other else None
    parameters[1].grad = gradient[split:].clone() if has_focal else None

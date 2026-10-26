"""One public cache with field-routed gradients for two soft-prompt segments."""
import torch


def is_partitioned(cfg):
    return cfg.get('loss', {}).get('type') == 'birth_focal_split'


def routed_gradient(other_loss, birth_loss, prompt, split, scale):
    other = torch.autograd.grad(other_loss / scale, prompt, retain_graph=True)[0]
    birth = torch.autograd.grad(birth_loss / scale, prompt)[0]
    return torch.cat((other[:split], birth[split:]), dim=0)


def optimizer_parameters(initial, cfg):
    if not is_partitioned(cfg):
        return [torch.nn.Parameter(initial.clone())]
    split = cfg['prompt_partition']['other_length']
    return [torch.nn.Parameter(initial[:split].clone()), torch.nn.Parameter(initial[split:].clone())]


def assign_partition_gradients(parameters, gradient, records, indices):
    """None skips Adam momentum/step updates when an entire global batch lacks a group."""
    split = parameters[0].shape[0]
    has_birth = any('birth_date' in records[i]['fields'] for i in indices)
    has_other = any(any(f != 'birth_date' for f in records[i]['fields']) for i in indices)
    parameters[0].grad = gradient[:split].clone() if has_other else None
    parameters[1].grad = gradient[split:].clone() if has_birth else None

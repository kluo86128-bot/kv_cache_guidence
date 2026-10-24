"""Validate a soft-training continuation independently of its GPU topology."""
import json
from .config import ROOT, fingerprint
from .io import restore_rng, seed_all


def training_contract(backend, cfg, dataset, subsets, code):
    return {'config': cfg, 'dataset': dataset, 'backend': backend,
            'subsets': {k: v for k, v in subsets.items() if k != 'parallel_world_size'},
            'method': 'soft', 'code': code}


def matching_code(actual, profile):
    return set(actual) == set(profile) and all(actual[k] in profile[k] for k in actual)


def validate_resume(checkpoint, contract):
    """Keep data/model/hyperparameter checks; allow only a reviewed code migration."""
    info = checkpoint['run_info']
    if checkpoint['method'] != 'soft' or info.get('method') != 'soft':
        raise ValueError('Resume identity mismatch: expected a soft-training checkpoint')
    if checkpoint.get('resume_format') == 2:
        original = info['origin_identity']
        previous = info['resume_contract']
    else:
        original = {'config': info['config'], 'dataset': info['dataset'],
                    'subsets': info['subsets'], 'backend': info['backend'],
                    'method': 'soft', 'code': info['code_sha256']}
        previous = training_contract(info['backend'], info['config'], info['dataset'],
                                     info['subsets'], info['code_sha256'])
    if (fingerprint(original) != checkpoint['run_signature']
            or info['run_signature'] != checkpoint['run_signature']):
        raise ValueError('Resume identity mismatch: invalid checkpoint origin signature')
    for key in ('config', 'dataset', 'subsets', 'backend', 'method'):
        if previous[key] != contract[key]:
            raise ValueError(f'Resume identity mismatch: {key} changed')
    migration = False
    if previous['code'] != contract['code']:
        registry = json.loads((ROOT / 'configs/resume_compatibility_v1.json').read_text(encoding='utf-8'))
        sources = (registry['legacy_sources'] if checkpoint.get('resume_format') is None
                   else registry.get('reviewed_v2_sources', []) if checkpoint.get('resume_format') == 2 else [])
        if not (matching_code(contract['code'], registry['target_code'])
                and any(matching_code(previous['code'], source) for source in sources)):
            raise ValueError('Resume identity mismatch: code changed without a reviewed migration')
        migration = True
    # Keep the original run identity so a best.pt selected before a GPU change
    # remains a valid candidate for the final automatic test.
    return original, migration


def restore_training_rng(checkpoint, runtime, seed):
    states = checkpoint.get('rng_by_rank') or [checkpoint['rng']]
    if len(states) == runtime.world_size:
        restore_rng(states[runtime.rank])
        return 'restored_per_rank'
    if runtime.rank < len(states):
        restore_rng(states[runtime.rank])
    else:
        # New ranks have no historical RNG stream. Sampling/order are already
        # functions of record/epoch, independent of rank. Record this policy.
        derived = int(fingerprint({'seed': seed, 'step': checkpoint['state']['step'],
                                  'rank': runtime.rank, 'world_size': runtime.world_size})[:15], 16)
        seed_all(derived)
    return 'retained_existing_ranks_seeded_new_ranks'

"""Launch a fresh process with the user's explicit visible GPU selection."""
import os
import subprocess
import sys


def parse_gpus(value):
    pieces = [part.strip() for part in value.split(',')]
    if not pieces or any(not part.isascii() or not part.isdigit() for part in pieces):
        raise ValueError('--gpus must be comma-separated nonnegative GPU IDs, for example 0,2')
    ids = [int(part) for part in pieces]
    if len(set(ids)) != len(ids):
        raise ValueError('--gpus cannot contain duplicate GPU IDs')
    return ids


def launch(gpus, argv):
    ids = parse_gpus(gpus)
    if int(os.environ.get('WORLD_SIZE', '1')) > 1:
        raise ValueError('Do not combine --gpus with an existing torchrun launch')
    clean = []
    skip = False
    for argument in argv:
        if skip:
            skip = False
            continue
        if argument == '--gpus':
            skip = True
        elif not argument.startswith('--gpus='):
            clean.append(argument)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(map(str, ids)), CUDA_DEVICE_ORDER='PCI_BUS_ID', PYTHONUNBUFFERED='1')
    if len(ids) == 1:
        command = [sys.executable, '-m', 'medical_kv', *clean]
    else:
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
                   f'--nproc-per-node={len(ids)}', '--max-restarts=0', '--module', 'medical_kv', *clean]
    print(f"Selected GPUs: {ids}; global batch size is unchanged.", flush=True)
    result = subprocess.run(command, env=env)
    if result.returncode:
        raise SystemExit(result.returncode)

"""Linux server checks: two GPU/NCCL workers, or CPU/Gloo without --gpus."""
import argparse
import contextlib
import copy
from datetime import timedelta
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def fixtures(out, local_device='cpu'):
    import torch
    from medical_kv.config import load_config
    from medical_kv.data import dataset_identity, training_data
    from medical_kv.cli import tiny_backend
    torch.set_num_threads(2)
    cfg = load_config('configs/experiment_v1.json', {'device': 'cuda:0' if local_device.startswith('cuda') else 'cpu', 'dtype': 'float32',
        'train_limit': 3, 'validation_limit': 1, 'epochs': 2, 'max_new_tokens': 2,
        'output_directory': str(out)})
    cfg['batch_records'] = 2  # Last global batch has one patient: rank 1 must contribute zero.
    train, validation, subsets = training_data(cfg, True)
    return tiny_backend(dict(cfg, device=local_device)), cfg, train, validation, dataset_identity(cfg), subsets


def worker(rank, out_string, port, use_cuda):
    import torch
    import torch.distributed as dist
    from medical_kv import training
    from medical_kv.distributed import Runtime
    from medical_kv.evaluation import evaluate_records
    from medical_kv.io import read_checkpoint, write_json
    out = Path(out_string)
    device = f'cuda:{rank}' if use_cuda else 'cpu'
    if use_cuda:
        torch.cuda.set_device(rank)
    store = dist.TCPStore('127.0.0.1', port, world_size=2, is_master=rank == 0,
                         timeout=timedelta(minutes=3), use_libuv=False)
    dist.init_process_group('nccl' if use_cuda else 'gloo', rank=rank, world_size=2, store=store, timeout=timedelta(minutes=3))
    try:
        with (out / f'worker{rank}.log').open('w', encoding='utf-8') as log, contextlib.redirect_stdout(log):
            b, cfg, train, val, identity, subsets = fixtures(out / 'parallel_full', device)
            seen_prompts = []
            evaluate = training.evaluate_records
            def capture(*args, **kwargs):
                seen_prompts.append(kwargs['vectors'].detach().cpu().clone())
                return evaluate(*args, **kwargs)
            training.evaluate_records = capture
            result = training.train_soft(b, cfg, train, val, identity, subsets)
            expected = read_checkpoint(out / 'parallel_full/latest.pt')
            runtime = Runtime(device)
            rank_prompts = runtime.objects(seen_prompts[-1])
            for prompt in rank_prompts:
                torch.testing.assert_close(prompt, expected['prompt'], rtol=0, atol=0)
            # Only rank 0 receives a stop request; all workers must stop together.
            signal = {'requested': False}
            triggered = {'once': False}
            @contextlib.contextmanager
            def boundary():
                signal['requested'] = False
                yield signal
            original_save = training.save_checkpoint
            def stop_after_one(path, checkpoint):
                original_save(path, checkpoint)
                if checkpoint['state']['step'] == 1 and not triggered['once']:
                    triggered['once'] = True
                    signal['requested'] = True
            training.stop_at_boundary = boundary
            training.save_checkpoint = stop_after_one
            cfg2 = copy.deepcopy(cfg)
            cfg2['output_directory'] = str(out / 'parallel_resumed')
            partial = training.train_soft(b, cfg2, train, val, identity, subsets)
            assert partial['status'] == 'partial' and partial['state']['step'] == 1
            resumed = training.train_soft(b, cfg2, train, val, identity, subsets,
                resume=out / 'parallel_resumed/latest.pt')
            actual = read_checkpoint(out / 'parallel_resumed/latest.pt')
            torch.testing.assert_close(actual['prompt'], expected['prompt'], rtol=0, atol=0)
            assert len(actual['rng_by_rank']) == 2
            assert resumed['state']['step'] == result['state']['step'] == 4
            metrics, rows = evaluate_records(b, train, vectors=actual['prompt'], output=out / 'parallel_evaluation')
            assert metrics['records'] == 3 and len(rows) == 12
            assert [(row['record_id'], row['field']) for row in rows] == [
                (record['record_id'], qa['field']) for record in train for qa in record['qa']]
            write_json(out / f'rank{rank}_checks.json', {'status': 'PASS', 'rank': rank,
                'shared_prompt_identical': True, 'exact_resume': True, 'one_rank_stop_propagation': True,
                'short_global_batch_and_idle_rank': True, 'evaluation_records': metrics['records']})
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-directory', default='runs/implementation_checks/parallel_check_server')
    parser.add_argument('--gpus', help='Exactly two server GPU IDs, for example 0,2')
    args = parser.parse_args()
    import os
    if sys.platform != 'linux':
        parser.error('Run this distributed check on the Linux server')
    if args.gpus is not None:
        from medical_kv.launch import parse_gpus
        try:
            ids = parse_gpus(args.gpus)
        except ValueError as exc:
            parser.error(str(exc))
        if len(ids) != 2:
            parser.error('This comparison check needs exactly two GPUs; normal training accepts any GPU count')
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, ids))
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    import torch
    import torch.multiprocessing as mp
    from medical_kv.training import train_soft
    from medical_kv.io import code_identity, read_checkpoint, write_json
    use_cuda = args.gpus is not None
    if use_cuda and (torch.cuda.device_count() != 2 or not torch.distributed.is_nccl_available()):
        parser.error('Need two selected CUDA GPUs and a PyTorch build with NCCL')
    device = 'cuda:0' if use_cuda else 'cpu'
    if use_cuda:
        torch.cuda.set_device(0)
    out = Path(args.output_directory)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=False)
    with (out / 'single.log').open('w', encoding='utf-8') as log, contextlib.redirect_stdout(log):
        b, cfg, train, val, identity, subsets = fixtures(out / 'single', device)
        train_soft(b, cfg, train, val, identity, subsets)
        single = read_checkpoint(out / 'single/latest.pt')
    import socket
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    mp.spawn(worker, args=(str(out), port, use_cuda), nprocs=2, join=True)
    parallel = read_checkpoint(out / 'parallel_full/latest.pt')
    torch.testing.assert_close(single['prompt'], parallel['prompt'], rtol=2e-4, atol=2e-6)
    for rank in range(2):
        assert json.loads((out / f'rank{rank}_checks.json').read_text())['status'] == 'PASS'
    # Timings concern only random tiny forward/backward, never the full 4B model.
    timings = {}
    for batched in (False, True):
        b.cfg['batch_questions'] = batched
        prompt = b.embedding(b.tensor(b.initial_ids()))[0].detach().clone().requires_grad_(True)
        b.record_loss(train[0], vectors=prompt).backward()
        if use_cuda:
            torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(3):
            prompt.grad = None
            b.record_loss(train[0], vectors=prompt).backward()
        if use_cuda:
            torch.cuda.synchronize()
        timings['batched' if batched else 'sequential'] = (time.perf_counter() - started) / 3
    report = {'status': 'PASS', 'implementation_only': True,
        'hardware': 'Two CUDA/NCCL workers' if use_cuda else 'Two CPU/Gloo workers', 'model': 'random tiny Qwen3',
        'multi_gpu_hardware_tested': use_cuda, 'world_size': 2, 'global_batch_records': 2,
        'single_vs_parallel_prompt_max_abs_difference': float((single['prompt'] - parallel['prompt']).abs().max()),
        'exact_distributed_resume': True, 'one_rank_stop_propagation': True,
        'short_last_batch_no_duplicates': True, 'idle_rank_validation': True,
        'evaluation_merge_preserves_record_order': True, 'seconds_per_record_forward_backward': timings,
        'code_sha256': code_identity()}
    write_json(out / 'verification.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

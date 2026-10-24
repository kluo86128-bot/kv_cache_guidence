"""Patient data parallelism: synchronize only the trainable shared prompt."""
import os
from datetime import timedelta
import torch
import torch.distributed as dist


class Runtime:
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        self.active = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if self.active else 0
        self.world_size = dist.get_world_size() if self.active else 1
        self.primary = self.rank == 0

    def barrier(self):
        if self.active:
            dist.barrier()

    def objects(self, obj):
        if not self.active:
            return [obj]
        result = [None] * self.world_size
        dist.all_gather_object(result, obj)
        return result

    def broadcast(self, obj):
        result = [obj if self.primary else None]
        if self.active:
            dist.broadcast_object_list(result, src=0)
        return result[0]

    def sum_(self, tensor):
        if self.active:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor

    def maximum(self, value):
        if not self.active:
            return value
        tensor = torch.tensor(float(value), device=self.device, dtype=torch.float64)
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        return tensor.item()

    def any(self, value):
        return bool(self.maximum(bool(value)))

    def metadata(self):
        return {'world_size': self.world_size, 'backend': dist.get_backend() if self.active else None,
                'visible_gpu_ids': os.environ.get('CUDA_VISIBLE_DEVICES'),
                'global_batch_semantics': 'batch_records is global; no padding or duplicated patients'}


def initialize(*, tiny=False):
    world = int(os.environ.get('WORLD_SIZE', '1'))
    if world <= 1:
        return Runtime('cpu' if tiny else 'cuda:0')
    if not dist.is_available():
        raise RuntimeError('This PyTorch build does not support distributed execution')
    local = int(os.environ['LOCAL_RANK'])
    if tiny:
        device, backend = 'cpu', 'gloo'
    else:
        if not torch.cuda.is_available() or local >= torch.cuda.device_count():
            raise RuntimeError('Each worker needs one available selected GPU')
        if not dist.is_nccl_available():
            raise RuntimeError('Multi-GPU runs require a PyTorch build with NCCL (use the Linux CUDA server)')
        torch.cuda.set_device(local)
        device, backend = f'cuda:{local}', 'nccl'
    dist.init_process_group(backend=backend, timeout=timedelta(minutes=15))
    return Runtime(device)


def cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

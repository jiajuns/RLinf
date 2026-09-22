"""Small dependency-free distributed primitives for Event sidecars."""
from __future__ import annotations

import torch
import torch.distributed as dist
from torch import nn


def distributed_ready(world_size: int) -> bool:
    return world_size > 1 and dist.is_available() and dist.is_initialized()


def global_count(local_count: int, *, world_size: int, device: torch.device) -> int:
    value = torch.tensor(int(local_count), dtype=torch.long, device=device)
    if distributed_ready(world_size):
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return int(value.item())


def global_scalar_sum(local_value: torch.Tensor, *, world_size: int) -> torch.Tensor:
    value = local_value.detach().clone()
    if distributed_ready(world_size):
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


def all_reduce_gradients(module: nn.Module, *, world_size: int, normalizer: int | None = None) -> None:
    """Sum gradients across ranks and divide by a global sample normalizer."""
    for parameter in module.parameters():
        if parameter.grad is None:
            continue
        if distributed_ready(world_size):
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        parameter.grad.div_(normalizer if normalizer is not None else world_size)


def broadcast_module(module: nn.Module, *, world_size: int, src: int = 0) -> None:
    if not distributed_ready(world_size):
        return
    for tensor in list(module.parameters()) + list(module.buffers()):
        dist.broadcast(tensor.data, src=src)

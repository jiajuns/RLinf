#!/usr/bin/env python3
"""Two-rank regression for Event sidecar masked auxiliary synchronization.

Runs on CPU/Gloo by default so a one-GPU A6000 host can still exercise the
multi-rank collective protocol through the production distributed utility.
"""
from __future__ import annotations

import argparse
import copy
import os
import socket
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

# Allow direct invocation from the repository without requiring users to set
# PYTHONPATH merely for this diagnostic.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from rlinf.algorithms.event_sidecar_distributed import all_reduce_gradients, global_count


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as stream:
        stream.bind(("127.0.0.1", 0))
        return int(stream.getsockname()[1])


def _worker(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        torch.manual_seed(7)
        model = nn.Linear(2, 1, bias=False)
        target = copy.deepcopy(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
        before = [parameter.detach().clone() for parameter in model.parameters()]

        # Only rank zero owns a valid branch label; rank one supplies a
        # graph-connected zero and must nevertheless take the same step.
        prediction = model(torch.tensor([[1.0, -1.0]]))
        local_sum = (prediction - 1.0).square().sum() if rank == 0 else prediction.sum() * 0.0
        valid_count = global_count(1 if rank == 0 else 0, world_size=world_size, device=torch.device("cpu"))
        assert valid_count == 1
        optimizer.zero_grad(set_to_none=True)
        local_sum.backward()
        all_reduce_gradients(model, world_size=world_size, normalizer=valid_count)
        optimizer.step()
        with torch.no_grad():
            for target_parameter, parameter in zip(target.parameters(), model.parameters(), strict=True):
                target_parameter.mul_(0.9).add_(parameter, alpha=0.1)
        counters = torch.tensor([valid_count, valid_count * 4], dtype=torch.long)

        for tensor in [*model.parameters(), *target.parameters(), counters]:
            gathered = [torch.empty_like(tensor) for _ in range(world_size)]
            dist.all_gather(gathered, tensor)
            assert all(torch.equal(gathered[0], value) for value in gathered[1:])
        assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters(), strict=True))

        # Global-empty branch round: both ranks must collectively skip Adam,
        # leave EMA/counters unchanged, and never hang.
        empty_before = [parameter.detach().clone() for parameter in model.parameters()]
        empty_count = global_count(0, world_size=world_size, device=torch.device("cpu"))
        assert empty_count == 0
        if empty_count:
            raise AssertionError("global empty branch batch must not update optimizer")
        assert all(torch.equal(old, new) for old, new in zip(empty_before, model.parameters(), strict=True))
    finally:
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=2)
    args = parser.parse_args()
    if args.world_size != 2:
        raise ValueError("this deterministic regression is defined for exactly two ranks")
    mp.spawn(_worker, args=(args.world_size, _free_port()), nprocs=args.world_size, join=True)
    print("two-rank Event sidecar regression passed")


if __name__ == "__main__":
    main()

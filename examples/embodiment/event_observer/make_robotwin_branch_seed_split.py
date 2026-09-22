#!/usr/bin/env python3
"""Create a fixed, disjoint RoboTwin reset-seed protocol for branch audits.

The output is intentionally a tiny JSON artifact rather than a random split
inside a training command.  It records the source manifest digest and emits
one RLinf-compatible ``seeds.json`` per split, so an independent episode can
never accidentally reappear in Influence training or test evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _seed_payload(task: str, seeds: list[int], source_task: dict) -> dict:
    payload = dict(source_task)
    payload["success_seeds"] = [int(seed) for seed in seeds]
    return {task: payload}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--task", default="adjust_bottle")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train", type=int, default=16)
    parser.add_argument("--validation", type=int, default=8)
    parser.add_argument("--test", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260923)
    args = parser.parse_args()
    counts = {"train": args.train, "validation": args.validation, "test": args.test}
    if any(count < 1 for count in counts.values()):
        raise ValueError("every split must contain at least one episode")

    raw = args.source.read_bytes()
    source = json.loads(raw)
    if args.task not in source or "success_seeds" not in source[args.task]:
        raise ValueError(f"{args.source} lacks {args.task}.success_seeds")
    candidates = np.asarray(source[args.task]["success_seeds"], dtype=np.int64)
    total = sum(counts.values())
    if len(candidates) < total:
        raise ValueError(f"requested {total} unique seeds, source provides only {len(candidates)}")

    rng = np.random.default_rng(args.seed)
    selected = rng.choice(candidates, size=total, replace=False).astype(np.int64).tolist()
    splits: dict[str, list[int]] = {}
    start = 0
    for name, count in counts.items():
        splits[name] = [int(seed) for seed in selected[start : start + count]]
        start += count
    flat = [seed for values in splits.values() for seed in values]
    if len(set(flat)) != len(flat):
        raise AssertionError("split construction unexpectedly produced overlapping seeds")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_task = source[args.task]
    for name, seeds in splits.items():
        (args.output_dir / f"{name}_seeds.json").write_text(
            json.dumps(_seed_payload(args.task, seeds, source_task), indent=2) + "\n"
        )
    manifest = {
        "protocol": "robotwin_branch_independent_episode_split_v1",
        "task": args.task,
        "source": str(args.source),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "random_seed": args.seed,
        "counts": counts,
        "splits": splits,
        "notes": (
            "All splits are disjoint reset seeds. Train/validation/test are fixed before "
            "branch collection; test seeds must not be used to train Observer, Event Value, "
            "Influence, or select hyperparameters."
        ),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

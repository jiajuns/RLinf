#!/usr/bin/env python3
"""Summarize matched πRL and Event-SMDP TensorBoard runs.

Example:
    python tools/summarize_pirl_comparison.py \
      --baseline outputs/action_gae_seed0 outputs/action_gae_seed1 \
      --event outputs/event_smdp_seed0 outputs/event_smdp_seed1 \
      --output outputs/comparison_summary.json

The script treats an unreached N80/N90 as censored (``null``), rather than
silently claiming a sample-efficiency gain.  It uses evaluation success for
all headline metrics; rollout returns remain diagnostic only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


SUCCESS_TAG = "eval/success_once"


def load_scalars(run_dir: Path, tag: str) -> list[tuple[int, float]]:
    event_files = sorted(run_dir.rglob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event file under {run_dir}")
    accumulator = EventAccumulator(str(event_files[-1]))
    accumulator.Reload()
    available = accumulator.Tags().get("scalars", [])
    if tag not in available:
        raise KeyError(f"{run_dir} has no '{tag}'. Available scalar tags: {available}")
    return [(item.step, float(item.value)) for item in accumulator.Scalars(tag)]


def threshold_step(series: list[tuple[int, float]], threshold: float) -> int | None:
    return next((step for step, value in series if value >= threshold), None)


def auc(series: list[tuple[int, float]]) -> float:
    if len(series) < 2:
        return float("nan")
    steps = np.asarray([step for step, _ in series], dtype=np.float64)
    values = np.asarray([value for _, value in series], dtype=np.float64)
    span = steps[-1] - steps[0]
    return float(np.trapezoid(values, steps) / span) if span > 0 else float(values[-1])


def summarize_method(run_dirs: list[Path]) -> dict[str, Any]:
    per_seed: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        series = load_scalars(run_dir, SUCCESS_TAG)
        per_seed.append(
            {
                "run": str(run_dir),
                "final_success": series[-1][1],
                "success_auc": auc(series),
                "n80": threshold_step(series, 0.80),
                "n90": threshold_step(series, 0.90),
                "series": [{"step": step, "success": value} for step, value in series],
            }
        )

    def mean_std(key: str) -> dict[str, float | None]:
        values = np.asarray([seed[key] for seed in per_seed], dtype=np.float64)
        if np.isnan(values).all():
            return {"mean": None, "std": None}
        return {"mean": float(np.nanmean(values)), "std": float(np.nanstd(values))}

    # Threshold values are deliberately reported only for seeds which reached
    # them; ``reached_seeds`` prevents a censored result being misread as zero.
    def threshold_summary(key: str) -> dict[str, Any]:
        reached = [seed[key] for seed in per_seed if seed[key] is not None]
        return {
            "reached_seeds": len(reached),
            "total_seeds": len(per_seed),
            "mean": float(np.mean(reached)) if reached else None,
            "std": float(np.std(reached)) if reached else None,
        }

    return {
        "num_seeds": len(per_seed),
        "final_success": mean_std("final_success"),
        "success_auc": mean_std("success_auc"),
        "n80": threshold_summary("n80"),
        "n90": threshold_summary("n90"),
        "per_seed": per_seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", nargs="+", required=True, type=Path)
    parser.add_argument("--event", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if len(args.baseline) != len(args.event):
        raise ValueError("baseline and event must have the same number of matched seeds")

    summary = {
        "metric": SUCCESS_TAG,
        "baseline": summarize_method(args.baseline),
        "event_smdp": summarize_method(args.event),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

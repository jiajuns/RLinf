#!/usr/bin/env python3
"""Audit Monte-Carlo stability of policy-relative Influence references.

This is inference-only: saved same-observation Flow-SDE action samples are
rescored by a frozen Influence Model.  It neither runs the simulator nor
changes the actor.  The result answers whether two reference samples are a
reasonable cost/stability trade-off before Event credit is enabled.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rlinf.algorithms.event_intervention import EventInfluenceModel, policy_relative_influence


def _load_diagnostics(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    representations, primary_actions, references = [], [], []
    files = sorted(path.glob("branch_diag_rank*_opt*_seen*.npz"))
    if not files:
        raise FileNotFoundError(f"no branch diagnostics under {path}")
    for file in files:
        with np.load(file, allow_pickle=False) as payload:
            required = {"state_representations", "candidate_actions", "policy_reference_actions"}
            missing = required.difference(payload.files)
            if missing:
                raise ValueError(f"{file} lacks {sorted(missing)}; recollect with reference persistence enabled")
            rep = np.asarray(payload["state_representations"], dtype=np.float32)
            candidates = np.asarray(payload["candidate_actions"], dtype=np.float32)
            refs = np.asarray(payload["policy_reference_actions"], dtype=np.float32)
            if rep.ndim != 2 or candidates.ndim != 4 or refs.ndim != 4:
                raise ValueError(f"{file} has malformed reference tensors")
            if rep.shape[0] != candidates.shape[0] or rep.shape[0] != refs.shape[0] or refs.shape[1] < 2:
                raise ValueError(f"{file} has unaligned or insufficient reference samples")
            representations.append(rep)
            primary_actions.append(candidates[:, 0].reshape(len(rep), -1))
            references.append(refs.reshape(len(rep), refs.shape[1], -1))
    return tuple(np.concatenate(values, axis=0) for values in (representations, primary_actions, references))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True,
                        help="Frozen sidecar containing the Influence Model to audit.")
    parser.add_argument("--reference-counts", default="2,4,8")
    parser.add_argument("--trials", type=int, default=128)
    parser.add_argument("--sign-eps", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.trials < 1 or args.sign_eps < 0:
        raise ValueError("trials and sign-eps must be valid")

    reps_np, actions_np, refs_np = _load_diagnostics(args.diagnostics)
    maximum = refs_np.shape[1]
    counts = sorted({int(value) for value in args.reference_counts.split(",") if value})
    if not counts or min(counts) < 2 or max(counts) > maximum:
        raise ValueError(f"reference counts must be in [2,{maximum}]")
    payload = torch.load(args.sidecar, map_location="cpu", weights_only=False)
    state = payload.get("influence_model")
    if not state:
        raise ValueError("sidecar does not contain a trained Influence Model")
    hidden_dim = int(state["network.1.weight"].shape[0])
    model = EventInfluenceModel(reps_np.shape[1], actions_np.shape[1], hidden_dim=hidden_dim).eval()
    model.load_state_dict(state)
    reps = torch.from_numpy(reps_np)
    actions = torch.from_numpy(actions_np)
    refs = torch.from_numpy(refs_np)
    with torch.no_grad():
        full = policy_relative_influence(model, reps, actions, refs).numpy()

    rng = np.random.default_rng(args.seed)
    eligible_sign = np.abs(full) > args.sign_eps
    report: dict[str, object] = {
        "diagnostics": str(args.diagnostics), "sidecar": str(args.sidecar),
        "states": int(len(reps_np)), "available_reference_candidates": int(maximum),
        "trials": args.trials, "sign_eps": args.sign_eps,
        "full_reference_score": {
            "mean": float(full.mean()), "std": float(full.std()),
            "absolute_median": float(np.median(np.abs(full))),
            "sign_eligible_states": int(eligible_sign.sum()),
        },
        "counts": {},
    }
    for count in counts:
        samples = []
        for _ in range(args.trials):
            indices = np.stack([rng.choice(maximum, size=count, replace=False) for _ in range(len(reps_np))])
            row = torch.arange(len(reps_np))[:, None]
            subset = refs[row, torch.from_numpy(indices)]
            with torch.no_grad():
                samples.append(policy_relative_influence(model, reps, actions, subset).numpy())
        values = np.stack(samples)
        per_state_std = values.std(axis=0)
        signs = np.sign(values[:, eligible_sign]) if eligible_sign.any() else np.empty((args.trials, 0))
        agreement = signs == np.sign(full[eligible_sign])[None, :] if eligible_sign.any() else np.empty((args.trials, 0), bool)
        report["counts"][str(count)] = {
            "score_mean": float(values.mean()), "score_std_over_trials_mean": float(per_state_std.mean()),
            "score_std_over_trials_p90": float(np.quantile(per_state_std, .9)),
            "sign_agreement_vs_full": float(agreement.mean()) if agreement.size else None,
            "sign_flip_rate_vs_full": float((~agreement).mean()) if agreement.size else None,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

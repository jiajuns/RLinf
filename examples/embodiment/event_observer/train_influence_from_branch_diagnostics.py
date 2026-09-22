#!/usr/bin/env python3
"""Fit ``I_xi`` from frozen, matched-state branch diagnostics.

The online actor writes one lightweight row per cloned state: an Event
representation, all Flow-SDE candidate chunks and their controlled outcomes.
This utility turns those rows into a proper train/validation problem before
the model is allowed to influence PPO.  It intentionally consumes neither
RGB frames nor simulator oracle state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from rlinf.algorithms.event_intervention import EventInfluenceModel


def _duplicate_action_noise(actions: np.ndarray, outcomes: np.ndarray, *, action_eps: float) -> np.ndarray:
    """Return a conservative per-state noise floor from repeated candidates.

    Flow sampling can intentionally emit the policy chunk twice (candidate 0
    and its replay control).  Those pairs are a direct test of the complete
    clone -> render -> student -> Event-Value label path: the environment
    reward may agree while the bootstrap does not.  We never use this number
    as a training target.  It only filters states whose candidate spread is
    indistinguishable from the measured repeat noise.
    """
    noise = np.zeros(len(outcomes), dtype=np.float32)
    for state, (candidate_actions, candidate_returns) in enumerate(zip(actions, outcomes, strict=True)):
        repeated = []
        for left in range(len(candidate_returns)):
            for right in range(left + 1, len(candidate_returns)):
                if np.max(np.abs(candidate_actions[left] - candidate_actions[right])) <= action_eps:
                    repeated.append(abs(float(candidate_returns[left] - candidate_returns[right])))
        if repeated:
            noise[state] = max(repeated)
    return noise


def _load_rows(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    representations, actions, returns, groups = [], [], [], []
    paths = sorted(root.glob("branch_diag_rank*_opt*_seen*.npz"))
    if not paths:
        raise FileNotFoundError(f"no branch diagnostic files under {root}")
    for path in paths:
        with np.load(path, allow_pickle=False) as payload:
            required = {"state_representations", "candidate_actions", "branch_returns", "candidate_valid_mask", "state_ids"}
            missing = required.difference(payload.files)
            if missing:
                raise ValueError(f"{path} lacks required offline Influence fields: {sorted(missing)}")
            rep = np.asarray(payload["state_representations"], dtype=np.float32)
            action = np.asarray(payload["candidate_actions"], dtype=np.float32)
            outcome = np.asarray(payload["branch_returns"], dtype=np.float32)
            valid = np.asarray(payload["candidate_valid_mask"], dtype=bool)
            state_ids = np.asarray(payload["state_ids"], dtype=np.int64)
            if rep.ndim != 2 or action.ndim != 4 or outcome.ndim != 2:
                raise ValueError(f"{path} has malformed branch tensors")
            if action.shape[:2] != outcome.shape or valid.shape != outcome.shape or rep.shape[0] != outcome.shape[0]:
                raise ValueError(f"{path} has unaligned branch tensors")
            # All candidates must be valid for a meaningful centered ranking.
            keep = valid.all(axis=1)
            representations.append(rep[keep])
            actions.append(action[keep].reshape(int(keep.sum()), outcome.shape[1], -1))
            returns.append(outcome[keep])
            # Reset seed is a stable episode-level grouping key.  Hold it out
            # as a group rather than randomly splitting correlated chunks.
            groups.append(state_ids[keep, 0])
    return tuple(np.concatenate(items, axis=0) for items in (representations, actions, returns, groups))


def _validation_mask(groups: np.ndarray, fraction: float) -> np.ndarray:
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError("at least two reset-seed groups are needed for held-out Influence validation")
    selected = []
    for group in unique:
        value = int(hashlib.sha256(str(int(group)).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        if value < fraction:
            selected.append(group)
    if not selected:
        selected = [unique[-1]]
    if len(selected) == len(unique):
        selected.pop()
    return np.isin(groups, np.asarray(selected))


def _ranking_metrics(true_values: torch.Tensor, predicted_values: torch.Tensor, tie_eps: float) -> dict[str, float]:
    truth = true_values.detach().cpu().numpy()
    prediction = predicted_values.detach().cpu().numpy()
    centered = truth - truth.mean(axis=1, keepdims=True)
    eligible = concordant = discordant = predicted_ties = true_ties = 0
    regrets = []
    for target, score in zip(truth, prediction, strict=True):
        for left in range(len(target)):
            for right in range(left + 1, len(target)):
                delta = target[left] - target[right]
                predicted_delta = score[left] - score[right]
                if abs(float(delta)) <= tie_eps:
                    true_ties += 1
                    continue
                eligible += 1
                if abs(float(predicted_delta)) <= tie_eps:
                    predicted_ties += 1
                elif np.sign(delta) == np.sign(predicted_delta):
                    concordant += 1
                else:
                    discordant += 1
        regrets.append(float(target.max() - target[np.argmax(score)]))
    denominator = np.sqrt((concordant + discordant + true_ties) * (concordant + discordant + predicted_ties))
    zero_mse = float(np.mean(centered**2))
    model_mse = float(np.mean((prediction - centered) ** 2))
    return {
        "tie_aware_pairwise_accuracy": float(concordant / eligible) if eligible else float("nan"),
        "top1_regret_mean": float(np.mean(regrets)),
        "eligible_pairs": float(eligible),
        "concordant": float(concordant),
        "discordant": float(discordant),
        "predicted_ties": float(predicted_ties),
        "true_ties": float(true_ties),
        "kendall_tau_b": float((concordant - discordant) / denominator) if denominator else float("nan"),
        "zero_predictor_mse": zero_mse,
        "model_mse": model_mse,
        "model_vs_zero_mse_ratio": model_mse / max(zero_mse, np.finfo(np.float64).eps),
    }


def _statewise_shuffle_baseline(
    true_values: torch.Tensor, predicted_values: torch.Tensor, tie_eps: float, rng: np.random.Generator, trials: int
) -> dict[str, float]:
    """Break candidate correspondence while preserving each state's score distribution."""
    scores = predicted_values.detach().cpu().numpy()
    values = []
    for _ in range(trials):
        shuffled = np.stack([row[rng.permutation(len(row))] for row in scores])
        values.append(_ranking_metrics(true_values, torch.from_numpy(shuffled), tie_eps))
    keys = ("tie_aware_pairwise_accuracy", "kendall_tau_b", "top1_regret_mean")
    return {f"random_statewise_{key}_mean": float(np.nanmean([item[key] for item in values])) for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--output-sidecar", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--validation-fraction", type=float, default=.25)
    parser.add_argument("--tie-eps", type=float, default=1e-4)
    parser.add_argument("--random-trials", type=int, default=100)
    parser.add_argument("--action-repeat-eps", type=float, default=0.0)
    parser.add_argument("--min-spread-over-repeat-noise", type=float, default=0.0,
                        help="Keep states only when range(Y) exceeds this multiple of their repeated-action noise.")
    parser.add_argument("--min-absolute-spread", type=float, default=0.0)
    parser.add_argument("--normalized-target-loss", action="store_true",
                        help="Scale MSE gradients by train target std; model outputs remain in original return units.")
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--overfit-all-states", action="store_true",
                        help="Diagnostic only: train and report on the same frozen branch states; never a generalization result.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not 0 < args.validation_fraction < 1 or args.epochs < 1:
        raise ValueError("validation fraction and epochs must be positive")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    representations, actions, outcomes, groups = _load_rows(args.diagnostics)
    repeat_noise = _duplicate_action_noise(actions, outcomes, action_eps=args.action_repeat_eps)
    state_spread = outcomes.max(axis=1) - outcomes.min(axis=1)
    threshold = np.maximum(args.min_absolute_spread, args.min_spread_over_repeat_noise * repeat_noise)
    keep = state_spread > threshold
    if not keep.any():
        raise ValueError("signal filter removed every branch state")
    representations, actions, outcomes, groups, repeat_noise, state_spread = (
        value[keep] for value in (representations, actions, outcomes, groups, repeat_noise, state_spread)
    )
    targets = outcomes - outcomes.mean(axis=1, keepdims=True)
    validation = np.zeros(len(groups), dtype=bool) if args.overfit_all_states else _validation_mask(groups, args.validation_fraction)
    if not args.overfit_all_states and (not validation.any() or validation.all()):
        raise RuntimeError("invalid grouped train/validation split")
    payload = torch.load(args.sidecar, map_location="cpu", weights_only=False)
    if payload.get("format") != "eventvalue_rl_sidecars_v2":
        raise ValueError("sidecar format is not eventvalue_rl_sidecars_v2")
    hidden_dim = int(payload["influence_model"]["network.1.weight"].shape[0])
    model = EventInfluenceModel(representations.shape[-1], actions.shape[-1], hidden_dim=hidden_dim)
    model.load_state_dict(payload["influence_model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_rep = torch.from_numpy(representations[~validation])
    train_actions = torch.from_numpy(actions[~validation])
    train_targets = torch.from_numpy(targets[~validation])
    train_returns = torch.from_numpy(outcomes[~validation])
    train = TensorDataset(train_rep, train_actions, train_targets)
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    val_rep = torch.from_numpy(representations[validation])
    val_actions = torch.from_numpy(actions[validation])
    val_targets = torch.from_numpy(targets[validation])
    val_returns = torch.from_numpy(outcomes[validation])
    target_scale = float(train_targets.std().clamp_min(1e-8)) if args.normalized_target_loss else 1.0
    best = float("inf"); best_state = None; history = []
    for epoch in range(args.epochs):
        model.train()
        for rep, action, target in loader:
            prediction = model(rep.unsqueeze(1).expand(-1, action.shape[1], -1), action)
            loss = EventInfluenceModel.loss(prediction / target_scale, target / target_scale)
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        model.eval()
        with torch.no_grad():
            train_prediction = model(train_rep.unsqueeze(1).expand(-1, train_actions.shape[1], -1), train_actions)
            evaluation_rep = train_rep if args.overfit_all_states else val_rep
            evaluation_actions = train_actions if args.overfit_all_states else val_actions
            evaluation_targets = train_targets if args.overfit_all_states else val_targets
            evaluation_returns = train_returns if args.overfit_all_states else val_returns
            prediction = model(evaluation_rep.unsqueeze(1).expand(-1, evaluation_actions.shape[1], -1), evaluation_actions)
            value_loss = float(EventInfluenceModel.loss(prediction / target_scale, evaluation_targets / target_scale))
            train_ranking = _ranking_metrics(train_returns, train_prediction, args.tie_eps)
            ranking = _ranking_metrics(evaluation_returns, prediction, args.tie_eps)
            random_baseline = _statewise_shuffle_baseline(
                evaluation_returns, prediction, args.tie_eps, np.random.default_rng(args.seed + epoch), args.random_trials
            )
        history.append({"epoch": epoch, "train": train_ranking, "validation": ranking, "validation_random_baseline": random_baseline})
        if value_loss < best:
            best = value_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    payload["influence_model"] = model.state_dict()
    payload["influence_optimizer"] = optimizer.state_dict()
    best_metrics = history[int(np.argmin([row["validation"]["model_mse"] for row in history]))]
    payload["offline_influence_training"] = {
        "diagnostics": str(args.diagnostics), "states": int(len(outcomes)), "train_states": int((~validation).sum()),
        "validation_states": int(validation.sum()), "best_validation_mse": best,
        "best_epoch": int(np.argmin([row["validation"]["model_mse"] for row in history])),
        "diagnostic_settings": {
            "overfit_all_states": args.overfit_all_states, "normalized_target_loss": args.normalized_target_loss,
            "target_scale": target_scale, "action_repeat_eps": args.action_repeat_eps,
            "min_spread_over_repeat_noise": args.min_spread_over_repeat_noise,
            "min_absolute_spread": args.min_absolute_spread,
        },
        "signal_audit": {
            "repeat_noise_median": float(np.median(repeat_noise)), "repeat_noise_p90": float(np.quantile(repeat_noise, .9)),
            "state_spread_median": float(np.median(state_spread)), "state_spread_p90": float(np.quantile(state_spread, .9)),
        },
    }
    args.output_sidecar.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output_sidecar)
    report = {
        "split_groups": {"train": sorted(map(int, np.unique(groups[~validation]))), "validation": sorted(map(int, np.unique(groups[validation])))},
        "best_epoch_metrics": best_metrics, "history": history, **payload["offline_influence_training"],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

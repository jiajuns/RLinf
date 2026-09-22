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
    eligible = concordant = 0
    regrets = []
    for target, score in zip(truth, prediction, strict=True):
        for left in range(len(target)):
            for right in range(left + 1, len(target)):
                delta = target[left] - target[right]
                if abs(float(delta)) <= tie_eps:
                    continue
                eligible += 1
                if np.sign(delta) == np.sign(score[left] - score[right]):
                    concordant += 1
        regrets.append(float(target.max() - target[np.argmax(score)]))
    return {
        "tie_aware_pairwise_accuracy": float(concordant / eligible) if eligible else float("nan"),
        "top1_regret_mean": float(np.mean(regrets)),
        "eligible_pairs": float(eligible),
    }


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
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not 0 < args.validation_fraction < 1 or args.epochs < 1:
        raise ValueError("validation fraction and epochs must be positive")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    representations, actions, outcomes, groups = _load_rows(args.diagnostics)
    targets = outcomes - outcomes.mean(axis=1, keepdims=True)
    validation = _validation_mask(groups, args.validation_fraction)
    if not validation.any() or validation.all():
        raise RuntimeError("invalid grouped train/validation split")
    payload = torch.load(args.sidecar, map_location="cpu", weights_only=False)
    if payload.get("format") != "eventvalue_rl_sidecars_v2":
        raise ValueError("sidecar format is not eventvalue_rl_sidecars_v2")
    hidden_dim = int(payload["influence_model"]["network.1.weight"].shape[0])
    model = EventInfluenceModel(representations.shape[-1], actions.shape[-1], hidden_dim=hidden_dim)
    model.load_state_dict(payload["influence_model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    train = TensorDataset(
        torch.from_numpy(representations[~validation]), torch.from_numpy(actions[~validation]), torch.from_numpy(targets[~validation])
    )
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    val_rep = torch.from_numpy(representations[validation])
    val_actions = torch.from_numpy(actions[validation])
    val_targets = torch.from_numpy(targets[validation])
    val_returns = torch.from_numpy(outcomes[validation])
    best = float("inf"); best_state = None; history = []
    for epoch in range(args.epochs):
        model.train()
        for rep, action, target in loader:
            loss = EventInfluenceModel.loss(model(rep.unsqueeze(1).expand(-1, action.shape[1], -1), action), target)
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        model.eval()
        with torch.no_grad():
            prediction = model(val_rep.unsqueeze(1).expand(-1, val_actions.shape[1], -1), val_actions)
            value_loss = float(EventInfluenceModel.loss(prediction, val_targets))
            ranking = _ranking_metrics(val_returns, prediction, args.tie_eps)
        history.append({"epoch": epoch, "validation_mse": value_loss, **ranking})
        if value_loss < best:
            best = value_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    payload["influence_model"] = model.state_dict()
    payload["influence_optimizer"] = optimizer.state_dict()
    payload["offline_influence_training"] = {
        "diagnostics": str(args.diagnostics), "states": int(len(outcomes)), "train_states": int((~validation).sum()),
        "validation_states": int(validation.sum()), "best_validation_mse": best, "best_epoch": int(np.argmin([row["validation_mse"] for row in history])),
    }
    args.output_sidecar.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output_sidecar)
    report = {"split_groups": {"train": sorted(map(int, np.unique(groups[~validation]))), "validation": sorted(map(int, np.unique(groups[validation])))}, "history": history, **payload["offline_influence_training"]}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

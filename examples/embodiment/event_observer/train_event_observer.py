#!/usr/bin/env python3
"""Train the frozen-perception Role-Graph Event Observer and Event Value.

Input files are produced by the SAM cache stage, one ``.npz`` per episode.
They contain *only* derived RGB track features plus replay-readback proprio;
oracle fields are targets.  Keeping this trainer independent from SAM makes
PPO able to load the small observer checkpoint without importing SAM.
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from rlinf.models.embodiment.event_observer import (
    EventObserver,
    EventValueCritic,
    event_observer_supervision_loss,
)


@dataclass(frozen=True)
class Episode:
    features: torch.Tensor
    mount_tokens: torch.Tensor
    posterior_target: torch.Tensor
    state_target: torch.Tensor
    boundary_target: torch.Tensor
    progress_target: torch.Tensor
    geometric_relation_target: torch.Tensor
    state_change_target: torch.Tensor
    valid_mask: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor


class CachedEpisodes(Dataset):
    """Validated visual-only feature caches; no simulator state is accepted."""

    REQUIRED = {
        "features", "mount_tokens", "posterior_target", "state_target",
        "boundary_target", "progress_target", "geometric_relation_target",
        "state_change_target", "valid_mask", "rewards", "dones",
    }

    def __init__(self, root: Path, holdout_tasks: set[str], validation: bool) -> None:
        self.paths = []
        for path in sorted(root.rglob("*.npz")):
            with np.load(path, allow_pickle=False) as data:
                missing = self.REQUIRED.difference(data.files)
                if missing:
                    raise ValueError(f"{path} lacks cache fields: {sorted(missing)}")
                task = str(data.get("task", ""))
            if (task in holdout_tasks) == validation:
                self.paths.append(path)
        if not self.paths:
            split = "validation" if validation else "training"
            raise ValueError(f"no {split} episodes after task-level split")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Episode:
        with np.load(self.paths[index], allow_pickle=False) as data:
            values = {key: torch.from_numpy(np.asarray(data[key])) for key in self.REQUIRED}
        features = values["features"].float()
        if features.ndim != 2 or len(features) < 2:
            raise ValueError(f"invalid feature sequence in {self.paths[index]}")
        if any(value.shape[0] != len(features) for value in values.values()):
            raise ValueError(f"unaligned feature/label sequence in {self.paths[index]}")
        return Episode(**values)


def collate(episodes: list[Episode]) -> dict[str, torch.Tensor]:
    max_steps = max(len(episode.features) for episode in episodes)
    output: dict[str, list[torch.Tensor]] = {}
    for episode in episodes:
        for key, value in asdict(episode).items():
            fill = 0 if value.dtype != torch.bool else False
            padded = torch.full((max_steps, *value.shape[1:]), fill, dtype=value.dtype)
            padded[: len(value)] = value
            output.setdefault(key, []).append(padded)
    batch = {key: torch.stack(value) for key, value in output.items()}
    batch["lengths"] = torch.tensor([len(episode.features) for episode in episodes])
    return batch


def discounted_returns(rewards: torch.Tensor, dones: torch.Tensor, gamma: float) -> torch.Tensor:
    """Monte-Carlo Event-Value targets from cached expert replay rewards."""
    returns = torch.zeros_like(rewards, dtype=torch.float32)
    future = torch.zeros(rewards.shape[0], dtype=torch.float32, device=rewards.device)
    for step in range(rewards.shape[1] - 1, -1, -1):
        future = rewards[:, step].float() + gamma * future * (~dones[:, step].bool())
        returns[:, step] = future
    return returns


def evaluate(
    observer: EventObserver, critic: EventValueCritic, loader: DataLoader, device: torch.device, gamma: float
) -> dict[str, float]:
    observer.eval(); critic.eval()
    totals = {"loss": 0.0, "boundary_correct": 0.0, "boundary_count": 0.0, "progress_abs": 0.0, "value_abs": 0.0}
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            prediction = observer(batch["features"].float(), batch["mount_tokens"].long())
            losses = event_observer_supervision_loss(prediction, **{key: batch[key] for key in (
                "posterior_target", "state_target", "boundary_target", "progress_target",
                "geometric_relation_target", "state_change_target", "valid_mask",
            )})
            values = critic(prediction.representation)
            returns = discounted_returns(batch["rewards"], batch["dones"], gamma)
            mask = batch["valid_mask"].bool()
            totals["loss"] += float(sum(losses.values()).item())
            totals["boundary_correct"] += float(((prediction.boundary_logits.sigmoid() >= .5) == batch["boundary_target"].bool())[mask].sum())
            totals["boundary_count"] += float(mask.sum())
            totals["progress_abs"] += float((prediction.progress.sub(batch["progress_target"]).abs()[mask]).sum())
            totals["value_abs"] += float((values.sub(returns).abs()[mask]).sum())
    count = max(len(loader), 1); frames = max(totals["boundary_count"], 1.0)
    return {"loss": totals["loss"] / count, "boundary_accuracy": totals["boundary_correct"] / frames,
            "progress_mae": totals["progress_abs"] / frames, "event_value_mae": totals["value_abs"] / frames}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--holdout-tasks", default="")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=.99)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    held_out = {task for task in args.holdout_tasks.split(",") if task}
    train = CachedEpisodes(args.cache, held_out, validation=False)
    validation = CachedEpisodes(args.cache, held_out, validation=True) if held_out else None
    feature_dim = train[0].features.shape[-1]
    posterior_dim = train[0].posterior_target.shape[-1]
    state_dim = int(max(item.state_target.max().item() for item in (train[index] for index in range(len(train)))) + 1)
    geometric_dim = train[0].geometric_relation_target.shape[-1]
    state_change_dim = train[0].state_change_target.shape[-1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    observer = EventObserver(feature_dim, posterior_dim, state_dim, num_geometric_primitives=geometric_dim,
                             num_state_change_primitives=state_change_dim).to(device)
    critic = EventValueCritic(256).to(device)
    optimizer = torch.optim.AdamW([*observer.parameters(), *critic.parameters()], lr=args.lr)
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=4, pin_memory=True)
    val_loader = DataLoader(validation, batch_size=args.batch_size, collate_fn=collate, num_workers=2) if validation else None
    args.output.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    for epoch in range(args.epochs):
        observer.train(); critic.train(); epoch_loss = 0.0
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            prediction = observer(batch["features"].float(), batch["mount_tokens"].long())
            losses = event_observer_supervision_loss(prediction, **{key: batch[key] for key in (
                "posterior_target", "state_target", "boundary_target", "progress_target",
                "geometric_relation_target", "state_change_target", "valid_mask",
            )})
            mask = batch["valid_mask"].bool()
            value_loss = nn.functional.huber_loss(critic(prediction.representation)[mask],
                                                  discounted_returns(batch["rewards"], batch["dones"], args.gamma)[mask])
            loss = sum(losses.values()) + value_loss
            optimizer.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_([*observer.parameters(), *critic.parameters()], 1.0); optimizer.step()
            epoch_loss += float(loss.item())
        metrics = {"epoch": epoch, "train_loss": epoch_loss / max(len(loader), 1)}
        if val_loader:
            metrics.update({f"val_{key}": value for key, value in evaluate(observer, critic, val_loader, device, args.gamma).items()})
        print(json.dumps(metrics, sort_keys=True), flush=True)
        score = metrics.get("val_loss", metrics["train_loss"])
        if score < best:
            best = score
            torch.save({"observer": observer.state_dict(), "event_value": critic.state_dict(), "feature_dim": feature_dim,
                        "posterior_dim": posterior_dim, "state_dim": state_dim, "geometric_dim": geometric_dim,
                        "state_change_dim": state_change_dim, "metrics": metrics}, args.output / "best.pt")


if __name__ == "__main__":
    main()

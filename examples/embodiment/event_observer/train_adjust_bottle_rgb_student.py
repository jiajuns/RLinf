#!/usr/bin/env python3
"""Distil offline SAM teacher features into an online RGB role frontend."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from rlinf.models.embodiment.event_observer import RGBRoleFeatureStudent


class CachedRGBEpisodes(Dataset):
    def __init__(self, root: Path, *, validation: bool, fraction: float) -> None:
        self.paths: list[Path] = []
        for path in sorted(root.glob("*.npz")):
            with np.load(path, allow_pickle=False) as data:
                if "source_episode" not in data or "features" not in data:
                    continue
                bucket = int(hashlib.sha256(path.name.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
            if (bucket < fraction) == validation:
                self.paths.append(path)
        if not self.paths:
            raise ValueError("no RGB student episodes for requested split")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with np.load(self.paths[index], allow_pickle=False) as data:
            source = Path(str(data["source_episode"]))
            target = torch.from_numpy(np.asarray(data["features"], np.float32))
        with h5py.File(source, "r") as handle:
            head = torch.from_numpy(np.asarray(handle["rgb"]["head_camera"], np.uint8))
            wrist = torch.from_numpy(np.asarray(handle["rgb"]["right_camera"], np.uint8))
            measured_state16 = torch.from_numpy(np.asarray(handle["ee_state16"], np.float32))
        if len(head) != len(wrist) or len(head) != len(target) or len(measured_state16) != len(target):
            raise ValueError(f"unaligned RGB/teacher cache: {self.paths[index]}")
        return head, wrist, measured_state16, target


def collate(rows: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    maximum = max(len(target) for _, _, _, target in rows)
    head, wrist, measured_state16, target, mask = [], [], [], [], []
    for current_head, current_wrist, current_state, current_target in rows:
        current_steps = len(current_target)
        padded_head = torch.zeros((maximum, *current_head.shape[1:]), dtype=current_head.dtype)
        padded_wrist = torch.zeros((maximum, *current_wrist.shape[1:]), dtype=current_wrist.dtype)
        padded_target = torch.zeros((maximum, current_target.shape[-1]), dtype=current_target.dtype)
        padded_state = torch.zeros((maximum, 16), dtype=current_state.dtype)
        padded_head[:current_steps], padded_wrist[:current_steps], padded_target[:current_steps], padded_state[:current_steps] = current_head, current_wrist, current_target, current_state
        head.append(padded_head); wrist.append(padded_wrist); measured_state16.append(padded_state); target.append(padded_target)
        mask.append(torch.arange(maximum) < current_steps)
    return torch.stack(head), torch.stack(wrist), torch.stack(measured_state16), torch.stack(target), torch.stack(mask)


def evaluate(model: RGBRoleFeatureStudent, loader: DataLoader, device: torch.device, proprio_time_delta: float) -> float:
    model.eval(); total = count = 0.0
    with torch.no_grad():
        for head, wrist, measured_state16, target, mask in loader:
            predicted = model(head.to(device), wrist.to(device), measured_state16.to(device), proprio_time_delta=proprio_time_delta)
            error = (predicted - target.to(device)).abs()[mask.to(device)].mean()
            total += float(error); count += 1
    return total / max(count, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--validation-fraction", type=float, default=.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--proprio-time-delta", type=float, default=1.0)
    parser.add_argument("--online-mount-token", type=int, default=2)
    args = parser.parse_args()
    if not 0 < args.validation_fraction < 1:
        raise ValueError("validation fraction must be in (0,1)")
    if args.proprio_time_delta <= 0:
        raise ValueError("--proprio-time-delta must be positive")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    train = CachedRGBEpisodes(args.cache, validation=False, fraction=args.validation_fraction)
    validation = CachedRGBEpisodes(args.cache, validation=True, fraction=args.validation_fraction)
    feature_dim = train[0][3].shape[-1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RGBRoleFeatureStudent(feature_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=collate, pin_memory=True)
    val_loader = DataLoader(validation, batch_size=args.batch_size, num_workers=1, collate_fn=collate)
    args.output.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    for epoch in range(args.epochs):
        model.train(); total = 0.0
        for head, wrist, measured_state16, target, mask in train_loader:
            prediction = model(head.to(device), wrist.to(device), measured_state16.to(device), proprio_time_delta=args.proprio_time_delta)
            loss = F.smooth_l1_loss(prediction[mask.to(device)], target.to(device)[mask.to(device)])
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            total += float(loss.detach())
        validation_mae = evaluate(model, val_loader, device, args.proprio_time_delta)
        metrics = {"epoch": epoch, "train_huber": total / max(len(train_loader), 1), "val_teacher_feature_mae": validation_mae}
        print(json.dumps(metrics), flush=True)
        if validation_mae < best:
            best = validation_mae
            torch.save({"rgb_student": model.state_dict(), "feature_dim": feature_dim, "metrics": metrics,
                        "online_input_contract": {"version": 1, "proprio_time_delta": args.proprio_time_delta,
                                                  "mount_token": args.online_mount_token}}, args.output / "best.pt")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate the deployment path ``RGB student -> frozen Event Observer``.

The cache-feature Observer score alone is not an online result: PPO receives
fresh RGB and measured proprioception, not the SAM-teacher feature vectors in
the Zarr/NPZ cache.  This diagnostic evaluates both paths on *the same held
out cached episodes*, with the same causal history and mount token, and never
uses an oracle feature as an input to the student path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from rlinf.models.embodiment.event_observer import EventObserver, EventValueCritic, RGBRoleFeatureStudent


def _selected_paths(root: Path, fraction: float) -> list[Path]:
    paths = []
    for path in sorted(root.glob("*.npz")):
        with np.load(path, allow_pickle=False) as payload:
            required = {"features", "mount_tokens", "boundary_target", "progress_target", "rewards", "dones", "valid_mask", "source_episode"}
            if not required.issubset(payload.files):
                continue
        bucket = int(hashlib.sha256(path.name.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        if bucket < fraction:
            paths.append(path)
    if not paths:
        raise ValueError("no held-out cache episodes")
    return paths


def _returns(rewards: torch.Tensor, dones: torch.Tensor, gamma: float) -> torch.Tensor:
    output = torch.zeros_like(rewards, dtype=torch.float32)
    future = torch.zeros((), dtype=torch.float32)
    for step in range(len(rewards) - 1, -1, -1):
        future = rewards[step].float() + gamma * future * (~dones[step].bool())
        output[step] = future
    return output


def _f1(prediction: torch.Tensor, target: torch.Tensor, tolerance: int = 0) -> tuple[int, int, int]:
    predicted = prediction.nonzero(as_tuple=False).flatten().tolist()
    truth = target.nonzero(as_tuple=False).flatten().tolist()
    unmatched = set(truth); tp = 0
    for item in predicted:
        candidates = [candidate for candidate in unmatched if abs(item - candidate) <= tolerance]
        if candidates:
            unmatched.remove(min(candidates, key=lambda candidate: abs(item - candidate)))
            tp += 1
    return tp, len(predicted) - tp, len(truth) - tp


def _score(accumulator: dict[str, float]) -> dict[str, float]:
    def f1(prefix: str) -> dict[str, float]:
        precision = accumulator[f"{prefix}_tp"] / max(accumulator[f"{prefix}_tp"] + accumulator[f"{prefix}_fp"], 1.0)
        recall = accumulator[f"{prefix}_tp"] / max(accumulator[f"{prefix}_tp"] + accumulator[f"{prefix}_fn"], 1.0)
        return {f"{prefix}_precision": precision, f"{prefix}_recall": recall, f"{prefix}_f1": 2 * precision * recall / max(precision + recall, 1e-12)}
    frames = max(accumulator["frames"], 1.0)
    return {
        **f1("boundary"), **f1("boundary_at_3"),
        "progress_mae": accumulator["progress_abs"] / frames,
        "event_value_mae": accumulator["value_abs"] / frames,
        "frames": accumulator["frames"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--observer", type=Path, required=True)
    parser.add_argument("--rgb-student", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=.1)
    parser.add_argument("--gamma", type=float, default=.99)
    args = parser.parse_args()
    observer_payload = torch.load(args.observer, map_location="cpu", weights_only=False)
    student_payload = torch.load(args.rgb_student, map_location="cpu", weights_only=False)
    observer_contract = observer_payload["online_input_contract"]
    student_contract = student_payload["online_input_contract"]
    for key in ("proprio_time_delta", "mount_token", "control_step_stride", "proprio_derivative_unit"):
        if observer_contract[key] != student_contract[key]:
            raise ValueError(f"online input contract mismatch for {key}: observer={observer_contract[key]} student={student_contract[key]}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    observer = EventObserver(
        int(observer_payload["feature_dim"]), int(observer_payload["posterior_dim"]), int(observer_payload["state_dim"]),
        num_geometric_primitives=int(observer_payload["geometric_dim"]),
        num_state_change_primitives=int(observer_payload["state_change_dim"]),
    ).to(device).eval()
    observer.load_state_dict(observer_payload["observer"])
    critic = EventValueCritic(256).to(device).eval(); critic.load_state_dict(observer_payload["event_value"])
    student = RGBRoleFeatureStudent(int(student_payload["feature_dim"])).to(device).eval(); student.load_state_dict(student_payload["rgb_student"])
    paths = _selected_paths(args.cache, args.validation_fraction)
    totals = {name: {"boundary_tp": 0.0, "boundary_fp": 0.0, "boundary_fn": 0.0, "boundary_at_3_tp": 0.0, "boundary_at_3_fp": 0.0, "boundary_at_3_fn": 0.0, "progress_abs": 0.0, "value_abs": 0.0, "frames": 0.0} for name in ("teacher", "online")}
    distillation = {"feature_abs": 0.0, "representation_abs": 0.0, "value_abs": 0.0, "frames": 0.0}
    with torch.no_grad():
        for path in paths:
            with np.load(path, allow_pickle=False) as data:
                stride = int(data["control_step_stride"])
                if stride != int(observer_contract["control_step_stride"]):
                    raise ValueError(f"cache stride mismatch: {path}")
                source = Path(str(data["source_episode"])); indices = np.asarray(data.get("frame_indices", np.arange(len(data["features"]))), np.int64)
                feature = torch.from_numpy(np.asarray(data["features"], np.float32)).unsqueeze(0).to(device)
                mount = torch.from_numpy(np.asarray(data["mount_tokens"], np.int64)).unsqueeze(0).to(device)
                boundary = torch.from_numpy(np.asarray(data["boundary_target"], bool)).to(device)
                progress = torch.from_numpy(np.asarray(data["progress_target"], np.float32)).to(device)
                valid = torch.from_numpy(np.asarray(data["valid_mask"], bool)).to(device)
                returns = _returns(torch.from_numpy(np.asarray(data["rewards"], np.float32)).to(device), torch.from_numpy(np.asarray(data["dones"], bool)).to(device), args.gamma)
            with h5py.File(source, "r") as handle:
                head = torch.from_numpy(np.asarray(handle["rgb"]["head_camera"][indices], np.uint8)).unsqueeze(0).to(device)
                wrist = torch.from_numpy(np.asarray(handle["rgb"]["right_camera"][indices], np.uint8)).unsqueeze(0).to(device)
                state16 = torch.from_numpy(np.asarray(handle["ee_state16"][indices], np.float32)).unsqueeze(0).to(device)
            online_feature = student(head, wrist, state16, proprio_time_delta=float(observer_contract["proprio_time_delta"]))
            teacher = observer(feature, mount); online = observer(online_feature, mount)
            teacher_value, online_value = critic(teacher.representation).squeeze(0), critic(online.representation).squeeze(0)
            distillation["feature_abs"] += float((online_feature.squeeze(0).sub(feature.squeeze(0)).abs()[valid]).sum())
            distillation["representation_abs"] += float((online.representation.squeeze(0).sub(teacher.representation.squeeze(0)).abs()[valid]).sum())
            distillation["value_abs"] += float((online_value.sub(teacher_value).abs()[valid]).sum())
            distillation["frames"] += float(valid.sum())
            for name, prediction, value in (("teacher", teacher, teacher_value), ("online", online, online_value)):
                predicted_boundary = prediction.boundary_logits.squeeze(0).sigmoid() >= .5
                p, fp, fn = _f1(predicted_boundary[valid], boundary[valid], tolerance=0)
                p3, fp3, fn3 = _f1(predicted_boundary[valid], boundary[valid], tolerance=3)
                totals[name]["boundary_tp"] += p; totals[name]["boundary_fp"] += fp; totals[name]["boundary_fn"] += fn
                totals[name]["boundary_at_3_tp"] += p3; totals[name]["boundary_at_3_fp"] += fp3; totals[name]["boundary_at_3_fn"] += fn3
                totals[name]["progress_abs"] += float((prediction.progress.squeeze(0).sub(progress).abs()[valid]).sum())
                totals[name]["value_abs"] += float((value.sub(returns).abs()[valid]).sum())
                totals[name]["frames"] += float(valid.sum())
    frames = max(distillation.pop("frames"), 1.0)
    result = {"episodes": len(paths), "online_input_contract": observer_contract, "teacher": _score(totals["teacher"]), "online_rgb_student": _score(totals["online"]), "teacher_to_online": {f"{key}_mae": value / frames for key, value in distillation.items()}}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

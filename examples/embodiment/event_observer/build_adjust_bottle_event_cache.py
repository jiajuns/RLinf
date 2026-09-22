#!/usr/bin/env python3
"""Fuse SAM teacher tracks with RGB/proprio-only Event Observer inputs.

The renderer sidecar supplies RGB and *label-only* oracle relations.  SAM's
offline track cache supplies the only object geometry admitted to features.
This conversion deliberately never copies object pose, target pose, contact,
segmentation or depth into the produced ``.npz`` training cache.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import h5py
import numpy as np

from prepare_robotwin_observer_dataset import (
    canonical_event_targets,
    event_boundary_and_progress,
    shared_relation_targets,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def causal_difference(values: np.ndarray, control_step_delta: int) -> np.ndarray:
    result = np.zeros_like(values, dtype=np.float32)
    if control_step_delta < 1:
        raise ValueError("control_step_delta must be positive")
    # Keep every velocity/opening derivative in *per control step* units.
    # Online PPO sees chunk boundaries separated by this same number of
    # controls, so using wall-clock sim_times offline would be a mismatch.
    result[1:] = np.diff(values, axis=0) / float(control_step_delta)
    return result


def make_features(
    tracks: np.ndarray, ee_state16: np.ndarray, times: np.ndarray, *, control_step_delta: int
) -> np.ndarray:
    """Build causal low-dimensional visual tracks plus measured proprioception."""
    if tracks.ndim != 3 or tracks.shape[2] != 6 or tracks.shape[0] < 1:
        raise ValueError("tracks must have shape [camera,time,6]")
    cameras, steps, _ = tracks.shape
    if ee_state16.shape != (steps, 16) or times.shape != (steps,):
        raise ValueError("SAM/proprio/timestamp sequences are not aligned")
    visual = []
    for track in tracks:
        velocity = causal_difference(track[:, :2], control_step_delta)
        missing = 1.0 - track[:, 5:6]
        # xywh, confidence, visibility, causal xy velocity, missing marker.
        visual.append(np.concatenate((track, velocity, missing), axis=-1))
    # Measurements are read after physics stepping.  For Aloha the collector
    # exposes gripper travel already normalized; clip only protects corrupt
    # frames and avoids deriving a future-dependent per-episode normalization.
    opening = np.clip(ee_state16[:, (7, 15)], 0.0, 1.0).astype(np.float32)
    opening_velocity = causal_difference(opening, control_step_delta)
    pose = np.stack((ee_state16[:, :7], ee_state16[:, 8:15]), axis=1)
    ee_linear = causal_difference(pose[:, :, :3], control_step_delta)
    ee_speed = np.linalg.norm(ee_linear, axis=-1)
    return np.concatenate(
        [*visual, opening, opening_velocity, ee_linear.reshape(steps, -1), ee_speed], axis=-1
    ).astype(np.float32)


def _chunk_indices(steps: int, stride: int) -> np.ndarray:
    if steps < 1 or stride < 1:
        raise ValueError("episode length and control-step stride must be positive")
    indices = np.arange(0, steps, stride, dtype=np.int64)
    # A success/terminal label often occurs in a short final chunk.  Dropping
    # that frame loses the only sparse reward in many expert episodes and
    # makes a non-divisible trajectory look as if it ended normally at the
    # previous full chunk boundary.
    if indices[-1] != steps - 1:
        indices = np.append(indices, np.int64(steps - 1))
    return indices


def convert(episode: Path, track_path: Path, output: Path, *, control_step_stride: int = 1) -> None:
    with np.load(track_path, allow_pickle=False) as track_data:
        if str(track_data["format"]) != "event_sam31_track_v1":
            raise ValueError("unsupported SAM cache format")
        if str(track_data["episode_sha256"]) != sha256(episode):
            raise ValueError("SAM track cache belongs to a different episode")
        tracks = np.asarray(track_data["tracks"], np.float32)
        cameras = tuple(map(str, track_data["cameras"]))
        track_hash = sha256(track_path)
    with h5py.File(episode, "r") as handle:
        times = np.asarray(handle["sim_times"], np.float64)
        ee = np.asarray(handle["ee_state16"], np.float32)
        relations = np.asarray(handle["relations"], np.float32)
        nodes = np.asarray(handle["node_features"], np.float32)
        success = np.asarray(handle["success"], bool)
        roster = str(handle.attrs["node_roster_json"])
        task = str(handle.attrs["task"])
    if len(times) != tracks.shape[1]:
        raise ValueError("SAM track length does not match oracle sidecar")
    if np.any(np.diff(times) <= 0):
        raise ValueError("sim_times must be strictly increasing")
    posterior_raw, state_raw = canonical_event_targets(relations, nodes, success, roster)
    boundary_raw, progress_raw = event_boundary_and_progress(posterior_raw)
    geometric_raw, state_change_raw = shared_relation_targets(relations, roster)
    rewards_raw = np.zeros(len(times), np.float32)
    rewards_raw[1:] = (success[1:] & ~success[:-1]).astype(np.float32)
    if len(rewards_raw) and success[0]:
        rewards_raw[0] = 1.0
    indices = _chunk_indices(len(times), control_step_stride)
    # Coarse boundaries/rewards summarize the transition since the preceding
    # chunk boundary, whereas state/progress describe the current boundary.
    boundary = np.zeros(len(indices), np.float32)
    rewards = np.zeros(len(indices), np.float32)
    for coarse_idx, source_idx in enumerate(indices):
        begin = 0 if coarse_idx == 0 else int(indices[coarse_idx - 1]) + 1
        boundary[coarse_idx] = boundary_raw[begin : source_idx + 1].max()
        rewards[coarse_idx] = rewards_raw[begin : source_idx + 1].sum()
    tracks = tracks[:, indices]
    ee = ee[indices]
    selected_times = times[indices]
    posterior = posterior_raw[indices]
    state = state_raw[indices]
    progress = progress_raw[indices]
    geometric = geometric_raw[indices]
    state_change = state_change_raw[indices]
    dones = np.zeros(len(indices), bool); dones[-1] = True
    # This cache is consumed as one sample per policy chunk.  Fail at the
    # producer if a future refactor leaves any supervision field at raw video
    # rate: accepting it would silently train the Observer on mismatched
    # feature/label pairs.
    chunk_fields = {
        "tracks": tracks,
        "ee_state16": ee,
        "posterior": posterior,
        "state": state,
        "progress": progress,
        "geometric": geometric,
        "state_change": state_change,
        "boundary": boundary,
        "rewards": rewards,
        "dones": dones,
    }
    wrong = {name: len(value) for name, value in chunk_fields.items() if len(value) != len(indices)}
    if wrong:
        raise RuntimeError(f"chunk cache fields are misaligned: expected {len(indices)}, got {wrong}")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        format=np.asarray("adjust_bottle_event_cache_v2"),
        episode_sha256=np.asarray(sha256(episode)), track_sha256=np.asarray(track_hash),
        # Path only: it enables the RGB student trainer to read the same
        # renderer frames.  It is not an online cache lookup and contains no
        # privileged simulator state.
        source_episode=np.asarray(str(episode.resolve())),
        task=np.asarray(task), cameras=np.asarray(cameras),
        features=make_features(tracks, ee, selected_times, control_step_delta=control_step_stride),
        frame_indices=indices,
        control_step_stride=np.asarray(control_step_stride, np.int64),
        proprio_derivative_unit=np.asarray("per_control_step"),
        # The two cameras are fused into a single frame feature vector.  Token
        # 2 identifies this fixed global+wrist fusion, not an embodiment/task.
        # Every emitted field is indexed at the chunk boundary.  Keeping
        # these at raw-frame length silently misaligns training examples when
        # ``control_step_stride > 1`` (e.g. the 50-control-step π0.5 chunk).
        mount_tokens=np.full(len(indices), 2, np.int64),
        posterior_target=posterior.astype(np.float32), state_target=state.astype(np.int64),
        boundary_target=boundary.astype(np.float32), progress_target=progress.astype(np.float32),
        geometric_relation_target=geometric.astype(np.float32),
        state_change_target=state_change.astype(np.float32),
        valid_mask=np.ones(len(indices), bool), rewards=rewards, dones=dones,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control-step-stride", type=int, default=1)
    args = parser.parse_args()
    convert(args.episode, args.tracks, args.output, control_step_stride=args.control_step_stride)
    print({"cache": str(args.output), "episode": str(args.episode)})


if __name__ == "__main__":
    main()

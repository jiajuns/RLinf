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


def causal_difference(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    result = np.zeros_like(values, dtype=np.float32)
    delta = np.diff(times)
    if np.any(delta <= 0):
        raise ValueError("sim_times must be strictly increasing")
    result[1:] = np.diff(values, axis=0) / delta.reshape((-1,) + (1,) * (values.ndim - 1))
    return result


def make_features(tracks: np.ndarray, ee_state16: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Build causal low-dimensional visual tracks plus measured proprioception."""
    if tracks.ndim != 3 or tracks.shape[2] != 6 or tracks.shape[0] < 1:
        raise ValueError("tracks must have shape [camera,time,6]")
    cameras, steps, _ = tracks.shape
    if ee_state16.shape != (steps, 16) or times.shape != (steps,):
        raise ValueError("SAM/proprio/timestamp sequences are not aligned")
    visual = []
    for track in tracks:
        velocity = causal_difference(track[:, :2], times)
        missing = 1.0 - track[:, 5:6]
        # xywh, confidence, visibility, causal xy velocity, missing marker.
        visual.append(np.concatenate((track, velocity, missing), axis=-1))
    # Measurements are read after physics stepping.  For Aloha the collector
    # exposes gripper travel already normalized; clip only protects corrupt
    # frames and avoids deriving a future-dependent per-episode normalization.
    opening = np.clip(ee_state16[:, (7, 15)], 0.0, 1.0).astype(np.float32)
    opening_velocity = causal_difference(opening, times)
    pose = np.stack((ee_state16[:, :7], ee_state16[:, 8:15]), axis=1)
    ee_linear = causal_difference(pose[:, :, :3], times)
    ee_speed = np.linalg.norm(ee_linear, axis=-1)
    return np.concatenate(
        [*visual, opening, opening_velocity, ee_linear.reshape(steps, -1), ee_speed], axis=-1
    ).astype(np.float32)


def convert(episode: Path, track_path: Path, output: Path) -> None:
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
    posterior, state = canonical_event_targets(relations, nodes, success, roster)
    boundary, progress = event_boundary_and_progress(posterior)
    geometric, state_change = shared_relation_targets(relations, roster)
    rewards = np.zeros(len(times), np.float32)
    rewards[1:] = (success[1:] & ~success[:-1]).astype(np.float32)
    if len(rewards) and success[0]:
        rewards[0] = 1.0
    dones = np.zeros(len(times), bool); dones[-1] = True
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        format=np.asarray("adjust_bottle_event_cache_v1"),
        episode_sha256=np.asarray(sha256(episode)), track_sha256=np.asarray(track_hash),
        # Path only: it enables the RGB student trainer to read the same
        # renderer frames.  It is not an online cache lookup and contains no
        # privileged simulator state.
        source_episode=np.asarray(str(episode.resolve())),
        task=np.asarray(task), cameras=np.asarray(cameras),
        features=make_features(tracks, ee, times),
        # The two cameras are fused into a single frame feature vector.  Token
        # 2 identifies this fixed global+wrist fusion, not an embodiment/task.
        mount_tokens=np.full(len(times), 2, np.int64),
        posterior_target=posterior.astype(np.float32), state_target=state.astype(np.int64),
        boundary_target=boundary.astype(np.float32), progress_target=progress.astype(np.float32),
        geometric_relation_target=geometric.astype(np.float32),
        state_change_target=state_change.astype(np.float32),
        valid_mask=np.ones(len(times), bool), rewards=rewards, dones=dones,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    convert(args.episode, args.tracks, args.output)
    print({"cache": str(args.output), "episode": str(args.episode)})


if __name__ == "__main__":
    main()

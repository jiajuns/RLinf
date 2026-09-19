#!/usr/bin/env python3
"""Materialize paired RobotWin RGB/proprio/oracle-event episodes.

The official RobotWin archives remain the authoritative RGB source.  A paired
``collect_robotwin.py --scene-backend renderer --replay-archive ...`` sidecar
is the authoritative source of privileged event labels.  This tool refuses to
materialize a label file without its replay provenance, so visual samples can
never silently be paired with a different expert trajectory.

The resulting HDF5 contains only deployment-time inputs: JPEG camera frames and
post-physics robot proprioception read back during replay.  It does not copy
camera extrinsics, object state, depth, oracle segmentation, contact state, or
actions.  The oracle event targets remain labels, never observer inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import h5py
import numpy as np


FORMAT = "robotwin_rgb_proprio_oracle_event_observer_v2"
EVENT_NAMES = (
    "approach",
    "grasp",
    "lift",
    "transport",
    "align",
    "place",
    "release",
    "failure",
    "idle",
)
RELATION_NAMES = (
    "near",
    "held_by",
    "supported_by",
    "inside",
    "on_top_of",
    "open",
    "pressed",
    "activated",
    "released",
    "dropped",
    "lifted",
    "away_from",
)
ARCHIVE_HASH_CACHE: dict[Path, str] = {}


def discover_cameras(source: h5py.File) -> tuple[str, ...]:
    """Return available RGB cameras in a stable role-first order.

    RoboTwin archives do not promise a ``front_camera``.  Camera names are
    therefore discovered from the recorded HDF5 rather than inferred from an
    embodiment convention.  The three names below are ordered first because
    they identify the two wrist views and the fixed global view when present.
    """
    observation = source.get("observation")
    if not isinstance(observation, h5py.Group):
        raise ValueError("official episode has no observation group")
    available = [
        name
        for name, group in observation.items()
        if isinstance(group, h5py.Group) and "rgb" in group
    ]
    preferred = ("left_camera", "right_camera", "head_camera")
    ordered = [name for name in preferred if name in available]
    ordered.extend(sorted(name for name in available if name not in ordered))
    if not ordered:
        raise ValueError("official episode has no RGB camera stream")
    return tuple(ordered)


def replay_proprio(ee_state16: np.ndarray, sim_times: np.ndarray) -> dict[str, np.ndarray]:
    """Split replay-readback state into causal deployment-time proprioception.

    ``ee_state16`` is captured *after physics stepping* by the replay
    collector.  Its layout is left pose (7), left gripper, right pose (7), and
    right gripper.  Velocities are one-sided finite differences, so no feature
    at time ``t`` reads a future state.
    """
    state = np.asarray(ee_state16, dtype=np.float32)
    times = np.asarray(sim_times, dtype=np.float64)
    if state.ndim != 2 or state.shape[1] != 16 or len(state) != len(times):
        raise ValueError("replay ee_state16 must have shape [T,16] aligned to sim_times")
    if len(times) < 2 or not np.isfinite(state).all() or not np.isfinite(times).all():
        raise ValueError("replay proprioception must be finite and contain two frames")
    delta_t = np.diff(times)
    if np.any(delta_t <= 0):
        raise ValueError("replay sim_times must be strictly increasing")
    poses = np.stack((state[:, :7], state[:, 8:15]), axis=1)
    opening = np.stack((state[:, 7], state[:, 15]), axis=1)
    linear_velocity = np.zeros((len(state), 2, 3), dtype=np.float32)
    linear_velocity[1:] = np.diff(poses[:, :, :3], axis=0) / delta_t[:, None, None]
    # Quaternion conversion is deliberately left to a validated embodiment
    # adapter.  Storing poses lets the adapter compute angular velocity in its
    # documented convention instead of assuming a quaternion layout here.
    return {
        "ee_pose7": poses,
        "gripper_opening_raw": opening,
        "ee_linear_velocity": linear_velocity,
    }


def sha256_file(path: Path) -> str:
    """Return a file's SHA256 without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def archive_sha256(path: Path) -> str:
    """Hash each multi-GB official archive once per materialization process."""
    resolved = path.resolve()
    if resolved not in ARCHIVE_HASH_CACHE:
        ARCHIVE_HASH_CACHE[resolved] = sha256_file(resolved)
    return ARCHIVE_HASH_CACHE[resolved]


def source_hdf5_member(archive: zipfile.ZipFile, index: int) -> str:
    """Find exactly one official RGB HDF5 member for a replay index."""
    suffix = f"/data/episode{index}.hdf5"
    matches = [name for name in archive.namelist() if name.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"expected one {suffix} in {archive.filename}, found {len(matches)}")
    return matches[0]


def alignment_indices(sim_times: np.ndarray, source_length: int) -> np.ndarray:
    """Align label samples to uniformly sampled official recorded RGB frames."""
    if source_length < 2 or len(sim_times) < 2:
        raise ValueError("both RGB and oracle sequence must contain at least two frames")
    elapsed = np.asarray(sim_times, np.float64) - float(sim_times[0])
    duration = float(elapsed[-1])
    if not np.isfinite(elapsed).all() or duration <= 0:
        raise ValueError("oracle sim_times must be finite and strictly increasing")
    return np.rint(elapsed / duration * (source_length - 1)).astype(np.int64).clip(0, source_length - 1)


def event_boundary_and_progress(events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Derive label-only transition boundaries and within-segment progress.

    ``events`` is multi-label because RobotWin can express, for example,
    transport and grasp evidence concurrently.  Boundaries are changes in that
    complete posterior target, not a hand-coded clock.  Progress increases only
    inside an unchanged posterior segment and resets at every boundary.
    """
    values = np.asarray(events, np.float32)
    if values.ndim != 2 or values.shape[1] != len(EVENT_NAMES):
        raise ValueError(f"events must have shape [T,{len(EVENT_NAMES)}]")
    binary = values >= 0.5
    boundary = np.empty(len(binary), dtype=np.bool_)
    boundary[0] = True
    boundary[1:] = np.any(binary[1:] != binary[:-1], axis=1)
    progress = np.zeros(len(binary), dtype=np.float32)
    start = 0
    for end in np.r_[np.flatnonzero(boundary[1:]) + 1, len(binary)]:
        width = int(end - start)
        progress[start:end] = np.linspace(0.0, 1.0, width, dtype=np.float32)
        start = int(end)
    return boundary, progress


def canonical_event_targets(
    relations: np.ndarray,
    node_features: np.ndarray,
    success: np.ndarray,
    roster_json: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert replay oracle relations into exclusive Event-SMDP states.

    This conversion runs only while materializing labels.  Object position,
    contact-derived ``held_by`` and target membership are never written as
    observer inputs.  ``lift`` is the vertically dominant part of a held,
    lifted motion; the subsequent held motion is ``transport`` until target
    alignment.  ``release`` remains a posterior target but is not allowed to
    overwrite a terminal place/failure state.
    """
    relation_values = np.asarray(relations, dtype=np.float32)
    nodes = np.asarray(node_features, dtype=np.float32)
    succeeded = np.asarray(success, dtype=bool)
    if relation_values.ndim != 4 or relation_values.shape[1:] != (len(RELATION_NAMES), 8, 8):
        raise ValueError("legacy replay relations must have shape [T,12,8,8]")
    if nodes.shape != (len(relation_values), 8, 24) or succeeded.shape != (len(relation_values),):
        raise ValueError("replay oracle arrays are not time aligned")
    roster = json.loads(roster_json)
    index = {str(item["name"]): offset for offset, item in enumerate(roster)}
    moving, target = index.get("moving"), index.get("target")
    if moving is None:
        raise ValueError("replay roster has no moving object for Event-SMDP labels")
    grippers = [
        offset
        for offset, item in enumerate(roster)
        if str(item.get("type")) in {"left_gripper", "right_gripper"}
    ]
    if not grippers:
        raise ValueError("replay roster has no gripper for Event-SMDP labels")
    relation = {name: relation_values[:, offset] >= 0.5 for offset, name in enumerate(RELATION_NAMES)}
    held = relation["held_by"][:, moving, grippers].any(axis=1)
    lifted = relation["lifted"][:, moving, moving]
    released = relation["released"][:, moving, moving]
    dropped = relation["dropped"][:, moving, moving]
    placed = np.zeros(len(relation_values), dtype=bool)
    aligned = np.zeros(len(relation_values), dtype=bool)
    if target is not None:
        aligned = relation["near"][:, moving, target]
        placed = (
            relation["supported_by"][:, moving, target]
            | relation["inside"][:, moving, target]
            | relation["on_top_of"][:, moving, target]
        )
    object_velocity = np.zeros((len(nodes), 3), dtype=np.float32)
    object_velocity[1:] = np.diff(nodes[:, moving, :3], axis=0)
    vertical_lift = held & lifted & (
        (object_velocity[:, 2] > 0.0)
        & (object_velocity[:, 2] >= np.linalg.norm(object_velocity[:, :2], axis=1))
    )
    state = np.full(len(nodes), EVENT_NAMES.index("approach"), dtype=np.int64)
    state[held & ~lifted] = EVENT_NAMES.index("grasp")
    state[held & lifted] = EVENT_NAMES.index("transport")
    state[vertical_lift] = EVENT_NAMES.index("lift")
    state[held & lifted & aligned] = EVENT_NAMES.index("align")
    state[placed | succeeded] = EVENT_NAMES.index("place")
    state[dropped] = EVENT_NAMES.index("failure")
    target_values = np.zeros((len(nodes), len(EVENT_NAMES)), dtype=np.float32)
    target_values[np.arange(len(nodes)), state] = 1.0
    # Release is an event that can coexist with the terminal state.  It is a
    # supervised posterior head, while ``state`` stays exclusive for SMDP.
    target_values[released, EVENT_NAMES.index("release")] = 1.0
    return target_values, state


def read_label(path: Path, require_success: bool) -> dict[str, Any]:
    """Read only label fields and validate that the source is an official replay."""
    with h5py.File(path, "r") as handle:
        archive = Path(str(handle.attrs.get("replay_archive", "")))
        replay_index = int(handle.attrs.get("replay_index", -1))
        if not archive.is_file() or replay_index < 0:
            raise ValueError("label is not a replay-backed official-expert sidecar")
        success = bool(handle.attrs.get("native_success", False))
        if require_success and not success:
            raise ValueError("replayed expert did not satisfy native success")
        sim_times = np.asarray(handle["sim_times"], dtype=np.float64)
        if "ee_state16" not in handle:
            raise ValueError(
                "replay sidecar lacks ee_state16; rerun with post-physics proprioception capture"
            )
        proprio = replay_proprio(np.asarray(handle["ee_state16"]), sim_times)
        for name in ("relations", "node_features", "success"):
            if name not in handle:
                raise ValueError(f"replay sidecar lacks oracle label source {name}")
        event_target, event_state_id = canonical_event_targets(
            np.asarray(handle["relations"]),
            np.asarray(handle["node_features"]),
            np.asarray(handle["success"]),
            str(handle.attrs.get("node_roster_json", "")),
        )
        return {
            "archive": archive,
            "archive_sha256": str(handle.attrs.get("replay_archive_sha256", "")),
            "replay_index": replay_index,
            "event_target": event_target,
            "event_state_id": event_state_id,
            "sim_times": sim_times,
            "actions14": np.asarray(handle["actions14"], dtype=np.float32),
            "proprio": proprio,
            "task": str(handle.attrs["task"]),
            "body": str(handle.attrs["body"]),
            "condition": str(handle.attrs["condition"]),
            "seed": int(handle.attrs["seed"]),
            "native_success": success,
        }


def create_episode(label_path: Path, output_root: Path, require_success: bool) -> dict[str, Any]:
    """Write one visual-only observer episode and return its manifest row."""
    label = read_label(label_path, require_success)
    archive_path = label["archive"]
    source_hash = archive_sha256(archive_path)
    if label["archive_sha256"] and source_hash != label["archive_sha256"]:
        raise ValueError("official archive SHA256 differs from the replay sidecar")
    relative = Path(label["task"]) / label["body"] / label["condition"] / f"seed_{label['seed']}.hdf5"
    destination = output_root / "episodes" / relative
    if destination.exists():
        return {"status": "existing", "path": str(destination), "seed": label["seed"]}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        member = source_hdf5_member(archive, label["replay_index"])
        with h5py.File(io.BytesIO(archive.read(member)), "r") as source:
            cameras = discover_cameras(source)
            first_camera = source[f"observation/{cameras[0]}/rgb"]
            frame_index = alignment_indices(label["sim_times"], len(first_camera))
            if label["actions14"].shape != (len(frame_index) - 1, 14):
                raise ValueError("oracle action length does not match oracle labels")
            temporary = destination.with_suffix(".partial")
            if temporary.exists():
                temporary.unlink()
            try:
                with h5py.File(temporary, "w") as target:
                    jpeg_type = h5py.vlen_dtype(np.dtype("uint8"))
                    rgb_group = target.create_group("rgb")
                    for camera in cameras:
                        camera_group = source[f"observation/{camera}"]
                        rgb = camera_group["rgb"]
                        if len(rgb) != len(first_camera):
                            raise ValueError(f"inconsistent camera length for {camera}")
                        output = rgb_group.create_dataset(camera, (len(frame_index),), dtype=jpeg_type)
                        for offset, source_index in enumerate(frame_index):
                            output[offset] = np.frombuffer(bytes(rgb[int(source_index)]), dtype=np.uint8)
                    proprio_group = target.create_group("proprio")
                    for key, value in label["proprio"].items():
                        proprio_group.create_dataset(
                            key, data=value, compression="gzip", compression_opts=1
                        )
                    boundary, progress = event_boundary_and_progress(label["event_target"])
                    target.create_dataset("frame_index", data=frame_index)
                    target.create_dataset("event_target", data=label["event_target"], compression="gzip", compression_opts=1)
                    target.create_dataset("event_state_id", data=label["event_state_id"])
                    target.create_dataset("event_boundary", data=boundary)
                    target.create_dataset("event_progress", data=progress)
                    target.attrs.update({
                        "format": FORMAT,
                        "task": label["task"], "body": label["body"],
                        "condition": label["condition"], "seed": label["seed"],
                        "native_success": label["native_success"],
                        "source_archive": str(archive_path),
                        "source_archive_sha256": source_hash,
                        "source_member": member,
                        "source_replay_index": label["replay_index"],
                        "oracle_sidecar": str(label_path),
                        "oracle_sidecar_sha256": sha256_file(label_path),
                        "event_names_json": json.dumps(EVENT_NAMES),
                        "camera_names_json": json.dumps(cameras),
                        "proprio_source": "post_physics_replay_readback",
                        "forbidden_observer_inputs_json": json.dumps([
                            "actions14", "camera_extrinsics", "camera_intrinsics",
                            "depth", "oracle_segmentation", "object_pose",
                            "contact", "oracle_event_target",
                        ]),
                    })
                os.replace(temporary, destination)
            finally:
                if temporary.exists():
                    temporary.unlink()
    return {
        "status": "written", "path": str(destination), "seed": label["seed"],
        "frames": int(len(label["sim_times"])), "task": label["task"],
        "body": label["body"], "condition": label["condition"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True, help="renderer replay sidecar root")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--allow-unsuccessful", action="store_true")
    args = parser.parse_args()
    label_paths = sorted(args.labels.resolve().rglob("*.hdf5"))
    if args.limit:
        label_paths = label_paths[: args.limit]
    if not label_paths:
        raise FileNotFoundError(f"no label HDF5 files below {args.labels}")
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.jsonl"
    rows: list[dict[str, Any]] = []
    for label_path in label_paths:
        try:
            rows.append(create_episode(label_path, output, not args.allow_unsuccessful))
        except Exception as exc:  # keep failures auditable rather than silently dropping them
            rows.append({"status": "error", "sidecar": str(label_path), "error": f"{type(exc).__name__}: {exc}"})
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output, delete=False) as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        temporary_manifest = Path(stream.name)
    os.replace(temporary_manifest, manifest)
    written = sum(row["status"] in {"written", "existing"} for row in rows)
    errors = len(rows) - written
    print(json.dumps({"format": FORMAT, "written": written, "errors": errors, "manifest": str(manifest)}))
    if written == 0:
        raise RuntimeError("no valid RGB/oracle paired episodes were materialized")


if __name__ == "__main__":
    main()

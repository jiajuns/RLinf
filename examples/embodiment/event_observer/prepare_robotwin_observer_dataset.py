#!/usr/bin/env python3
"""Materialize paired RobotWin RGB/oracle-event episodes for Event Observer training.

The official RobotWin archives remain the authoritative RGB source.  A paired
``collect_robotwin.py --scene-backend renderer --replay-archive ...`` sidecar
is the authoritative source of privileged event labels.  This tool refuses to
materialize a label file without its replay provenance, so visual samples can
never silently be paired with a different expert trajectory.

The resulting HDF5 contains no privileged scene state: only JPEG camera frames,
camera calibration, actions, and the training labels ``event_target``,
``event_boundary`` and ``event_progress``.  Privileged quantities stay in the
input sidecar and are never copied into observer inputs.
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


FORMAT = "robotwin_rgb_oracle_event_observer_v1"
EVENT_NAMES = (
    "approach",
    "grasp",
    "transport",
    "place",
    "release",
    "drop",
    "regrasp",
    "open",
    "activate",
    "idle",
)
CAMERAS = ("left_camera", "right_camera", "head_camera", "front_camera")
ARCHIVE_HASH_CACHE: dict[Path, str] = {}


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
        events = np.asarray(handle["events"], dtype=np.float32)
        if events.shape[1:] != (len(EVENT_NAMES),):
            raise ValueError(f"unsupported events shape {events.shape}")
        sim_times = np.asarray(handle["sim_times"], dtype=np.float64)
        return {
            "archive": archive,
            "archive_sha256": str(handle.attrs.get("replay_archive_sha256", "")),
            "replay_index": replay_index,
            "events": events,
            "sim_times": sim_times,
            "actions14": np.asarray(handle["actions14"], dtype=np.float32),
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
            first_camera = source[f"observation/{CAMERAS[0]}/rgb"]
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
                    for camera in CAMERAS:
                        camera_group = source[f"observation/{camera}"]
                        rgb = camera_group["rgb"]
                        if len(rgb) != len(first_camera):
                            raise ValueError(f"inconsistent camera length for {camera}")
                        output = rgb_group.create_dataset(camera, (len(frame_index),), dtype=jpeg_type)
                        for offset, source_index in enumerate(frame_index):
                            output[offset] = np.frombuffer(bytes(rgb[int(source_index)]), dtype=np.uint8)
                        for key in ("intrinsic_cv", "extrinsic_cv", "cam2world_gl"):
                            target.create_dataset(
                                f"camera/{camera}/{key}",
                                data=np.asarray(camera_group[key])[frame_index],
                                compression="gzip",
                                compression_opts=1,
                            )
                    boundary, progress = event_boundary_and_progress(label["events"])
                    target.create_dataset("frame_index", data=frame_index)
                    target.create_dataset("event_target", data=label["events"], compression="gzip", compression_opts=1)
                    target.create_dataset("event_boundary", data=boundary)
                    target.create_dataset("event_progress", data=progress)
                    target.create_dataset("actions14", data=label["actions14"], compression="gzip", compression_opts=1)
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

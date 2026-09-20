#!/usr/bin/env python3
"""Create an offline SAM 3.1 RGB teacher track for one RobotWin sidecar.

This tool is deliberately run in the isolated SAM Python environment.  It
reads only the ``rgb`` group written by the renderer collector, grounds one
role on frame zero, and then propagates that visual anchor through the whole
episode.  It never reads oracle masks, depth, object poses, contact, or event
labels.  The output is a compact, versionable [camera,time,6] track cache that
the lightweight Event Observer can consume without importing SAM online.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import tempfile
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_frames(handle: h5py.File, camera: str, directory: Path) -> tuple[int, int, int]:
    if "rgb" not in handle or camera not in handle["rgb"]:
        raise ValueError(f"episode has no RGB stream {camera!r}")
    frames = handle["rgb"][camera]
    if frames.ndim != 4 or frames.shape[-1] != 3 or len(frames) < 2:
        raise ValueError(f"invalid RGB stream {camera}: {frames.shape}")
    directory.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        Image.fromarray(np.asarray(frame, np.uint8), mode="RGB").save(directory / f"{index:05d}.jpg")
    return int(len(frames)), int(frames.shape[2]), int(frames.shape[1])


def _best_box(probabilities: object, boxes: object, width: int, height: int) -> np.ndarray:
    """Return normalized xywh, confidence, visibility for the best proposal."""
    probabilities = np.asarray(probabilities, np.float32).reshape(-1)
    boxes = np.asarray(boxes, np.float32)
    if not len(probabilities) or boxes.size == 0:
        return np.zeros(6, np.float32)
    boxes = boxes.reshape(-1, 4)
    count = min(len(probabilities), len(boxes))
    if count == 0:
        return np.zeros(6, np.float32)
    index = int(np.argmax(probabilities[:count]))
    x, y, w, h = boxes[index]
    if not np.isfinite([x, y, w, h, probabilities[index]]).all() or w <= 0 or h <= 0:
        return np.zeros(6, np.float32)
    # SAM returns xywh pixels.  Cx/cy and extent are normalized per image so
    # camera mounts/resolutions never enter the learned observer as IDs.
    return np.asarray(
        [(x + 0.5 * w) / width, (y + 0.5 * h) / height, w / width, h / height,
         float(probabilities[index]), 1.0],
        np.float32,
    )


def _track_camera(predictor: object, frames: Path, length: int, width: int, height: int, prompt: str) -> np.ndarray:
    state = predictor.model.init_state

    # The pinned SAM 3.1 multiplex predictor does not accept the legacy CPU
    # offload flag forwarded by the generic public wrapper.
    def init_state_compat(**kwargs):
        kwargs.pop("offload_state_to_cpu", None)
        return state(**kwargs)

    predictor.model.init_state = init_state_compat
    session = predictor.handle_request({"type": "start_session", "resource_path": str(frames)})
    session_id = session["session_id"]
    proposal = predictor.handle_request(
        {"type": "add_prompt", "session_id": session_id, "frame_index": 0, "text": prompt}
    )["outputs"]
    if not len(np.asarray(proposal["out_probs"]).reshape(-1)):
        return np.zeros((length, 6), np.float32)
    track = np.zeros((length, 6), np.float32)
    for response in predictor.handle_stream_request(
        {"type": "propagate_in_video", "session_id": session_id}
    ):
        index = int(response["frame_index"])
        if not 0 <= index < length:
            raise ValueError(f"SAM emitted frame {index} outside [0,{length})")
        outputs = response["outputs"]
        track[index] = _best_box(outputs["out_probs"], outputs["out_boxes_xywh"], width, height)
    return track


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", default="bottle")
    parser.add_argument("--cameras", default="right_camera,head_camera")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("SAM teacher tracking requires an allocated CUDA GPU")
    if not args.checkpoint.is_file() or not args.episode.is_file():
        raise FileNotFoundError("episode or SAM checkpoint is missing")
    from sam3.model_builder import build_sam3_multiplex_video_predictor

    cameras = tuple(name.strip() for name in args.cameras.split(",") if name.strip())
    if not cameras:
        raise ValueError("at least one camera is required")
    predictor = build_sam3_multiplex_video_predictor(
        checkpoint_path=str(args.checkpoint), use_fa3=False, compile=False,
        warm_up=False, async_loading_frames=False,
    )
    workspace = Path(tempfile.mkdtemp(prefix="sam31_robotwin_"))
    try:
        tracks, lengths = [], []
        with h5py.File(args.episode, "r") as handle:
            for camera in cameras:
                frame_dir = workspace / camera
                length, width, height = _write_frames(handle, camera, frame_dir)
                tracks.append(_track_camera(predictor, frame_dir, length, width, height, args.prompt))
                lengths.append(length)
        if len(set(lengths)) != 1:
            raise ValueError(f"camera streams have incompatible lengths: {lengths}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output,
            format=np.asarray("event_sam31_track_v1"),
            episode_sha256=np.asarray(_hash_file(args.episode)),
            checkpoint_sha256=np.asarray(_hash_file(args.checkpoint)),
            prompt=np.asarray(args.prompt),
            cameras=np.asarray(cameras),
            tracks=np.stack(tracks),
        )
        print({"output": str(args.output), "frames": lengths[0], "cameras": cameras,
               "visible_frames": [int(track[:, 5].sum()) for track in tracks]})
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


if __name__ == "__main__":
    main()

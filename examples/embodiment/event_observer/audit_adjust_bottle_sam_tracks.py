#!/usr/bin/env python3
"""Audit SAM teacher-track health before Event Observer training.

This is deliberately a *perception-only* gate.  It can reject empty,
out-of-range, or collapsed tracks, but never uses simulator masks, poses, or
relations to decide whether a track is correct.  The resulting accepted HDF
manifest is the sole input to cache construction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def track_health(path: Path, *, min_visible_fraction: float, min_box_area: float) -> tuple[bool, dict]:
    with np.load(path, allow_pickle=False) as data:
        required = {"format", "cameras", "tracks", "episode_sha256", "checkpoint_sha256"}
        missing = required.difference(data.files)
        if missing or str(data.get("format", "")) != "event_sam31_track_v1":
            return False, {"reason": f"invalid_format_or_missing:{sorted(missing)}"}
        cameras = tuple(map(str, data["cameras"]))
        tracks = np.asarray(data["tracks"], np.float32)
    if tracks.ndim != 3 or tracks.shape[0] != len(cameras) or tracks.shape[-1] != 6:
        return False, {"reason": "invalid_track_shape", "shape": list(tracks.shape)}
    if "head_camera" not in cameras:
        return False, {"reason": "head_camera_missing"}
    head = tracks[cameras.index("head_camera")]
    visible = head[:, 5] > 0.5
    fraction = float(visible.mean())
    diagnostics = {"frames": int(len(head)), "head_visible_fraction": fraction}
    if not visible.any():
        return False, {**diagnostics, "reason": "head_never_visible"}
    boxes = head[visible, :4]
    confidence = head[visible, 4]
    diagnostics.update({
        "head_confidence_median": float(np.median(confidence)),
        "head_box_area_median": float(np.median(boxes[:, 2] * boxes[:, 3])),
    })
    in_bounds = np.isfinite(boxes).all() and np.isfinite(confidence).all() and bool(
        ((boxes[:, :2] >= 0).all()) and ((boxes[:, :2] <= 1).all())
        and ((boxes[:, 2:] > 0).all()) and ((boxes[:, 2:] <= 1).all())
    )
    if not in_bounds:
        return False, {**diagnostics, "reason": "nonfinite_or_out_of_bounds"}
    if fraction < min_visible_fraction:
        return False, {**diagnostics, "reason": "low_head_visibility"}
    if diagnostics["head_box_area_median"] < min_box_area:
        return False, {**diagnostics, "reason": "collapsed_head_box"}
    return True, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, required=True, help="one absolute HDF path per line")
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--accepted", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--min-visible-fraction", type=float, default=0.5)
    parser.add_argument("--min-box-area", type=float, default=0.002)
    args = parser.parse_args()
    if not 0 < args.min_visible_fraction <= 1 or args.min_box_area <= 0:
        raise ValueError("invalid audit thresholds")
    episodes = [Path(line.strip()) for line in args.episodes.read_text().splitlines() if line.strip()]
    rows, accepted = [], []
    for episode in episodes:
        track = args.tracks / f"{episode.stem}.npz"
        if not track.is_file():
            ok, info = False, {"reason": "track_missing"}
        else:
            ok, info = track_health(track, min_visible_fraction=args.min_visible_fraction, min_box_area=args.min_box_area)
        row = {"episode": str(episode), "track": str(track), "accepted": ok, **info}
        rows.append(row)
        if ok:
            accepted.append(episode)
    args.accepted.parent.mkdir(parents=True, exist_ok=True)
    args.accepted.write_text("".join(f"{path}\n" for path in accepted))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({"accepted": len(accepted), "total": len(rows), "rows": rows}, indent=2) + "\n")
    print({"accepted": len(accepted), "total": len(rows), "report": str(args.report)})
    if not accepted:
        raise RuntimeError("SAM audit accepted no trajectories")


if __name__ == "__main__":
    main()

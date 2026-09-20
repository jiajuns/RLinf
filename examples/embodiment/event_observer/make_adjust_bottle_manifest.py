#!/usr/bin/env python3
"""Make a deterministic manifest of valid successful adjust_bottle sidecars.

The manifest is an audit boundary between simulator collection and perception:
only renderer sidecars with both required RGB streams and native task success
can become SAM teacher inputs.  It contains paths only—never oracle fields—so
SAM and the frozen Observer cannot accidentally consume privileged labels.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py


def valid_success(path: Path) -> bool:
    try:
        with h5py.File(path, "r") as handle:
            return (
                bool(handle.attrs.get("native_success", False))
                and {"head_camera", "right_camera"}.issubset(handle.get("rgb", {}))
                and all(key in handle for key in ("events", "ee_state16", "success", "sim_times"))
                and len(handle["success"]) >= 2
            )
    except OSError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="0 selects every valid success")
    args = parser.parse_args()
    if args.limit < 0:
        raise ValueError("--limit must be non-negative")
    paths = [path.resolve() for path in sorted(args.episodes.glob("episode_seed_*.hdf5")) if valid_success(path)]
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        raise RuntimeError("no valid successful renderer sidecars found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(f"{path}\n" for path in paths))
    print({"manifest": str(args.output), "successful_episodes": len(paths)})


if __name__ == "__main__":
    main()

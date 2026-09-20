#!/bin/bash
# Renderer-backed, task-matched RoboTwin trajectories for adjust_bottle.
#
# Every array element owns a disjoint contiguous seed range.  The collector records RGB,
# measured proprioception, and simulator-only event labels in the same HDF5
# sidecar.  SAM is deliberately *not* run here: it consumes these RGB streams
# later in its own offline array, so PPO never has a SAM dependency.
#
# Usage (after the one-seed validation):
#   sbatch --array=0-3%4 --export=ALL,EPISODES_PER_TASK=100 this_file
# If the cluster's QOS counts every array member at submission time, use one
# long-running member at a time instead:
#   sbatch --array=0 --export=ALL,EPISODES_PER_TASK=100 this_file
# Optional overrides: COLLECTION_SEED_BASE, COLLECTION_ROOT, SAMPLE_HZ,
# EPISODES_PER_TASK.
#
# The renderer source remains the RoboTwin checkout because task assets are
# resolved relative to it.  The collector itself is external only because it
# carries the task-specific oracle schema; output is portable HDF5.
#SBATCH --job-name=rw_adjust_collect
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/data/user/leviccdong/EKSF/outputs/rw_adjust_collect_%A_%a.out
#SBATCH --error=/data/user/leviccdong/EKSF/outputs/rw_adjust_collect_%A_%a.err

set -euo pipefail

ROOT=/data/user/leviccdong/EKSF
ROBOTWIN_ROOT=$ROOT/runtime/etsf_stage0/RoboTwin_v14_dual_gpu_20260831
COLLECTOR=$ROOT/experiments/current/robotwin2_universal_event_pretrain_v1_20260909/collect_robotwin.py
OUTPUT=${COLLECTION_ROOT:-$ROOT/data/robotwin_adjust_bottle_event_rgb_v1}
SEED_BASE=${COLLECTION_SEED_BASE:-10000}
SAMPLE_HZ=${SAMPLE_HZ:-10}
EPISODES_PER_TASK=${EPISODES_PER_TASK:-1}
TASK_ID=${SLURM_ARRAY_TASK_ID:?submit this as a Slurm array}
test "$EPISODES_PER_TASK" -gt 0
SEED=$((SEED_BASE + TASK_ID * EPISODES_PER_TASK))

test -d "$ROBOTWIN_ROOT"
test -f "$COLLECTOR"
source /share/anaconda3/bin/activate
conda activate "$ROOT/env_etsf"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$(dirname "$COLLECTOR"):${PYTHONPATH:-}"
export ASSETS_PATH="$ROBOTWIN_ROOT"
# The compute nodes need the environment-local NVIDIA Vulkan ICD, rather than
# a desktop-path default.  Renderer capture is required for the RGB protocol.
export VK_DRIVER_FILES="$ROOT/env_etsf/lib/python3.10/site-packages/sapien/vulkan_library/nvidia_icd.json"
export VK_ICD_FILENAMES="$VK_DRIVER_FILES"

cd "$ROBOTWIN_ROOT"
python -u "$COLLECTOR" \
  --robotwin-root "$ROBOTWIN_ROOT" \
  --task adjust_bottle --body aloha-agilex --condition clean \
  --seed-start "$SEED" --episodes "$EPISODES_PER_TASK" --sample-hz "$SAMPLE_HZ" \
  --scene-backend renderer --output "$OUTPUT"

EPISODE_DIR="$OUTPUT/adjust_bottle/aloha-agilex/clean"
python - "$EPISODE_DIR" "$SEED" "$EPISODES_PER_TASK" <<'PY'
import sys
from pathlib import Path
import h5py

directory, start, count = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
successes = 0
for seed in range(start, start + count):
    path = directory / f"episode_seed_{seed}.hdf5"
    if not path.is_file():
        raise SystemExit(f"collector did not produce {path}")
    with h5py.File(path, "r") as handle:
        required = {"rgb", "events", "ee_state16", "success"}
        missing = required.difference(handle)
        cameras = set(handle.get("rgb", {}))
        if missing or not {"head_camera", "right_camera"}.issubset(cameras):
            raise SystemExit(f"invalid sidecar: missing={sorted(missing)}, cameras={sorted(cameras)}")
        successes += bool(handle.attrs.get("native_success", False))
print({"episodes": count, "native_successes": successes,
       "seed_range_inclusive": [start, start + count - 1]})
PY

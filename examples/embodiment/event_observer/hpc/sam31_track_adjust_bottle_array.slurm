#!/bin/bash
# Offline-only SAM 3.1 teacher tracks for renderer-collected adjust_bottle
# sidecars.  Submit as an array after writing one absolute HDF5 path per line
# to SAM_EPISODE_LIST; online πRL/PPO never invokes this job or SAM itself.
#SBATCH --job-name=sam31_adjust_tracks
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/data/user/leviccdong/EKSF/outputs/sam31_adjust_%A_%a.out
#SBATCH --error=/data/user/leviccdong/EKSF/outputs/sam31_adjust_%A_%a.err

set -euo pipefail
ROOT=/data/user/leviccdong/EKSF
REPO=$ROOT/code/RLinf-piRL
SAM_PYTHON=/share/anaconda3/envs/pytorch271/bin/python
SAM_SOURCE=$ROOT/third_party/sam3_2345a4a
SAM_SITE=$ROOT/third_party/sam3_py312_site
CHECKPOINT=$ROOT/models/sam3.1/sam3.1_multiplex.pt
EPISODE_LIST=${SAM_EPISODE_LIST:?set a one-HDF5-path-per-line manifest}
TRACK_ROOT=${SAM_TRACK_ROOT:-$ROOT/data/robotwin_adjust_bottle_event_tracks_v1}

test -f "$CHECKPOINT"
test -f "$EPISODE_LIST"
EPISODE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$EPISODE_LIST")
test -n "$EPISODE" && test -f "$EPISODE"
mkdir -p "$TRACK_ROOT" "$ROOT/outputs"
OUTPUT="$TRACK_ROOT/$(basename "${EPISODE%.hdf5}").npz"
FRAMES=$(mktemp -d "${TMPDIR:-/tmp}/sam31_adjust_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_XXXX")
trap 'rm -rf "$FRAMES"' EXIT

# HDF5 is available in the RoboTwin Python environment, but intentionally not
# in SAM's Python 3.12 environment.  Decode RGB once, then pass JPEG-only
# camera streams to SAM so the perception teacher stays isolated.
source /share/anaconda3/bin/activate
conda activate "$ROOT/env_etsf"
python - "$EPISODE" "$FRAMES" <<'PY'
import sys
from pathlib import Path
import h5py
from PIL import Image

episode, root = map(Path, sys.argv[1:])
with h5py.File(episode, "r") as handle:
    for camera in ("right_camera", "head_camera"):
        out = root / camera
        out.mkdir(parents=True, exist_ok=True)
        for index, frame in enumerate(handle["rgb"][camera]):
            Image.fromarray(frame).save(out / f"{index:05d}.jpg", quality=95)
PY
EPISODE_SHA=$(sha256sum "$EPISODE" | awk '{print $1}')
conda deactivate

export PYTHONNOUSERSITE=1
export PYTHONPATH="$SAM_SOURCE:$SAM_SITE:$REPO/examples/embodiment/event_observer"
export PYTHONUNBUFFERED=1
"$SAM_PYTHON" "$REPO/examples/embodiment/event_observer/sam31_track_robotwin_episode.py" \
  --frames-root "$FRAMES" --episode-sha256 "$EPISODE_SHA" --checkpoint "$CHECKPOINT" --output "$OUTPUT" \
  --prompt bottle --cameras right_camera,head_camera

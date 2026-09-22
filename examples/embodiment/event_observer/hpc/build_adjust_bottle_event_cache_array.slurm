#!/bin/bash
# Convert matched renderer HDF5 + SAM track pairs to compact Observer caches.
# This is deliberately a separate stage: output features contain only the
# SAM-derived visual geometry and replay-readback proprioception; simulator
# contacts/poses/relations remain labels in the conversion process.
#
# Use the same manifest as SAM.  QOS-friendly serial batching:
#   sbatch --array=0 --export=ALL,EPISODES_PER_TASK=100 this_file
# or conventional arrays when available:
#   sbatch --array=0-3%4 --export=ALL,EPISODES_PER_TASK=100 this_file
#SBATCH --job-name=rw_adjust_cache
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/data/user/leviccdong/EKSF/outputs/rw_adjust_cache_%A_%a.out
#SBATCH --error=/data/user/leviccdong/EKSF/outputs/rw_adjust_cache_%A_%a.err

set -euo pipefail
ROOT=/data/user/leviccdong/EKSF
REPO=$ROOT/code/RLinf-piRL
EPISODE_LIST=${SAM_EPISODE_LIST:?set a successful adjust_bottle manifest}
TRACK_ROOT=${SAM_TRACK_ROOT:-$ROOT/data/robotwin_adjust_bottle_event_tracks_v1}
CACHE_ROOT=${EVENT_CACHE_ROOT:-$ROOT/data/robotwin_adjust_bottle_event_cache_chunk50_v2}
CONTROL_STEP_STRIDE=${CONTROL_STEP_STRIDE:-50}
EPISODES_PER_TASK=${EPISODES_PER_TASK:-1}
REBUILD_EVENT_CACHE=${REBUILD_EVENT_CACHE:-0}
TASK_ID=${SLURM_ARRAY_TASK_ID:?submit this as a Slurm array}
START=$((TASK_ID * EPISODES_PER_TASK))
TOTAL=$(wc -l < "$EPISODE_LIST")
if (( START >= TOTAL )); then exit 0; fi
END=$((START + EPISODES_PER_TASK)); if (( END > TOTAL )); then END=$TOTAL; fi

source /share/anaconda3/bin/activate
conda activate "$ROOT/env_etsf"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$REPO/examples/embodiment/event_observer:$REPO:${PYTHONPATH:-}"
mkdir -p "$CACHE_ROOT"
for INDEX in $(seq "$START" "$((END - 1))"); do
  EPISODE=$(sed -n "$((INDEX + 1))p" "$EPISODE_LIST")
  NAME=$(basename "${EPISODE%.hdf5}")
  TRACK="$TRACK_ROOT/$NAME.npz"
  OUTPUT="$CACHE_ROOT/$NAME.npz"
  test -s "$TRACK"
  if test -s "$OUTPUT" && test "$REBUILD_EVENT_CACHE" != 1; then
    # Old caches produced before the RGB-student stage lack source_episode;
    # rebuild those automatically rather than silently training a student on
    # an incomplete teacher corpus.
    if python - "$OUTPUT" <<'PY'
import sys
import numpy as np
with np.load(sys.argv[1], allow_pickle=False) as data:
    raise SystemExit(0 if "source_episode" in data and int(data.get("control_step_stride", -1)) == 50 else 1)
PY
    then
      echo "existing compatible $OUTPUT"
      continue
    fi
  fi
  python -u "$REPO/examples/embodiment/event_observer/build_adjust_bottle_event_cache.py" \
    --episode "$EPISODE" --tracks "$TRACK" --output "$OUTPUT" --control-step-stride "$CONTROL_STEP_STRIDE"
done

python - "$CACHE_ROOT" <<'PY'
import sys
from pathlib import Path
import numpy as np

root = Path(sys.argv[1])
for path in root.glob("*.npz"):
    with np.load(path, allow_pickle=False) as data:
        if not np.isfinite(data["features"]).all() or data["features"].shape[-1] != 30 or int(data.get("control_step_stride", -1)) != 50:
            raise SystemExit(f"invalid observer cache {path}")
print({"cache_files": len(list(root.glob("*.npz")))})
PY

#!/bin/bash
# Convert every verified RobotWin replay sidecar into paired RGB/proprio/event
# episodes.  This is CPU-only: GPU time is reserved for SAM 3.1 tracking.
# Each array member owns a separate output root, avoiding concurrent manifest
# writes; the roots are merged only after all receipts are successful.
#SBATCH --job-name=rw_observe_materialize
#SBATCH --partition=acd_u
# ACD's scheduler requires every job in this partition to declare a GPU even
# though this materialization process is CPU/I/O-only.
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --array=0-9%2
#SBATCH --output=/data/user/leviccdong/EKSF/outputs/robotwin_event_observer/materialize_%A_%a.out
#SBATCH --error=/data/user/leviccdong/EKSF/outputs/robotwin_event_observer/materialize_%A_%a.err

set -euo pipefail
ROOT=/data/user/leviccdong/EKSF
REPO=$ROOT/code/RLinf-piRL
LABELS=$ROOT/data/robotwin_event_observer_v1/oracle_labels_official
OUTPUT=$ROOT/data/robotwin_event_observer_v3/materialized
TASKS=(move_can_pot press_stapler)
BODIES=(aloha-agilex arx-x5 franka piper ur5)
TASK_INDEX=$((SLURM_ARRAY_TASK_ID / ${#BODIES[@]}))
BODY_INDEX=$((SLURM_ARRAY_TASK_ID % ${#BODIES[@]}))
TASK=${TASKS[$TASK_INDEX]}
BODY=${BODIES[$BODY_INDEX]}
SOURCE=$LABELS/$TASK/$BODY/clean
DESTINATION=$OUTPUT/$TASK/$BODY

test -d "$SOURCE"
source /share/anaconda3/bin/activate "$ROOT/env_etsf"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
mkdir -p "$DESTINATION" "$ROOT/outputs/robotwin_event_observer"

python -u "$REPO/examples/embodiment/event_observer/prepare_robotwin_observer_dataset.py" \
  --labels "$SOURCE" --output "$DESTINATION"

python - "$DESTINATION/manifest.jsonl" <<'PY'
import json
import sys
from pathlib import Path

rows = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines()]
written = [row for row in rows if row["status"] in {"written", "existing"}]
if len(rows) != 50 or len(written) != 50:
    raise SystemExit(f"expected 50 verified paired episodes, got {len(written)}/{len(rows)}")
print({"materialization": "passed", "episodes": len(written)})
PY

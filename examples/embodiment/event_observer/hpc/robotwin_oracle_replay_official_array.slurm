#!/bin/bash
# Build trusted oracle sidecars for every direct RobotWin public expert episode.
# Each output is checked against native success before becoming eligible for
# visual Event Observer pretraining.
#SBATCH --job-name=rw_observe_labels
#SBATCH --partition=acd_u
#SBATCH --nodelist=ACD1-54
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=08:00:00
#SBATCH --array=0-9%2
#SBATCH --output=/data/user/leviccdong/EKSF/outputs/robotwin_event_observer/labels_%A_%a.out
#SBATCH --error=/data/user/leviccdong/EKSF/outputs/robotwin_event_observer/labels_%A_%a.err

set -euo pipefail
ROOT=/data/user/leviccdong/EKSF
LEGACY_EXP=$ROOT/experiments/current/robotwin2_universal_event_pretrain_v1_20260909
ROBOTWIN=$ROOT/runtime/etsf_stage0/RoboTwin_v14_dual_gpu_20260831
OUT=$ROOT/data/robotwin_event_observer_v1/oracle_labels_official
# Only these two direct public tasks are already represented in the validated
# oracle schema.  The other two archive tasks are held out until their task
# semantics are added and unit-tested rather than being mislabeled.
TASKS=(move_can_pot press_stapler)
BODIES=(aloha-agilex arx-x5 franka piper ur5)
TASK_INDEX=$((SLURM_ARRAY_TASK_ID / 5))
BODY_INDEX=$((SLURM_ARRAY_TASK_ID % 5))
TASK=${TASKS[$TASK_INDEX]}
BODY=${BODIES[$BODY_INDEX]}
ARCHIVE=$ROOT/data/official/robotwin2_four_task_actor/dataset/$TASK/${BODY}_clean_50.zip

source /share/anaconda3/bin/activate
conda activate "$ROOT/env_etsf"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$LEGACY_EXP"
export ASSETS_PATH="$ROBOTWIN"
export PYTHONUNBUFFERED=1
export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
mkdir -p "$OUT" "$ROOT/outputs/robotwin_event_observer"
cd "$ROBOTWIN"
python -u "$LEGACY_EXP/collect_robotwin.py" \
  --robotwin-root "$ROBOTWIN" --task "$TASK" --body "$BODY" \
  --condition clean --seed-start 0 --episodes 50 --sample-hz 15 \
  --scene-backend renderer --replay-archive "$ARCHIVE" --replay-index-start 0 \
  --output "$OUT"
python - "$OUT/$TASK/$BODY/clean" <<'PY'
import sys
from pathlib import Path
import h5py

root = Path(sys.argv[1])
paths = sorted(root.glob("episode_seed_*.hdf5"))
if len(paths) != 50:
    raise SystemExit(f"expected 50 oracle sidecars, found {len(paths)}")
failed = []
for path in paths:
    with h5py.File(path, "r") as handle:
        if not bool(handle.attrs.get("native_success", False)):
            failed.append(path.name)
if failed:
    raise SystemExit(f"replay labels failed native success: {failed[:5]}")
print({"task_body": str(root), "verified_episodes": len(paths)})
PY

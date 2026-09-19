#!/bin/bash
# A short scheduling probe.  It deliberately does not request a fixed node:
# only a probe that passes RobotWin's native success is allowed to nominate a
# future formal collection node.
#SBATCH --job-name=rw_observe_probe
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --array=0-3%4
#SBATCH --output=/data/user/leviccdong/EKSF/outputs/robotwin_event_observer/probe_%A_%a.out
#SBATCH --error=/data/user/leviccdong/EKSF/outputs/robotwin_event_observer/probe_%A_%a.err

set -euo pipefail
ROOT=/data/user/leviccdong/EKSF
LEGACY_EXP=$ROOT/experiments/current/robotwin2_universal_event_pretrain_v1_20260909
ROBOTWIN=$ROOT/runtime/etsf_stage0/RoboTwin_v14_dual_gpu_20260831
ARCHIVE=$ROOT/data/official/robotwin2_four_task_actor/dataset/move_can_pot/aloha-agilex_clean_50.zip
OUT=$ROOT/data/robotwin_event_observer_v1/renderer_probes/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}

source /share/anaconda3/bin/activate
conda activate "$ROOT/env_etsf"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$LEGACY_EXP"
export ASSETS_PATH="$ROBOTWIN"
export PYTHONUNBUFFERED=1
SAPIEN_ICD=$ROOT/env_etsf/lib/python3.10/site-packages/sapien/vulkan_library/nvidia_icd.json
test -f "$SAPIEN_ICD"
export VK_DRIVER_FILES="$SAPIEN_ICD"
export VK_ICD_FILENAMES="$SAPIEN_ICD"
mkdir -p "$OUT" "$ROOT/outputs/robotwin_event_observer"
cd "$ROBOTWIN"
python -u "$LEGACY_EXP/collect_robotwin.py" \
  --robotwin-root "$ROBOTWIN" --task move_can_pot --body aloha-agilex \
  --condition clean --seed-start 0 --episodes 1 --sample-hz 15 \
  --scene-backend renderer --replay-archive "$ARCHIVE" --replay-index-start "$SLURM_ARRAY_TASK_ID" \
  --output "$OUT"
python - "$OUT/move_can_pot/aloha-agilex/clean" <<'PY'
import sys
from pathlib import Path
import h5py

paths = sorted(Path(sys.argv[1]).glob("episode_seed_*.hdf5"))
if len(paths) != 1:
    raise SystemExit(f"expected one oracle sidecar, found {len(paths)}")
with h5py.File(paths[0], "r") as handle:
    if not bool(handle.attrs.get("native_success", False)):
        raise SystemExit("renderer replay did not pass native success")
print({"renderer_probe": "passed", "host": __import__("socket").gethostname(), "sidecar": str(paths[0])})
PY

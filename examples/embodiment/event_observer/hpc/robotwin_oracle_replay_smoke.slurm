#!/bin/bash
# Validate one official RobotWin expert replay before any large-scale collection.
# This runs only on the known Vulkan-capable node; H100/headless replays are not
# admissible label sources for the visual Event Observer.
#SBATCH --job-name=rw_observe_label_smoke
#SBATCH --partition=acd_u
#SBATCH --nodelist=ACD1-54
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=/data/user/leviccdong/EKSF/outputs/robotwin_event_observer/label_smoke_%j.out
#SBATCH --error=/data/user/leviccdong/EKSF/outputs/robotwin_event_observer/label_smoke_%j.err

set -euo pipefail
ROOT=/data/user/leviccdong/EKSF
LEGACY_EXP=$ROOT/experiments/current/robotwin2_universal_event_pretrain_v1_20260909
ROBOTWIN=$ROOT/runtime/etsf_stage0/RoboTwin_v14_dual_gpu_20260831
ARCHIVE=$ROOT/data/official/robotwin2_four_task_actor/dataset/move_can_pot/aloha-agilex_clean_50.zip
OUT=$ROOT/data/robotwin_event_observer_v1/oracle_labels_smoke

source /share/anaconda3/bin/activate
conda activate "$ROOT/env_etsf"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$LEGACY_EXP"
export ASSETS_PATH="$ROBOTWIN"
export PYTHONUNBUFFERED=1
# Do not override the Vulkan ICD.  SAPIEN selects its compatible bundled ICD
# on this cluster; forcing /usr/share (which does not exist here) hides the
# allocated H100 from the renderer.
mkdir -p "$(dirname "$OUT")" "$ROOT/outputs/robotwin_event_observer"
cd "$ROBOTWIN"
python -u "$LEGACY_EXP/collect_robotwin.py" \
  --robotwin-root "$ROBOTWIN" --task move_can_pot --body aloha-agilex \
  --condition clean --seed-start 0 --episodes 1 --sample-hz 15 \
  --scene-backend renderer --replay-archive "$ARCHIVE" --replay-index-start 0 \
  --output "$OUT"

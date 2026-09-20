#!/bin/bash
# Actual SAPIEN/RoboTwin same-state validation required before interventions.
#SBATCH --job-name=rw_snapshot_validate
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/data/user/leviccdong/EKSF/outputs/rw_snapshot_validate_%j.out
#SBATCH --error=/data/user/leviccdong/EKSF/outputs/rw_snapshot_validate_%j.err
set -euo pipefail
ROOT=/data/user/leviccdong/EKSF
REPO=$ROOT/code/RLinf-piRL
SOURCE=$ROOT/runtime/RoboTwin-RLinf_support
ASSETS=$ROOT/runtime/etsf_stage0/RoboTwin_v14_dual_gpu_20260831
source /share/anaconda3/bin/activate "$ROOT/env_pirl_pi05"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$REPO:$SOURCE:${PYTHONPATH:-}"
export EMBODIED_PATH="$REPO/examples/embodiment"
export REPO_PATH="$REPO"
export ASSETS_PATH="$ASSETS"
export ROBOTWIN_CAMERA_SHADER=default
cd "$SOURCE"
python -u "$REPO/examples/embodiment/event_observer/validate_robotwin_snapshot.py" \
  --repo "$REPO" --robotwin-assets "$ASSETS" --robotwin-source "$SOURCE" --shader rt --disable-denoiser

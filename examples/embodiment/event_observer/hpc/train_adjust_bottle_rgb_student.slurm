#!/bin/bash
# Distil the offline SAM teacher to the deployment-capable RGB frontend.
#SBATCH --job-name=rw_adjust_rgb_student
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/data/user/leviccdong/EKSF/outputs/rw_adjust_rgb_student_%j.out
#SBATCH --error=/data/user/leviccdong/EKSF/outputs/rw_adjust_rgb_student_%j.err
set -euo pipefail
ROOT=/data/user/leviccdong/EKSF
REPO=$ROOT/code/RLinf-piRL
CACHE=${EVENT_CACHE_ROOT:-$ROOT/data/robotwin_adjust_bottle_event_cache_chunk50_v2}
OUT=${RGB_STUDENT_OUTPUT:-$ROOT/outputs/robotwin_adjust_bottle_rgb_student_chunk50_v2}
source /share/anaconda3/bin/activate "$ROOT/env_pirl_pi05"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
cd "$REPO"
python -u examples/embodiment/event_observer/train_adjust_bottle_rgb_student.py --cache "$CACHE" --output "$OUT" --control-step-stride 50 --proprio-time-delta 50.0 --online-mount-token 2
test -s "$OUT/best.pt"

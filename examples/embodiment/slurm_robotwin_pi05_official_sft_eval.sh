#!/usr/bin/env bash
# RLinf RoboTwin π0.5 SFT evaluation under the paper-wide 4-GPU profile.
# No algorithm, batch, environment-count, seed, or evaluation overrides are
# applied: this is the upstream 128 fixed-seed OpenPI evaluation recipe.
#SBATCH --job-name=robotwin_pi05_official_sft_eval
#SBATCH --partition=acd_ue
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --time=12:00:00
#SBATCH --output=/data/user/leviccdong/EKSF/outputs/robotwin_pi05_official_sft_eval_%j.out
#SBATCH --error=/data/user/leviccdong/EKSF/outputs/robotwin_pi05_official_sft_eval_%j.err

set -euo pipefail
export RLINF_LOCAL_RAY=1
unset RAY_ADDRESS
export RLINF_RAY_TMPDIR="/tmp/rlinf_ray_${SLURM_JOB_ID:?}"
export RLINF_COMPONENT_PLACEMENT=0-3
export RLINF_EVAL_COMPONENT_PLACEMENT=0-3
export REPO_PATH="${REPO_PATH:-/data/user/leviccdong/EKSF/code/RLinf-piRL}"
export EMBODIED_PATH="${REPO_PATH}/examples/embodiment"
export ROBOTWIN_PATH="${ROBOTWIN_PATH:-/data/user/leviccdong/EKSF/runtime/RoboTwin-RLinf_support}"
export ROBOTWIN_ASSETS_PATH="${ROBOTWIN_ASSETS_PATH:-/data/user/leviccdong/EKSF/runtime/etsf_stage0/RoboTwin_v14_dual_gpu_20260831}"
export ROBOTWIN_PI05_MODEL="${ROBOTWIN_PI05_MODEL:-/data/user/leviccdong/EKSF/models/RLinf-Pi05-RoboTwin-SFT-adjust_bottle}"
export ROBOT_PLATFORM=ALOHA
export PYTHONPATH="${REPO_PATH}:${ROBOTWIN_PATH}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1

source /share/anaconda3/bin/activate /data/user/leviccdong/EKSF/env_pirl_pi05
cd "${REPO_PATH}"

# This calls the upstream evaluation entry point and its published
# robotwin_adjust_bottle_openpi_pi05_eval.yaml without reducing 128 eval envs.
bash evaluations/run_eval.sh robotwin robotwin_adjust_bottle_openpi_pi05_eval \
  "env.eval.assets_path=${ROBOTWIN_ASSETS_PATH}" \
  env.eval.video_cfg.save_video=false \
  "rollout.model.model_path=${ROBOTWIN_PI05_MODEL}" \
  +rollout.unnorm_key=adjust_bottle

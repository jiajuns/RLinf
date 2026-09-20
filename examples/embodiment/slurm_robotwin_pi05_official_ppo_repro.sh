#!/usr/bin/env bash
# Strict RLinf RoboTwin π0.5 + PPO/GAE reproduction.
# Deliberately preserve all published YAML defaults: 8 GPU placement, 256
# train envs, rollout_epoch=4, batch=2048/32, chunk-level GAE, update_epoch=5,
# lr=5e-6, clip=0.2, and max_epochs=1000.
#SBATCH --job-name=robotwin_pi05_official_ppo
#SBATCH --partition=acd_ue
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --time=72:00:00
#SBATCH --output=/data/user/leviccdong/EKSF/outputs/robotwin_pi05_official_ppo_%j.out
#SBATCH --error=/data/user/leviccdong/EKSF/outputs/robotwin_pi05_official_ppo_%j.err

set -euo pipefail
export RLINF_LOCAL_RAY=1
unset RAY_ADDRESS
export RLINF_RAY_TMPDIR="/tmp/rlinf_ray_${SLURM_JOB_ID:?}"
export RLINF_COMPONENT_PLACEMENT=0-7
export REPO_PATH="${REPO_PATH:-/data/user/leviccdong/EKSF/code/RLinf-piRL}"
export EMBODIED_PATH="${REPO_PATH}/examples/embodiment"
export ROBOTWIN_PATH="${ROBOTWIN_PATH:-/data/user/leviccdong/EKSF/runtime/RoboTwin-RLinf_support}"
export ROBOTWIN_ASSETS_PATH="${ROBOTWIN_ASSETS_PATH:-/data/user/leviccdong/EKSF/runtime/etsf_stage0/RoboTwin_v14_dual_gpu_20260831}"
export ROBOTWIN_PI05_MODEL="${ROBOTWIN_PI05_MODEL:-/data/user/leviccdong/EKSF/models/RLinf-Pi05-RoboTwin-SFT-adjust_bottle}"
export ROBOT_PLATFORM=ALOHA
export PYTHONPATH="${REPO_PATH}:${ROBOTWIN_PATH}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1

source /share/anaconda3/bin/activate /data/user/leviccdong/EKSF/env_pirl_pi05
cd "${ROBOTWIN_PATH}"

# Paths and the checkpoint-specific action normalization key are the only
# overrides.  Do not add smaller batches, action-level credit, custom PPO
# hyperparameters, or shortened horizons to this official reproduction.
exec python "${REPO_PATH}/examples/embodiment/train_embodied_agent.py" \
  --config-path config \
  --config-name robotwin_adjust_bottle_ppo_openpi_pi05 \
  runner.logger.log_path=/data/user/leviccdong/EKSF/outputs/robotwin_pi05_official_ppo \
  runner.logger.experiment_name=robotwin_adjust_bottle_pi05_official_ppo \
  env.train.assets_path="${ROBOTWIN_ASSETS_PATH}" \
  env.eval.assets_path="${ROBOTWIN_ASSETS_PATH}" \
  actor.model.model_path="${ROBOTWIN_PI05_MODEL}" \
  rollout.model.model_path="${ROBOTWIN_PI05_MODEL}" \
  +rollout.unnorm_key=adjust_bottle

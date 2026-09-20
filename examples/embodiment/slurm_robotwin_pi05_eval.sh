#!/usr/bin/env bash
# Deterministic, evaluation-only RoboTwin adjust_bottle runner.
#
# Usage (on the HPC):
#   ROBOTWIN_EVAL_CKPT=/path/to/full_weights.pt \
#   ROBOTWIN_EVAL_TAG=pi_rl_step25 \
#   sbatch examples/embodiment/slurm_robotwin_pi05_eval.sh
#
# Leave ROBOTWIN_EVAL_CKPT unset to evaluate the original π0.5 SFT model.
# All evaluations use the same first 64 fixed RoboTwin evaluation reset states.
# This script deliberately does not construct the Event sidecars: they affect
# credit assignment during training, not the deterministic actor policy used at
# evaluation time.
#SBATCH --job-name=robotwin_pi05_eval
#SBATCH --partition=acd_ue
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=06:00:00
#SBATCH --output=/data/user/leviccdong/EKSF/outputs/robotwin_eval_%j.out
#SBATCH --error=/data/user/leviccdong/EKSF/outputs/robotwin_eval_%j.err

set -euo pipefail

export RLINF_LOCAL_RAY=1
unset RAY_ADDRESS
export RLINF_RAY_TMPDIR="/tmp/rlinf_ray_${SLURM_JOB_ID:?}"
# A unique dashboard port is equally necessary: Ray defaults every local head
# to 8265, even when its temp directory is job-local.  The chosen range is
# deterministic per Slurm job and avoids the standard dashboard port.
export RLINF_RAY_DASHBOARD_PORT="${RLINF_RAY_DASHBOARD_PORT:-$((20000 + SLURM_JOB_ID % 20000))}"
export RLINF_COMPONENT_PLACEMENT="${RLINF_COMPONENT_PLACEMENT:-0}"
export REPO_PATH="${REPO_PATH:-/data/user/leviccdong/EKSF/code/RLinf-piRL}"
export EMBODIED_PATH="${REPO_PATH}/examples/embodiment"
export ROBOTWIN_PATH="${ROBOTWIN_PATH:-/data/user/leviccdong/EKSF/runtime/RoboTwin-RLinf_support}"
export ROBOTWIN_ASSETS_PATH="${ROBOTWIN_ASSETS_PATH:-/data/user/leviccdong/EKSF/runtime/etsf_stage0/RoboTwin_v14_dual_gpu_20260831}"
export ROBOTWIN_PI05_MODEL="${ROBOTWIN_PI05_MODEL:-/data/user/leviccdong/EKSF/models/RLinf-Pi05-RoboTwin-SFT-adjust_bottle}"
export ROBOT_PLATFORM=ALOHA
export PYTHONPATH="${REPO_PATH}:${ROBOTWIN_PATH}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1

export ROBOTWIN_EVAL_TAG="${ROBOTWIN_EVAL_TAG:-pi05_sft}"
export ROBOTWIN_EVAL_ENVS="${ROBOTWIN_EVAL_ENVS:-8}"
export ROBOTWIN_EVAL_ROLLOUT_EPOCHS="${ROBOTWIN_EVAL_ROLLOUT_EPOCHS:-8}"
export ROBOTWIN_EVAL_LOG_PATH="${ROBOTWIN_EVAL_LOG_PATH:-/data/user/leviccdong/EKSF/outputs/robotwin_checkpoint_eval}"

if [[ -n "${ROBOTWIN_EVAL_CKPT:-}" && ! -s "${ROBOTWIN_EVAL_CKPT}" ]]; then
  echo "ROBOTWIN_EVAL_CKPT is not a readable checkpoint: ${ROBOTWIN_EVAL_CKPT}" >&2
  exit 2
fi

cd "${ROBOTWIN_PATH}"
source /share/anaconda3/bin/activate /data/user/leviccdong/EKSF/env_pirl_pi05

args=(
  runner.only_eval=true
  runner.logger.log_path="${ROBOTWIN_EVAL_LOG_PATH}"
  runner.logger.experiment_name="${ROBOTWIN_EVAL_TAG}"
  env.eval.total_num_envs="${ROBOTWIN_EVAL_ENVS}"
  env.eval.rollout_epoch="${ROBOTWIN_EVAL_ROLLOUT_EPOCHS}"
  env.eval.assets_path="${ROBOTWIN_ASSETS_PATH}"
  env.eval.video_cfg.save_video=false
  +rollout.unnorm_key=adjust_bottle
  rollout.model.model_path="${ROBOTWIN_PI05_MODEL}"
  +rollout.model.model_type=openpi
  +rollout.model.num_action_chunks=50
  +rollout.model.action_dim=14
  +rollout.model.num_steps=5
  +rollout.model.add_value_head=true
  +rollout.model.openpi.config_name=pi05_aloha_robotwin
  +rollout.model.openpi.num_images_in_input=3
  +rollout.model.openpi.noise_level=0.3
  +rollout.model.openpi.detach_critic_input=true
  +rollout.model.openpi.noise_method=flow_sde
  +rollout.model.openpi.value_after_vlm=false
  +rollout.model.openpi.joint_logprob=false
)

if [[ -n "${ROBOTWIN_EVAL_CKPT:-}" ]]; then
  args+=(runner.ckpt_path="${ROBOTWIN_EVAL_CKPT}")
fi

exec python "${REPO_PATH}/examples/embodiment/train_embodied_agent.py" \
  --config-path config \
  --config-name robotwin_adjust_bottle_ppo_openpi_pi05 \
  "${args[@]}"

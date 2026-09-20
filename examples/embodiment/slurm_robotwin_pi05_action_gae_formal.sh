#!/usr/bin/env bash
# Official RoboTwin π0.5 + πRL Flow-SDE/PPO control for the Event-Value study.
# It is intentionally a normal-length 200-step training job, not a smoke test.
# Override ROBOTWIN_* variables at submission time to scale the interaction
# budget; all Event-SMDP ablations must retain the same actor rollout budget.
#SBATCH --job-name=robotwin_pi05_gae
#SBATCH --partition=acd_ue
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=48:00:00
#SBATCH --output=/data/user/leviccdong/EKSF/outputs/robotwin_pi05_gae_%j.out
#SBATCH --error=/data/user/leviccdong/EKSF/outputs/robotwin_pi05_gae_%j.err

set -euo pipefail

export RLINF_LOCAL_RAY=1
unset RAY_ADDRESS
export RLINF_RAY_TMPDIR="/tmp/rlinf_ray_${SLURM_JOB_ID:?}"

export REPO_PATH="${REPO_PATH:-/data/user/leviccdong/EKSF/code/RLinf-piRL}"
export EMBODIED_PATH="${REPO_PATH}/examples/embodiment"
export ROBOTWIN_PATH="${ROBOTWIN_PATH:-/data/user/leviccdong/EKSF/runtime/etsf_stage0/RoboTwin_v14_dual_gpu_20260831}"
export ROBOTWIN_ASSETS_PATH="${ROBOTWIN_ASSETS_PATH:-${ROBOTWIN_PATH}}"
export ROBOTWIN_PI05_MODEL="${ROBOTWIN_PI05_MODEL:-/data/user/leviccdong/EKSF/models/RLinf-Pi05-RoboTwin-SFT-adjust_bottle}"
export ROBOT_PLATFORM=ALOHA
export PYTHONPATH="${REPO_PATH}:${ROBOTWIN_PATH}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1

# These defaults preserve a 200-step RoboTwin episode.  They are deliberately
# small enough for a one-H100 reproducibility run; scaling must adjust baseline
# and Event-SMDP together.
export ROBOTWIN_MAX_EPOCHS="${ROBOTWIN_MAX_EPOCHS:-100}"
export ROBOTWIN_TRAIN_ENVS="${ROBOTWIN_TRAIN_ENVS:-2}"
export ROBOTWIN_EVAL_ENVS="${ROBOTWIN_EVAL_ENVS:-2}"
export ROBOTWIN_ROLLOUT_EPOCHS="${ROBOTWIN_ROLLOUT_EPOCHS:-1}"
export ROBOTWIN_EVAL_ROLLOUT_EPOCHS="${ROBOTWIN_EVAL_ROLLOUT_EPOCHS:-8}"
export ROBOTWIN_GLOBAL_BATCH="${ROBOTWIN_GLOBAL_BATCH:-2}"
export ROBOTWIN_MICRO_BATCH="${ROBOTWIN_MICRO_BATCH:-1}"
export ROBOTWIN_LOG_PATH="${ROBOTWIN_LOG_PATH:-/data/user/leviccdong/EKSF/outputs/robotwin_pi05_gae}"
export ROBOTWIN_EXPERIMENT_NAME="${ROBOTWIN_EXPERIMENT_NAME:-robotwin_adjust_bottle_pi05_flow_sde_action_gae}"

cd "${REPO_PATH}"
# This is a conda-prefix environment on the HPC, not a venv (there is no
# ``bin/activate`` inside it).  Match the environment activation used by the
# already-running ManiSkill πRL jobs.
source /share/anaconda3/bin/activate /data/user/leviccdong/EKSF/env_pirl_pi05

# Hydra resolves config paths relative to train_embodied_agent.py's directory.
# Passing examples/embodiment/config here therefore produced a duplicated
# examples/embodiment/examples/embodiment/config path on the HPC.  The
# upstream recipe is itself a Hydra primary config (it owns hydra.searchpath),
# so retain it as the primary and apply this fair-credit overlay as CLI flags.
exec python examples/embodiment/train_embodied_agent.py \
  --config-path config \
  --config-name robotwin_adjust_bottle_ppo_openpi_pi05 \
  runner.max_epochs="${ROBOTWIN_MAX_EPOCHS}" \
  runner.logger.log_path="${ROBOTWIN_LOG_PATH}" \
  runner.logger.experiment_name="${ROBOTWIN_EXPERIMENT_NAME}" \
  runner.val_check_interval=25 \
  runner.save_interval=25 \
  env.train.total_num_envs="${ROBOTWIN_TRAIN_ENVS}" \
  env.eval.total_num_envs="${ROBOTWIN_EVAL_ENVS}" \
  env.train.rollout_epoch="${ROBOTWIN_ROLLOUT_EPOCHS}" \
  env.eval.rollout_epoch="${ROBOTWIN_EVAL_ROLLOUT_EPOCHS}" \
  env.train.assets_path="${ROBOTWIN_ASSETS_PATH}" \
  env.eval.assets_path="${ROBOTWIN_ASSETS_PATH}" \
  actor.model.model_path="${ROBOTWIN_PI05_MODEL}" \
  rollout.model.model_path="${ROBOTWIN_PI05_MODEL}" \
  rollout.unnorm_key=adjust_bottle \
  algorithm.reward_type=action_level \
  algorithm.logprob_type=action_level \
  algorithm.adv_type=gae \
  algorithm.loss_type=actor_critic \
  actor.model.openpi.noise_method=flow_sde \
  actor.model.openpi.joint_logprob=false \
  actor.model.openpi.value_after_vlm=true \
  actor.global_batch_size="${ROBOTWIN_GLOBAL_BATCH}" \
  actor.micro_batch_size="${ROBOTWIN_MICRO_BATCH}"

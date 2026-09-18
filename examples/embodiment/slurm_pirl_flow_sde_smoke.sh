#!/usr/bin/env bash
#SBATCH --job-name=pirl_fsd_smoke
# acd_ue has the same ACD1 GPU pool as acd_u, but higher scheduling priority.
# Do not use debug: its per-user GPU QoS is already occupied by another job.
#SBATCH --partition=acd_ue
# Do not pin a GPU node.  A stale Vulkan context can make an individual node
# temporarily unavailable; Slurm must be free to select another H100 node for
# the single-job ManiSkill render probe and the matched training campaign.
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:12:00
#SBATCH --output=/data/user/leviccdong/EKSF/outputs/pirl_flow_sde_smoke_%j.out
#SBATCH --error=/data/user/leviccdong/EKSF/outputs/pirl_flow_sde_smoke_%j.err

# End-to-end official πRL smoke. Only scale is reduced; Flow-SDE, GAE and the
# stock PPO actor-critic algorithm remain those in the baseline config.
set -euo pipefail
ROOT=/data/user/leviccdong/EKSF/code/RLinf-piRL
MODEL=/data/user/leviccdong/EKSF/models/RLinf-Pi05-ManiSkill-25Main-SFT
ASSETS=/data/user/leviccdong/EKSF/staging/pirl_assets/maniskill_assets
source /share/anaconda3/bin/activate /data/user/leviccdong/EKSF/env_pirl_pi05
export EMBODIED_PATH="$ROOT/examples/embodiment"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export MANISKILL_ASSET_DIR="$ASSETS"
export MS_ASSET_DIR="$ASSETS"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TOKENIZERS_PARALLELISM=false
cd "$ROOT"

# Defaults are a one-step smoke test.  Every scale parameter is overridable so
# the same launcher can run a paired normal-horizon comparison without hidden
# differences between GAE and Event-SMDP.
PIRL_MAX_EPOCHS="${PIRL_MAX_EPOCHS:-1}"
PIRL_MAX_STEPS="${PIRL_MAX_STEPS:-1}"
PIRL_SAVE_INTERVAL="${PIRL_SAVE_INTERVAL:--1}"
PIRL_VAL_INTERVAL="${PIRL_VAL_INTERVAL:-10}"
PIRL_LOG_PATH="${PIRL_LOG_PATH:-/data/user/leviccdong/EKSF/outputs/pirl_flow_sde_smoke}"
PIRL_EXPERIMENT_NAME="${PIRL_EXPERIMENT_NAME:-official_pi05_flow_sde_smoke}"
# Keep the audited baseline default, while allowing a separate, explicit
# Event-SMDP smoke submission without editing this shared launcher.
PIRL_CONFIG_NAME="${PIRL_CONFIG_NAME:-maniskill_ppo_openpi_pi05_flow_sde}"
PIRL_TRAIN_ENVS="${PIRL_TRAIN_ENVS:-2}"
PIRL_EVAL_ENVS="${PIRL_EVAL_ENVS:-1}"
PIRL_TRAIN_EPISODE_STEPS="${PIRL_TRAIN_EPISODE_STEPS:-5}"
PIRL_EVAL_EPISODE_STEPS="${PIRL_EVAL_EPISODE_STEPS:-5}"
PIRL_TRAIN_ROLLOUT_STEPS="${PIRL_TRAIN_ROLLOUT_STEPS:-10}"
PIRL_EVAL_ROLLOUT_STEPS="${PIRL_EVAL_ROLLOUT_STEPS:-5}"
PIRL_TRAIN_ROLLOUT_EPOCHS="${PIRL_TRAIN_ROLLOUT_EPOCHS:-1}"
PIRL_EVAL_ROLLOUT_EPOCHS="${PIRL_EVAL_ROLLOUT_EPOCHS:-1}"
PIRL_SEED="${PIRL_SEED:-0}"
PIRL_REWARD_TYPE="${PIRL_REWARD_TYPE:-}"
PIRL_MICRO_BATCH_SIZE="${PIRL_MICRO_BATCH_SIZE:-1}"
PIRL_GLOBAL_BATCH_SIZE="${PIRL_GLOBAL_BATCH_SIZE:-2}"

extra_overrides=()
if [[ -n "$PIRL_REWARD_TYPE" ]]; then
  extra_overrides+=("algorithm.reward_type=$PIRL_REWARD_TYPE")
fi

python examples/embodiment/train_embodied_agent.py \
  --config-path config \
  --config-name "$PIRL_CONFIG_NAME" \
  runner.max_epochs="$PIRL_MAX_EPOCHS" runner.max_steps="$PIRL_MAX_STEPS" \
  runner.save_interval="$PIRL_SAVE_INTERVAL" runner.val_check_interval="$PIRL_VAL_INTERVAL" \
  runner.logger.log_path="$PIRL_LOG_PATH" \
  runner.logger.experiment_name="$PIRL_EXPERIMENT_NAME" \
  rollout.model.model_path="$MODEL" actor.model.model_path="$MODEL" \
  env.train.total_num_envs="$PIRL_TRAIN_ENVS" env.eval.total_num_envs="$PIRL_EVAL_ENVS" \
  env.train.rollout_epoch="$PIRL_TRAIN_ROLLOUT_EPOCHS" env.eval.rollout_epoch="$PIRL_EVAL_ROLLOUT_EPOCHS" \
  env.train.max_episode_steps="$PIRL_TRAIN_EPISODE_STEPS" env.train.max_steps_per_rollout_epoch="$PIRL_TRAIN_ROLLOUT_STEPS" \
  env.eval.max_episode_steps="$PIRL_EVAL_EPISODE_STEPS" env.eval.max_steps_per_rollout_epoch="$PIRL_EVAL_ROLLOUT_STEPS" \
  actor.seed="$PIRL_SEED" env.train.seed="$PIRL_SEED" env.eval.seed="$PIRL_SEED" \
  env.train.init_params.sensor_configs.shader_pack=minimal \
  env.eval.init_params.sensor_configs.shader_pack=minimal \
  env.eval.video_cfg.save_video=false \
  actor.micro_batch_size="$PIRL_MICRO_BATCH_SIZE" actor.global_batch_size="$PIRL_GLOBAL_BATCH_SIZE" \
  actor.model.openpi.noise_method=flow_sde actor.model.openpi.noise_level=0.5 \
  actor.model.openpi.joint_logprob=false algorithm.entropy_bonus=0.0 \
  "${extra_overrides[@]}"

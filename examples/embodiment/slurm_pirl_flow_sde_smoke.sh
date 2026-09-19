#!/usr/bin/env bash
#SBATCH --job-name=pirl_fsd_smoke
# acd_ue has the same ACD1 GPU pool as acd_u, but higher scheduling priority.
# Do not use debug: its per-user GPU QoS is already occupied by another job.
#SBATCH --partition=acd_ue
# ACD1-54 is the only node on which this cluster/environment combination has
# completed a real ManiSkill RGB rollout.  Other sampled nodes hit SAPIEN's
# Vulkan ErrorDeviceLost even at two environments, so preserve reproducibility
# by pinning the known-good renderer host and serialize our jobs on it.
#SBATCH --nodelist=ACD1-54
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
# SAPIEN otherwise guesses an incomplete system ICD on several ACD hosts.
# Pin the ICD shipped with the same Python environment as ManiSkill; this is
# required before Ray forks its EnvWorker process, not after it has crashed.
SAPIEN_ICD=/data/user/leviccdong/EKSF/env_pirl_pi05/lib/python3.10/site-packages/sapien/vulkan_library/nvidia_icd.json
test -f "$SAPIEN_ICD"
export VK_DRIVER_FILES="$SAPIEN_ICD"
export VK_ICD_FILENAMES="$SAPIEN_ICD"
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
# The historical Flow-SDE overlay nested the stock π0.5 root, whose Hydra
# searchpath is legal only for a primary config.  Reuse the primary Event-SMDP
# config as a schema carrier, then explicitly disable every event-only path
# below; this leaves the official action-level GAE/PPO baseline as the only
# active credit estimator.
PIRL_CONFIG_NAME="${PIRL_CONFIG_NAME:-maniskill_ppo_openpi_pi05_event_smdp}"
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
# When more than one GPU is allocated, keep the policy/rollout colocated for
# the stock patch weight sync but reserve a physically separate GPU for
# ManiSkill.  This preserves the πRL algorithm and observations; it only
# prevents SAPIEN camera buffers from competing with the two π0.5 copies.
# Empty keeps the original official colocated placement.
PIRL_ENV_GPU="${PIRL_ENV_GPU:-}"
# ManiSkill supports a CPU rendering backend while retaining GPU physics.  It
# is a compatibility fallback for hosts whose Vulkan device resets during
# batched CUDA-camera allocation; it keeps the same RGB+segmentation
# observation, but is intentionally benchmarked for throughput before use.
PIRL_RENDER_BACKEND="${PIRL_RENDER_BACKEND:-}"

extra_overrides=()
if [[ -n "$PIRL_REWARD_TYPE" ]]; then
  extra_overrides+=("algorithm.reward_type=$PIRL_REWARD_TYPE")
fi

placement_overrides=()
if [[ -n "$PIRL_ENV_GPU" ]]; then
  placement_overrides+=(
    "~cluster.component_placement"
    "+cluster.component_placement={actor: 0, rollout: 0, env: ${PIRL_ENV_GPU}}"
  )
fi

render_overrides=()
if [[ -n "$PIRL_RENDER_BACKEND" ]]; then
  render_overrides+=(
    "+env.train.init_params.render_backend=${PIRL_RENDER_BACKEND}"
    "+env.eval.init_params.render_backend=${PIRL_RENDER_BACKEND}"
  )
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
  env.train.init_params.render_mode=sensors env.eval.init_params.render_mode=sensors \
  env.eval.video_cfg.save_video=false \
  actor.micro_batch_size="$PIRL_MICRO_BATCH_SIZE" actor.global_batch_size="$PIRL_GLOBAL_BATCH_SIZE" \
  actor.model.openpi.noise_method=flow_sde actor.model.openpi.noise_level=0.5 \
  actor.model.openpi.joint_logprob=false algorithm.entropy_bonus=0.0 \
  algorithm.adv_type=gae algorithm.loss_type=actor_critic algorithm.reward_type=action_level \
  env.train.event_oracle.enabled=false \
  "${placement_overrides[@]}" \
  "${render_overrides[@]}" \
  "${extra_overrides[@]}"

#!/usr/bin/env bash
#SBATCH --job-name=pirl_pi05_formal
#SBATCH --partition=acd_ue
#SBATCH --nodelist=ACD1-54
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=/data/user/leviccdong/EKSF/outputs/pirl_pi05_formal_%j.out
#SBATCH --error=/data/user/leviccdong/EKSF/outputs/pirl_pi05_formal_%j.err

# ACD1-54 is the only host with two completed RGB+segmentation πRL rollouts
# in this installation.  Keep all workers on its one assigned H100 until a
# cross-node ManiSkill/SAPIEN validation is available; a generic multi-GPU
# allocation has repeatedly stalled during GPU scene construction elsewhere.
set -euo pipefail

# Each Slurm allocation must own its Ray runtime.  Without this, Ray's
# address=auto discovery attaches simultaneous seeds to the first seed's
# local head and schedules all π0.5 actors on one GPU.
unset RAY_ADDRESS
export RLINF_LOCAL_RAY=1
export RLINF_RAY_TMPDIR="/tmp/rlinf_ray_${SLURM_JOB_ID:?}"

export PIRL_MAX_EPOCHS="${PIRL_MAX_EPOCHS:-100}"
export PIRL_MAX_STEPS="${PIRL_MAX_STEPS:-100}"
export PIRL_SAVE_INTERVAL="${PIRL_SAVE_INTERVAL:-25}"
export PIRL_VAL_INTERVAL="${PIRL_VAL_INTERVAL:-25}"
export PIRL_TRAIN_ENVS="${PIRL_TRAIN_ENVS:-2}"
export PIRL_EVAL_ENVS="${PIRL_EVAL_ENVS:-1}"
export PIRL_TRAIN_EPISODE_STEPS="${PIRL_TRAIN_EPISODE_STEPS:-80}"
export PIRL_EVAL_EPISODE_STEPS="${PIRL_EVAL_EPISODE_STEPS:-80}"
export PIRL_TRAIN_ROLLOUT_STEPS="${PIRL_TRAIN_ROLLOUT_STEPS:-80}"
export PIRL_EVAL_ROLLOUT_STEPS="${PIRL_EVAL_ROLLOUT_STEPS:-80}"
export PIRL_TRAIN_ROLLOUT_EPOCHS="${PIRL_TRAIN_ROLLOUT_EPOCHS:-1}"
export PIRL_EVAL_ROLLOUT_EPOCHS="${PIRL_EVAL_ROLLOUT_EPOCHS:-8}"
export PIRL_MICRO_BATCH_SIZE="${PIRL_MICRO_BATCH_SIZE:-1}"
export PIRL_GLOBAL_BATCH_SIZE="${PIRL_GLOBAL_BATCH_SIZE:-2}"
# Do not set PIRL_ENV_GPU: actor, rollout and ManiSkill intentionally share
# the sole verified H100.  The local-GPU normalization keeps all APIs on 0.
export PIRL_SIM_BACKEND=physx_cuda:0
export PIRL_RENDER_BACKEND=sapien_cuda:0
export PIRL_LOG_PATH="${PIRL_LOG_PATH:-/data/user/leviccdong/EKSF/outputs/pirl_pi05_formal}"
export PIRL_EXPERIMENT_NAME="${PIRL_EXPERIMENT_NAME:-official_pi05_flow_sde_gae_formal}"

exec bash /data/user/leviccdong/EKSF/code/RLinf-piRL/examples/embodiment/slurm_pirl_flow_sde_smoke.sh

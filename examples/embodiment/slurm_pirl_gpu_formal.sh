#!/usr/bin/env bash
#SBATCH --job-name=pirl_pi05_formal
#SBATCH --partition=acd_ue
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=/data/user/leviccdong/EKSF/outputs/pirl_pi05_formal_%j.out
#SBATCH --error=/data/user/leviccdong/EKSF/outputs/pirl_pi05_formal_%j.err

# Normal-horizon official π0.5 + πRL baseline.  The actor and rollout share
# one GPU; ManiSkill's RGB+segmentation renderer has the other.  Inside the
# isolated environment worker, ManiskillEnv maps both SAPIEN and PhysX to the
# worker-local cuda:0 (rather than the physical Slurm index).
set -euo pipefail

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
# GPU 1 is isolated for EnvGroup.  The adapter converts its local namespace
# to physx_cuda:0 + sapien_cuda:0 before gym.make().
export PIRL_ENV_GPU=1
export PIRL_SIM_BACKEND=physx_cuda:0
export PIRL_RENDER_BACKEND=sapien_cuda:0
export PIRL_LOG_PATH="${PIRL_LOG_PATH:-/data/user/leviccdong/EKSF/outputs/pirl_pi05_formal}"
export PIRL_EXPERIMENT_NAME="${PIRL_EXPERIMENT_NAME:-official_pi05_flow_sde_gae_formal}"

exec bash /data/user/leviccdong/EKSF/code/RLinf-piRL/examples/embodiment/slurm_pirl_flow_sde_smoke.sh

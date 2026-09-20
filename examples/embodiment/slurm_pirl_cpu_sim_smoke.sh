#!/usr/bin/env bash
#SBATCH --job-name=pirl_cpu_sim
#SBATCH --partition=acd_ue
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:18:00
#SBATCH --output=/data/user/leviccdong/EKSF/outputs/pirl_cpu_sim_%j.out
#SBATCH --error=/data/user/leviccdong/EKSF/outputs/pirl_cpu_sim_%j.err

# Compatibility smoke: π0.5 remains GPU-resident, while ManiSkill uses its
# documented single-env CPU physics/render path so no Vulkan GPU camera group
# is created.  This checks the end-to-end πRL path before GPU throughput work.
set -euo pipefail
export PIRL_SIM_BACKEND=physx_cpu
export PIRL_RENDER_BACKEND=sapien_cpu
export PIRL_TRAIN_ENVS="${PIRL_TRAIN_ENVS:-1}"
export PIRL_EVAL_ENVS="${PIRL_EVAL_ENVS:-1}"
exec bash /data/user/leviccdong/EKSF/code/RLinf-piRL/examples/embodiment/slurm_pirl_flow_sde_smoke.sh

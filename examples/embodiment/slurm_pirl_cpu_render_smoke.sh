#!/usr/bin/env bash
#SBATCH --job-name=pirl_cpu_render
#SBATCH --partition=acd_ue
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:18:00
#SBATCH --output=/data/user/leviccdong/EKSF/outputs/pirl_cpu_render_%j.out
#SBATCH --error=/data/user/leviccdong/EKSF/outputs/pirl_cpu_render_%j.err

# CPU rendering avoids the host-specific Vulkan camera-group reset while the
# environment still uses GPU physics on a dedicated GPU.  Do not pin a node:
# this fallback is deliberately schedulable on the next free H100 pair.
set -euo pipefail
export PIRL_ENV_GPU="${PIRL_ENV_GPU:-1}"
export PIRL_RENDER_BACKEND="${PIRL_RENDER_BACKEND:-sapien_cpu}"
exec bash /data/user/leviccdong/EKSF/code/RLinf-piRL/examples/embodiment/slurm_pirl_flow_sde_smoke.sh

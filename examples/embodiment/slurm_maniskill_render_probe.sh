#!/usr/bin/env bash
#SBATCH --job-name=ms_render_probe
#SBATCH --partition=acd_ue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:03:00
#SBATCH --output=/data/user/leviccdong/EKSF/outputs/ms_render_probe_%j.out
#SBATCH --error=/data/user/leviccdong/EKSF/outputs/ms_render_probe_%j.err

set -euxo pipefail
source /share/anaconda3/bin/activate /data/user/leviccdong/EKSF/env_pirl_pi05
export EMBODIED_PATH=/data/user/leviccdong/EKSF/code/RLinf-piRL/examples/embodiment
export PYTHONPATH=/data/user/leviccdong/EKSF/code/RLinf-piRL
export MANISKILL_ASSET_DIR=/data/user/leviccdong/EKSF/staging/pirl_assets/maniskill_assets
export MS_ASSET_DIR="$MANISKILL_ASSET_DIR"
export PYTHONUNBUFFERED=1
nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader
python - <<'PY'
import os
import sapien

print("VK_ICD_FILENAMES=", os.environ.get("VK_ICD_FILENAMES"))
print("EGL_ICD=", os.environ.get("__EGL_VENDOR_LIBRARY_FILENAMES"))
import rlinf.envs.sim.maniskill.tasks.put_on_in_scene_multi
import gymnasium as gym

env = gym.make(
    "PutOnPlateInScene25Main-v3",
    obs_mode="rgb+segmentation",
    num_envs=1,
    sim_backend="gpu",
    obj_set="train",
    sensor_configs={"shader_pack": "minimal"},
)
print("MANISKILL_RENDER_OK")
env.close()
PY

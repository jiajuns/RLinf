#!/usr/bin/env bash
#SBATCH --job-name=ms_render_probe
#SBATCH --partition=acd_ue
# Keep this probe on the renderer host that previously completed a real
# ManiSkill RGB rollout.  The probe intentionally uses no RL/VLA process, so
# a failure is attributable to ManiSkill/SAPIEN camera allocation alone.
#SBATCH --nodelist=ACD1-54
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
SAPIEN_ICD=/data/user/leviccdong/EKSF/env_pirl_pi05/lib/python3.10/site-packages/sapien/vulkan_library/nvidia_icd.json
test -f "$SAPIEN_ICD"
export VK_DRIVER_FILES="$SAPIEN_ICD"
export VK_ICD_FILENAMES="$SAPIEN_ICD"
nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader
python - <<'PY'
import os
import sapien

print("VK_ICD_FILENAMES=", os.environ.get("VK_ICD_FILENAMES"))
print("EGL_ICD=", os.environ.get("__EGL_VENDOR_LIBRARY_FILENAMES"))
import rlinf.envs.sim.maniskill.tasks.put_on_in_scene_multi
import gymnasium as gym

num_envs = int(os.environ.get("PIRL_PROBE_ENVS", "2"))
render_backend = os.environ.get("PIRL_RENDER_BACKEND", "sapien_cuda:0")
print(f"probe num_envs={num_envs} render_backend={render_backend}")

env = gym.make(
    "PutOnPlateInScene25Main-v3",
    obs_mode="rgb+segmentation",
    num_envs=num_envs,
    sim_backend="gpu",
    render_backend=render_backend,
    obj_set="train",
    sensor_configs={"shader_pack": "minimal"},
)
env.reset(seed=0)
print("MANISKILL_RENDER_OK")
env.close()
PY

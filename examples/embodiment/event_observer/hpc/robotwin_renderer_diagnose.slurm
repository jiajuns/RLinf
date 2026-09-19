#!/bin/bash
#SBATCH --job-name=rw_renderer_diag
#SBATCH --partition=acd_u
#SBATCH --nodelist=ACD1-54
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/data/user/leviccdong/EKSF/outputs/robotwin_event_observer/renderer_diag_%j.out
#SBATCH --error=/data/user/leviccdong/EKSF/outputs/robotwin_event_observer/renderer_diag_%j.err

set -euxo pipefail
source /share/anaconda3/bin/activate
conda activate /data/user/leviccdong/EKSF/env_etsf
SAPIEN_ICD=/data/user/leviccdong/EKSF/env_etsf/lib/python3.10/site-packages/sapien/vulkan_library/nvidia_icd.json
test -f "$SAPIEN_ICD"
export VK_DRIVER_FILES="$SAPIEN_ICD"
export VK_ICD_FILENAMES="$SAPIEN_ICD"
echo "host=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true
find /usr/share/vulkan /etc/vulkan -type f -name '*nvidia*.json' -o -name '*nvidia*.so' 2>/dev/null || true
command -v vulkaninfo && vulkaninfo --summary || true
python - <<'PY'
import os
import sapien
print("sapien", sapien.__version__)
print("VK_DRIVER_FILES", os.environ.get("VK_DRIVER_FILES"))
print("VK_ICD_FILENAMES", os.environ.get("VK_ICD_FILENAMES"))
try:
    renderer = sapien.SapienRenderer()
    print("sapien_renderer=OK", renderer)
except Exception as exc:
    print("sapien_renderer=FAIL", type(exc).__name__, str(exc))
    raise
PY

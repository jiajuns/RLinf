#!/bin/bash
# Build the locally mirrored SAM 3.1 checkpoint once on an allocated GPU.
# This deliberately tests no RobotWin episode: it validates the isolated
# inference runtime before a costly offline cache array is submitted.
#SBATCH --job-name=sam31_model_smoke
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=/data/user/leviccdong/EKSF/outputs/sam31_model_smoke_%j.out
#SBATCH --error=/data/user/leviccdong/EKSF/outputs/sam31_model_smoke_%j.err

set -euo pipefail

ROOT=/data/user/leviccdong/EKSF
PYTHON=/share/anaconda3/envs/pytorch271/bin/python
SAM_SOURCE=$ROOT/third_party/sam3_2345a4a
SAM_SITE=$ROOT/third_party/sam3_py312_site
CHECKPOINT=$ROOT/models/sam3.1/sam3.1_multiplex.pt

test -x "$PYTHON"
test -d "$SAM_SOURCE"
test -d "$SAM_SITE"
test -f "$CHECKPOINT"
mkdir -p "$ROOT/outputs"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$SAM_SOURCE:$SAM_SITE"
export PYTHONUNBUFFERED=1

"$PYTHON" - <<PY
import torch
from sam3.model_builder import build_sam3_multiplex_video_predictor

assert torch.cuda.is_available(), "SAM 3.1 smoke must run on an allocated CUDA GPU"
print("cuda", torch.cuda.get_device_name(0), "torch", torch.__version__)
predictor = build_sam3_multiplex_video_predictor(
    checkpoint_path="$CHECKPOINT",
    use_fa3=False,
    compile=False,
    warm_up=False,
    async_loading_frames=False,
)
predictor.eval()
print("sam31_model_build_ok", type(predictor).__name__)
PY

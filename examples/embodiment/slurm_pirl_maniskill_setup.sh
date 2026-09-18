#!/usr/bin/env bash
#SBATCH --job-name=pirl_setup
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=00:25:00
#SBATCH --output=/data/user/leviccdong/EKSF/outputs/pirl_setup_%j.out
#SBATCH --error=/data/user/leviccdong/EKSF/outputs/pirl_setup_%j.err

# Build a private RLinf environment. Do not mutate the shared env_etsf because
# it is used by unrelated running jobs.
set -euo pipefail
ROOT=/data/user/leviccdong/EKSF/code/RLinf-piRL
VENV=/data/user/leviccdong/EKSF/env_pirl_pi05
module load cuda/12.4 gcc/13.3 cmake/3.27.9 2>/dev/null || true
source /share/anaconda3/bin/activate
cd "$ROOT"

if [ ! -x "$VENV/bin/python" ]; then
  # Clone instead of modifying env_etsf: that environment backs active jobs.
  conda create -y -p "$VENV" --clone /data/user/leviccdong/EKSF/env_etsf
fi

conda activate "$VENV"
python -m ensurepip --upgrade
python -m pip install --upgrade pip
python -m pip install 'ray[default]>=2.47.0' 'hydra-core<1.4.0.dev8' \
  'omegaconf>=2.3' 'rlinf-openpi==0.1.1' 'mani_skill==3.0.0b22'
python -m pip install -e . --no-deps
python - <<'PY'
import torch
import ray
import gymnasium
import mani_skill
import rlinf

print("SETUP_OK", torch.__version__, torch.cuda.is_available(), ray.__version__)
PY

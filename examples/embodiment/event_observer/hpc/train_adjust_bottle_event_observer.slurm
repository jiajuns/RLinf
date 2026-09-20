#!/bin/bash
# Train a task-matched frozen-perception Event Observer and initialise V_E.
# The train/validation split is episode-level only because this first phase is
# a single task; its validation metrics are an overfit diagnostic, not a
# cross-task generalization result.
#SBATCH --job-name=rw_adjust_event_pretrain
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/data/user/leviccdong/EKSF/outputs/rw_adjust_event_pretrain_%j.out
#SBATCH --error=/data/user/leviccdong/EKSF/outputs/rw_adjust_event_pretrain_%j.err

set -euo pipefail
ROOT=/data/user/leviccdong/EKSF
REPO=$ROOT/code/RLinf-piRL
CACHE_ROOT=${EVENT_CACHE_ROOT:-$ROOT/data/robotwin_adjust_bottle_event_cache_v1}
OUTPUT=${EVENT_PRETRAIN_OUTPUT:-$ROOT/outputs/robotwin_adjust_bottle_event_pretrain_v1}
MIN_EPISODES=${MIN_EPISODES:-80}

source /share/anaconda3/bin/activate
conda activate "$ROOT/env_pirl_pi05"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
COUNT=$(find "$CACHE_ROOT" -maxdepth 1 -type f -name '*.npz' | wc -l)
if (( COUNT < MIN_EPISODES )); then
  echo "need at least $MIN_EPISODES validated caches, found $COUNT" >&2
  exit 2
fi
cd "$REPO"
python -u examples/embodiment/event_observer/train_event_observer.py \
  --cache "$CACHE_ROOT" --output "$OUTPUT" --epochs 40 --batch-size 16 --lr 3e-4 \
  --gamma .99 --seed 0 --episode-validation-fraction .1
test -s "$OUTPUT/best.pt"
python - "$OUTPUT/best.pt" <<'PY'
import sys
import torch
checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
required = {"observer", "event_value", "feature_dim", "metrics"}
missing = required.difference(checkpoint)
if missing:
    raise SystemExit(f"incomplete checkpoint: {sorted(missing)}")
print({"checkpoint": sys.argv[1], "feature_dim": checkpoint["feature_dim"],
       "metrics": checkpoint["metrics"]})
PY

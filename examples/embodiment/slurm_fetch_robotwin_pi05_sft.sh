#!/usr/bin/env bash
#SBATCH --job-name=fetch_rw_pi05_sft
#SBATCH --partition=acd_u
# The local ACD policy rejects all jobs without an explicit GPU request, even
# purely I/O-bound model downloads.  The process does not initialize CUDA and
# releases this allocation immediately after the checkpoint is complete.
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=12:00:00
#SBATCH --output=/data/user/leviccdong/EKSF/outputs/fetch_rw_pi05_sft_%j.out
#SBATCH --error=/data/user/leviccdong/EKSF/outputs/fetch_rw_pi05_sft_%j.err

# Download is a scheduled, restartable prerequisite rather than an interactive
# login-node transfer.  A complete directory is published atomically only once
# the Hugging Face snapshot succeeds, so a formal job never consumes a partial
# checkpoint.
set -euo pipefail

source /share/anaconda3/bin/activate /data/user/leviccdong/EKSF/env_pirl_pi05
export HF_HOME="${HF_HOME:-/data/user/leviccdong/EKSF/cache/huggingface}"
export HF_HUB_ENABLE_HF_TRANSFER=1

MODEL_ID="RLinf/RLinf-Pi05-RoboTwin-SFT-adjust_bottle"
TARGET="/data/user/leviccdong/EKSF/models/RLinf-Pi05-RoboTwin-SFT-adjust_bottle"
PARTIAL="${TARGET}.partial-${SLURM_JOB_ID:?}"

if [[ -f "${TARGET}/config.json" ]]; then
  echo "checkpoint already complete: ${TARGET}"
  exit 0
fi

rm -rf "${PARTIAL}"
mkdir -p "${PARTIAL}"

# Prefer the canonical hub.  Retry once through the documented China mirror
# only if the direct route fails.
if ! hf download "${MODEL_ID}" --local-dir "${PARTIAL}"; then
  rm -rf "${PARTIAL}"
  mkdir -p "${PARTIAL}"
  HF_ENDPOINT=https://hf-mirror.com hf download "${MODEL_ID}" --local-dir "${PARTIAL}"
fi

test -f "${PARTIAL}/config.json"
test -d "${PARTIAL}/physical-intelligence"
rm -rf "${TARGET}"
mv "${PARTIAL}" "${TARGET}"
echo "checkpoint ready: ${TARGET}"

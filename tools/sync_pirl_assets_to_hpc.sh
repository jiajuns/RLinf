#!/usr/bin/env bash
# Continuously stage local Hugging Face downloads to HPC without exposing a
# partial checkpoint as a model path. Safe to restart: rsync append-verify
# resumes the transfer and validates completed blocks.
set -euo pipefail

LOCAL_ROOT=/home/jj/pirl_assets
REMOTE_ROOT=/data/user/leviccdong/EKSF/staging/pirl_assets
MODEL_DIR="$LOCAL_ROOT/RLinf-Pi05-ManiSkill-25Main-SFT"
REMOTE_HOST=hpc

while true; do
  mkdir -p "$LOCAL_ROOT"
  ssh "$REMOTE_HOST" "mkdir -p $REMOTE_ROOT/pi05_sft $REMOTE_ROOT/runtime_assets"
  if [[ -f "$MODEL_DIR/model.safetensors" ]]; then
    rsync -a --partial --append-verify "$MODEL_DIR/" "$REMOTE_HOST:$REMOTE_ROOT/pi05_sft/"
    ssh "$REMOTE_HOST" "mkdir -p /data/user/leviccdong/EKSF/models/RLinf-Pi05-ManiSkill-25Main-SFT && rsync -a --delete $REMOTE_ROOT/pi05_sft/ /data/user/leviccdong/EKSF/models/RLinf-Pi05-ManiSkill-25Main-SFT/"
  else
    partial=$(find "$MODEL_DIR/.cache/huggingface/download" -name '*.incomplete' -type f -print -quit 2>/dev/null || true)
    if [[ -n "$partial" ]]; then
      rsync -a --partial --append-verify "$partial" "$REMOTE_HOST:$REMOTE_ROOT/pi05_sft/model.safetensors.partial"
    fi
  fi
  if [[ -d "$LOCAL_ROOT/runtime_assets" ]]; then
    rsync -a --partial --append-verify "$LOCAL_ROOT/runtime_assets/" "$REMOTE_HOST:$REMOTE_ROOT/runtime_assets/"
  fi
  if [[ -d "$LOCAL_ROOT/maniskill_assets" ]]; then
    rsync -a --partial --append-verify "$LOCAL_ROOT/maniskill_assets/" "$REMOTE_HOST:$REMOTE_ROOT/maniskill_assets/"
  fi
  sleep 30
done

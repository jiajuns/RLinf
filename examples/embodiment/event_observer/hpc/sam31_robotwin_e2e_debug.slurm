#!/bin/bash
# One short, end-to-end RGB-only SAM 3.1 propagation validation.
# The semantic text is used exactly once to propose an anchor on frame 0.
# All later frames use the resulting visual box anchor plus video propagation.
#SBATCH --job-name=sam31_rw_e2e
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/data/user/leviccdong/EKSF/outputs/sam31_robotwin_e2e_%j.out
#SBATCH --error=/data/user/leviccdong/EKSF/outputs/sam31_robotwin_e2e_%j.err

set -euo pipefail
ROOT=/data/user/leviccdong/EKSF
EPISODE=$ROOT/data/robotwin_event_observer_v2/renderer_smoke_633757/episodes/move_can_pot/aloha-agilex/clean/seed_0.hdf5
FRAMES=$ROOT/tmp/sam31_e2e_${SLURM_JOB_ID}
RESULT=$ROOT/outputs/sam31_robotwin_e2e_${SLURM_JOB_ID}.npz
SAM_PYTHON=/share/anaconda3/envs/pytorch271/bin/python
SAM_SOURCE=$ROOT/third_party/sam3_2345a4a
SAM_SITE=$ROOT/third_party/sam3_py312_site
CHECKPOINT=$ROOT/models/sam3.1/sam3.1_multiplex.pt

test -f "$EPISODE"
test -f "$CHECKPOINT"
mkdir -p "$FRAMES" "$ROOT/outputs"
trap 'rm -rf "$FRAMES"' EXIT

# This archive's left wrist starts with no moving object in view.  The right
# wrist has a clear moving-object view; the head view is reserved for the
# later object--target relation stream.
CAMERA=right_camera
ANCHOR_PROMPT="blue jar"
# Decode four cached RGB-only wrist frames with the replay/materialization env.
source /share/anaconda3/bin/activate
conda activate "$ROOT/env_etsf"
python - "$EPISODE" "$FRAMES" "$CAMERA" <<'PY'
import io
import sys
import h5py
from PIL import Image

episode, out_dir, camera = sys.argv[1:]
with h5py.File(episode, "r") as handle:
    frames = handle[f"rgb/{camera}"]
    for index in range(4):
        encoded = bytes(frames[index])
        image = Image.open(io.BytesIO(encoded)).convert("RGB")
        image.save(f"{out_dir}/{index:05d}.jpg", quality=95)
PY
conda deactivate

export PYTHONNOUSERSITE=1
export PYTHONPATH="$SAM_SOURCE:$SAM_SITE"
export PYTHONUNBUFFERED=1
"$SAM_PYTHON" - "$FRAMES" "$CHECKPOINT" "$RESULT" "$ANCHOR_PROMPT" <<'PY'
import sys
import numpy as np
import torch
from sam3.model_builder import build_sam3_multiplex_video_predictor

frames, checkpoint, result, anchor_prompt = sys.argv[1:]
assert torch.cuda.is_available(), "allocated GPU is required"
predictor = build_sam3_multiplex_video_predictor(
    checkpoint_path=checkpoint, use_fa3=False, compile=False,
    warm_up=False, async_loading_frames=False,
)
# The public base predictor forwards this legacy keyword, while the multiplex
# tracker in the pinned SAM 3.1 source does not accept it.  It is a CPU
# offload flag and has no semantic effect in this single-GPU job.
init_state = predictor.model.init_state
def init_state_compat(**kwargs):
    kwargs.pop("offload_state_to_cpu", None)
    return init_state(**kwargs)
predictor.model.init_state = init_state_compat
session = predictor.handle_request({"type": "start_session", "resource_path": frames})
session_id = session["session_id"]
# One proposal only.  In this pinned multiplex API, a subsequent standalone
# box request resets text grounding and produces an empty stream.  Keeping the
# one frame-0 grounding request in the session is the supported anchor mode;
# it is never issued again on later frames.
proposal = predictor.handle_request({
    "type": "add_prompt", "session_id": session_id, "frame_index": 0, "text": anchor_prompt,
})["outputs"]
probs = np.asarray(proposal["out_probs"], dtype=np.float32)
if len(probs) == 0:
    raise RuntimeError("SAM 3.1 found no frame-0 object for the single anchor prompt")
outputs = []
for response in predictor.handle_stream_request({"type": "propagate_in_video", "session_id": session_id}):
    out = response["outputs"]
    outputs.append((response["frame_index"], out["out_probs"], out["out_boxes_xywh"], out["out_binary_masks"]))
if len(outputs) != 4:
    raise RuntimeError(f"expected 4 propagated frames, received {len(outputs)}")
if not any(len(item[1]) for item in outputs):
    raise RuntimeError("SAM 3.1 propagation returned no tracked candidates")
np.savez_compressed(
    result,
    frame_index=np.asarray([item[0] for item in outputs], dtype=np.int64),
    object_count=np.asarray([len(item[1]) for item in outputs], dtype=np.int64),
    max_confidence=np.asarray([np.max(item[1]) if len(item[1]) else 0.0 for item in outputs], dtype=np.float32),
    boxes=np.asarray([item[2] for item in outputs], dtype=object),
    masks=np.asarray([item[3] for item in outputs], dtype=object),
)
print("sam31_rgb_anchor_propagation_ok", result)
PY

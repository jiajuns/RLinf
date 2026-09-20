#!/bin/bash
# Gate SAM teacher tracks before they can enter the RGB/proprio Event cache.
# The audit uses only perception-track geometry and the episode/track hashes;
# simulator oracle labels are deliberately not consulted here.
#
# Submit after the matching SAM array, then pass SAM_ACCEPTED_LIST to the
# cache-array job.  This keeps the quality boundary explicit and reproducible.
#SBATCH --job-name=rw_adjust_sam_audit
#SBATCH --partition=acd_u
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/data/user/leviccdong/EKSF/outputs/rw_adjust_sam_audit_%j.out
#SBATCH --error=/data/user/leviccdong/EKSF/outputs/rw_adjust_sam_audit_%j.err

set -euo pipefail

ROOT=/data/user/leviccdong/EKSF
REPO=$ROOT/code/RLinf-piRL
EPISODE_LIST=${SAM_EPISODE_LIST:?set the matching successful HDF manifest}
TRACK_ROOT=${SAM_TRACK_ROOT:-$ROOT/data/robotwin_adjust_bottle_event_tracks_v1}
ACCEPTED_LIST=${SAM_ACCEPTED_LIST:?set accepted-manifest output}
REPORT=${SAM_AUDIT_REPORT:?set audit-report output}

source /share/anaconda3/bin/activate "$ROOT/env_etsf"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

python -u "$REPO/examples/embodiment/event_observer/audit_adjust_bottle_sam_tracks.py" \
  --episodes "$EPISODE_LIST" --tracks "$TRACK_ROOT" \
  --accepted "$ACCEPTED_LIST" --report "$REPORT"
test -s "$ACCEPTED_LIST"

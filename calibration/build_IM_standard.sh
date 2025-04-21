#!/usr/bin/env bash
#
# run_build_im.sh
# A simple script to invoke build_IM.py with fixed parameters.

set -euo pipefail

# Resolve script directory (so it works even if you cd elsewhere)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Path to the Python script
PY_SCRIPT="$SCRIPT_DIR/build_IM.py"

# Fixed parameters
CAM_FPS=200 
#1000
CAM_GAIN=1 
#10
BEAM_ID="1,2,3,4"
POKE_AMP=0.05
LO=2
PHASEMASK="H3"
SIGNAL_SPACE="dm"
DM_FLAT="baldr"

# (Optional) Activate your Python virtualenv here, e.g.:
# source /path/to/venv/bin/activate

echo "Running build_IM.py with:"
echo "  cam_fps      = $CAM_FPS"
echo "  cam_gain     = $CAM_GAIN"
echo "  beam_id      = $BEAM_ID"
echo "  poke_amp     = $POKE_AMP"
echo "  LO           = $LO"
echo "  phasemask    = $PHASEMASK"
echo "  signal_space = $SIGNAL_SPACE"
echo "  DM_flat      = $DM_FLAT"
echo

python "$PY_SCRIPT" \
  --cam_fps    "$CAM_FPS" \
  --cam_gain   "$CAM_GAIN" \
  --beam_id    "$BEAM_ID" \
  --poke_amp   "$POKE_AMP" \
  --LO         "$LO" \
  --phasemask  "$PHASEMASK" \
  --signal_space "$SIGNAL_SPACE" \
  --DM_flat    "$DM_FLAT"

echo "Done."
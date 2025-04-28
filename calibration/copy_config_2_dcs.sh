#!/usr/bin/env bash
#
# copy_baldr_configs.sh
# Copy baldr config files for beams 1–4 into the baldr repo directory.

set -euo pipefail

SRC_DIR="/home/asg/Progs/repos/asgard-alignment/config_files"
DEST_DIR="/usr/local/etc/baldr/"
#"/home/asg/Progs/repos/dcs/baldr"
#1 2 3 4
for beam_id in 1 2 3 4; do
    SRC_FILE="${SRC_DIR}/baldr_config_${beam_id}.toml"
    if [[ -f "$SRC_FILE" ]]; then
        cp "$SRC_FILE" "$DEST_DIR/"
        echo "Copied $SRC_FILE → $DEST_DIR/"
    else
        echo "Warning: $SRC_FILE not found, skipping." >&2
    fi
done
#!/bin/bash

# Check if at least one beam number is provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <beam_number1> [beam_number2 ...]"
    exit 1
fi

# Loop over all provided beam numbers
for beam in "$@"; do
    src="config_files/baldr_config_${beam}.toml"
    dst="config_files/baldr_config_${beam}_stable.toml"

    if [ -f "$src" ]; then
        cp "$src" "$dst"
        echo "Copied $src → $dst"
    else
        echo "Warning: $src does not exist, skipping."
    fi
done
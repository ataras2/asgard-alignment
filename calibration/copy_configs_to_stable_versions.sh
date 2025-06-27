#!/bin/bash

# Check if at least two arguments are provided: mode + at least one beam
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <mode: bright|faint> <beam_number1> [beam_number2 ...]"
    exit 1
fi

# First argument is mode
mode="$1"
shift  # Shift all arguments left, so $@ now contains only beam numbers

# Validate mode
if [[ "$mode" != "bright" && "$mode" != "faint" ]]; then
    echo "Error: mode must be 'bright' or 'faint'."
    exit 1
fi

# Get current full timestamp: YYYY-MM-DDTHH-MM-SS
full_timestamp=$(date +%F"T"%H-%M-%S)

# Extract just the date part: YYYY-MM-DD
today=${full_timestamp%%T*}

# Base destination directory
base_dst_dir="/usr/local/etc/baldr/rtc_config/${today}"

# Create base destination folder if it doesn't exist
mkdir -p "${base_dst_dir}"

# Loop over all provided beam numbers
for beam in "$@"; do
    src="/usr/local/etc/baldr/baldr_config_${beam}.toml"
    dst="${base_dst_dir}/baldr_config_${beam}_${mode}_${full_timestamp}.toml"

    if [ -f "$src" ]; then
        cp "$src" "$dst"
        echo "Copied $src → $dst"
    else
        echo "Warning: $src does not exist, skipping."
    fi
done
# #!/bin/bash

# # Check if at least one beam number is provided
# if [ "$#" -lt 1 ]; then
#     echo "Usage: $0 <beam_number1> [beam_number2 ...]"
#     exit 1
# fi

# # Get current full timestamp: YYYY-MM-DDTHH-MM-SS
# full_timestamp=$(date +%F"T"%H-%M-%S)

# # Extract just the date part: YYYY-MM-DD
# today=${full_timestamp%%T*}

# # Base destination directory
# base_dst_dir="/usr/local/etc/baldr/rtc_config/${today}"

# # Create base destination folder if it doesn't exist
# mkdir -p "${base_dst_dir}"

# # Loop over all provided beam numbers
# for beam in "$@"; do
#     src="/usr/local/etc/baldr/baldr_config_${beam}.toml"
#     dst="${base_dst_dir}/baldr_config_${beam}_${full_timestamp}.toml"

#     if [ -f "$src" ]; then
#         cp "$src" "$dst"
#         echo "Copied $src → $dst"
#     else
#         echo "Warning: $src does not exist, skipping."
#     fi
# done

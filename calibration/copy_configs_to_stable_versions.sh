#!/bin/bash

# Check if at least one beam number is provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <beam_number1> [beam_number2 ...]"
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
    dst="${base_dst_dir}/baldr_config_${beam}_${full_timestamp}.toml"

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

# # Loop over all provided beam numbers
# for beam in "$@"; do
#     src="/usr/local/etc/baldr/baldr_config_${beam}.toml"
#     #"config_files/baldr_config_${beam}.toml"
#     dst="/usr/local/etc/baldr/rtc_config/baldr_config_${beam}_stable.toml"
#     #"config_files/baldr_config_${beam}_stable.toml"

#     if [ -f "$src" ]; then
#         cp "$src" "$dst"
#         echo "Copied $src → $dst"
#     else
#         echo "Warning: $src does not exist, skipping."
#     fi
# done
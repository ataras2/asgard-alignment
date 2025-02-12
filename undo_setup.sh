#!/bin/bash
# usage: sudo bash undo_setup.sh

# Exit immediately if a command exits with a non-zero status
set -e

# Check if conda is installed
if command -v conda &> /dev/null
then
    # Remove conda environment if it exists
    if conda env list | grep -q 'asgard'; then
        conda env remove -n asgard -y
    fi
else
    echo "conda could not be found, skipping conda environment removal"
fi

# Remove the asgard-alignment directory if it exists
if [ -d ~/Documents/asgard-alignment/ ]; then
    sudo rm -rf ~/Documents/asgard-alignment/
fi

# Remove the miniconda3 directory if it exists
if [ -d ~/miniconda3/ ]; then
    sudo rm -rf ~/miniconda3/
fi

echo "Setup undone successfully."
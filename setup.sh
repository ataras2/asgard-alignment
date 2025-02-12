#!/bin/bash
# usage: sudo bash setup.sh

# Exit immediately if a command exits with a non-zero status
set -e

# Get the username of the user who invoked sudo
USER_HOME=$(eval echo ~${SUDO_USER})

# ==============================================================================
# Pre-checks
# ==============================================================================


# FLI?


# ==============================================================================
# System Update and Package Installation
# ==============================================================================

# Update package list and install necessary packages
echo "Updating package list and installing necessary packages..."

apt-get update 
# DCS
apt-get install -y \
    nlohmann-json3-dev \
    libfmt-dev \
    libzmq3-dev \
    libboost-all-dev

# spinview - see readme in spinnaker-4.2.0.46-amd64-22.04
apt-get install -y \
    libusb-1.0-0 \
    libavcodec58 \
    libavformat58 \
    libswscale5 \
    libswresample3 \
    libavutil56 \
    qt5-default

# general
apt-get install -y \
    cmake \
    git \
    wget \
    unzip \
    curl 

# TODO

# ==============================================================================
# Environment Setup
# ==============================================================================

# Allow USB device access
echo "Allowing USB device access..."
sudo usermod -a -G dialout $USER


# ==============================================================================
# Miniconda Installation
# ==============================================================================

# Check if Miniconda is already installed
if [[ -d "/${USER_HOME}/miniconda3" ]]; then
    echo "Miniconda is already installed."
else
    mkdir -p ${USER_HOME}/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ${USER_HOME}/miniconda3/miniconda.sh
    sudo bash ${USER_HOME}/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    rm ${USER_HOME}/miniconda3/miniconda.sh
    source ${USER_HOME}/miniconda3/bin/activate
fi


# ==============================================================================
# Visual Studio Code Insiders Installation
# ==============================================================================

# Check if Visual Studio Code Insiders is already installed
if ! command -v code-insiders &> /dev/null; then
    echo "Installing Visual Studio Code Insiders..."
    wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > microsoft.gpg
    install -o root -g root -m 644 microsoft.gpg /etc/apt/trusted.gpg.d/
    sh -c 'echo "deb [arch=amd64] https://packages.microsoft.com/repos/code stable main" > /etc/apt/sources.list.d/vscode.list'
    apt-get update
    apt-get install -y code-insiders
    rm microsoft.gpg
else
    echo "Visual Studio Code Insiders is already installed."
fi


# ==============================================================================
# Git Configuration
# ==============================================================================

# Configure Git username and email
echo "Configuring Git username and email..."
sudo -u ${SUDO_USER} git config --global user.name "ataras2"
sudo -u ${SUDO_USER} git config --global user.email "ataras2@gmail.com"

# Check if the repository already exists
REPO_DIR="${USER_HOME}/Documents/asgard-alignment"
if [[ -d "${REPO_DIR}" ]]; then
    echo "Repository already exists at ${REPO_DIR}."
else
    echo "Cloning Git repository..."
    # Replace <your-username> and <your-repo> with your actual repository URL
    sudo -u ${SUDO_USER} git clone https://github.com/ataras2/asgard-alignment.git ${REPO_DIR}
fi


# ==============================================================================
# System Configuration
# ==============================================================================

# Prevent the screen from turning blank
echo "Preventing the screen from turning blank..."
echo "xset s off" >> /etc/profile
echo "xset -dpms" >> /etc/profile
echo "xset s noblank" >> /etc/profile



# ==============================================================================
# Spinview setup
# ==============================================================================
# again following the readme

#sudo sh install_spinnaker.sh
#sudo apt install ethtool
#sudo ./gev_nettweak eth0 # TODO check ethernet port
 
# ==============================================================================
# Conda Environment Setup
# ==============================================================================
# Set environment variables for conda
export USER_HOME=$USER_HOME
export CONDA_HOME="/home/$USER_HOME/miniconda3"
export PATH="$CONDA_HOME/bin:$PATH"


# init conda
source "$CONDA_HOME/etc/profile.d/conda.sh"


# check if environment exists, if not, make env with 3.10
if ! conda info --envs | grep -q "asg"; then
    sudo -u $USER conda create -y -n asg python=3.10
fi


# activate asg environment
sudo -i -u $USER bash -c "source $CONDA_HOME/etc/profile.d/conda.sh && \
    conda activate asg && \
    pip install -r ${USER_HOME}/Documents/asgard-alignment/requirements.txt \
    pip install spinnaker_python-4.2.0.46/spinnaker_python-4.2.0.46-cp310-cp310-linux_x86_64.whl"
    
# any other custom pip installs here


# ==============================================================================
# No machine configuration
# ==============================================================================

if ! command -v /usr/NX/bin/nxserver &> /dev/null; then
    echo "Installing NoMachine..."
    sudo wget https://download.nomachine.com/download/8.16/Linux/nomachine_8.16.1_1_amd64.deb
    sudo dpkg -i nomachine_8.16.1_1_amd64.deb
    rm nomachine_8.16.1_1_amd64.deb
else
    echo "NoMachine is already installed."
fi


# ==============================================================================
# Desktop icons setup
# ==============================================================================


# things like moving the desktop icons from git repo to start mds, etc


# ==============================================================================
# Conclude
# ==============================================================================
echo "Done! Reboot system to take effect"
#!/bin/bash
# usage: sudo bash setup.sh
# detailed user instructions found at
# https://docs.google.com/document/d/12j7ZFyZ8E72clbRmer_jwnHb85poJUIWlCKUhd8aZ1Q/edit?usp=sharing

# Exit immediately if a command exits with a non-zero status
set -e

sudo echo "Starting setup script..."

export USER_HOME="/home/$(logname)"

# ==============================================================================
# Pre-checks
# ==============================================================================

# Check if the spinnaker python package is present
if [[ -f "$(pwd)/spinnaker_python-4.2.0.46-cp38-cp38-linux_x86_64-20.04.tar.gz" ]]; then
    echo "Extracting spinnaker_python-4.2.0.46-cp38-cp38-linux_x86_64-20.04.tar.gz..."
    mkdir -p spinnaker_python-4.2.0.46
    tar -xzf spinnaker_python-4.2.0.46-cp38-cp38-linux_x86_64-20.04.tar.gz -C spinnaker_python-4.2.0.46
else
    echo "Error: spinnaker_python-4.2.0.46-cp38-cp38-linux_x86_64-20.04.tar.gz file is not present in the current directory."
    exit 1
fi

# Check if the spinnaker SDK package is present
if [[ -f "$(pwd)/spinnaker-4.2.0.46-amd64-20.04-pkg.tar.gz" ]]; then
    echo "Extracting spinnaker-4.2.0.46-amd64-20.04-pkg.tar.gz..."
    tar -xzf spinnaker-4.2.0.46-amd64-20.04-pkg.tar.gz -C .
else
    echo "Error: spinnaker-4.2.0.46-amd64-20.04-pkg.tar.gz file is not present in the current directory."
    exit 1
fi

# FLI?
# TODO search google for pc edt f4 and frind drivers (get link)
wget https://edt.com/downloads/pdv_6-2-0_deb_amd64/
sudo apt-get install dkms
sudo dpkg -i edtpdv_6.2.0_amd64.deb


# TODO BMC stuff
wget https://bostonmicromachines.com/DMSDK/BMC-DMSDK.zip


# TODO also need to setup rc.local to include modprobe for TT motors...

# ==============================================================================
# Spinview setup
# ==============================================================================
# again following the readme

mkdir -p spinnaker-4.2.0.46-amd64
cd spinnaker-4.2.0.46-amd64/
if [[ ! -f "remove_spinnaker.sh" ]]; then
    {
        sudo sh install_spinnaker.sh 
        sudo apt install ethtool
        sudo /opt/spinnaker/bin/./gev_nettweak eno8403
    } || {
        echo "Error occurred during Spinview setup, but continuing with the script..."
    }
else
    echo "Spinnaker is already installed."
fi
cd ..


# ==============================================================================
# Zaber launcher setup
# ==============================================================================

mkdir -p zaber_launcher
cd zaber_launcher
if [[ ! -f "ZaberLauncher.AppImage" ]]; then
    wget https://zaber-launcher-release.s3-us-west-2.amazonaws.com/public/ZaberLauncher.AppImage
    chmod +x ZaberLauncher.AppImage
else
    echo "ZaberLauncher.AppImage already exists."
fi
cd ..

# ==============================================================================
# System Update and Package Installation
# ==============================================================================

# Update package list and install necessary packages
echo "Updating package list and installing necessary packages..."

sudo apt-get update 
# DCS
sudo apt-get install -y \
    nlohmann-json3-dev \
    libfmt-dev \
    libzmq3-dev \
    libboost-all-dev \

# spinview - see readme in spinnaker-4.2.0.46-amd64-22.04
sudo apt-get install -y \
    libusb-1.0-0 \
    libavcodec58 \
    libavformat58 \
    libswscale5 \
    libswresample3 \
    libavutil56 \
    qt5-default

# general
sudo apt-get install -y \
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
if [[ -d "${USER_HOME}/miniconda3" ]]; then
    echo "Miniconda is already installed."
else
    sudo mkdir -p ${USER_HOME}/miniconda3
    sudo wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ${USER_HOME}/miniconda3/miniconda.sh
    sudo bash ${USER_HOME}/miniconda3/miniconda.sh -b -u -p ${USER_HOME}/miniconda3
    sudo rm ${USER_HOME}/miniconda3/miniconda.sh
    source ${USER_HOME}/miniconda3/bin/activate
    echo "source ${USER_HOME}/miniconda3/bin/activate" >> ${USER_HOME}/.bashrc
    echo "conda activate asgard" >> ${USER_HOME}/.bashrc
fi


# ==============================================================================
# Visual Studio Code Insiders Installation
# ==============================================================================

# Check if Visual Studio Code Insiders is already installed
if ! command -v code-insiders &> /dev/null; then
    echo "Installing Visual Studio Code Insiders..."
    sudo wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > microsoft.gpg
    sudo install -o root -g root -m 644 microsoft.gpg /etc/apt/trusted.gpg.d/
    sudo sh -c 'echo "deb [arch=amd64] https://packages.microsoft.com/repos/code stable main" > /etc/apt/sources.list.d/vscode.list'
    sudo apt-get update
    sudo apt-get install -y code-insiders
    rm microsoft.gpg
else
    echo "Visual Studio Code Insiders is already installed."
fi


# ==============================================================================
# Git Configuration
# ==============================================================================

# Configure Git username and email
echo "Configuring Git username and email..."
git config --global user.name "ataras2"
git config --global user.email "ataras2@gmail.com"

# Check if the repository already exists
REPO_DIR="${USER_HOME}/Documents/asgard-alignment"
if [[ -d "${REPO_DIR}" ]]; then
    echo "Repository already exists at ${REPO_DIR}."
else
    echo "Cloning Git repository..."
    git clone https://github.com/ataras2/asgard-alignment.git ${REPO_DIR}
fi


# ==============================================================================
# System Configuration
# ==============================================================================

# Prevent the screen from turning blank
# echo "Preventing the screen from turning blank..."
# sudo echo "xset s off" >> /etc/profile
# sudo echo "xset -dpms" >> /etc/profile
# sudo echo "xset s noblank" >> /etc/profile


# ==============================================================================
# Conda Environment Setup
# ==============================================================================
# Set environment variables for conda
export CONDA_HOME="${USER_HOME}/miniconda3"
export ENV_NAME="asgard"

# init conda
source "$CONDA_HOME/etc/profile.d/conda.sh"

# check if environment exists, if not, make env with Python 3.10
if ! conda info --envs | grep -q "$ENV_NAME"; then
    echo "Creating conda environment '$ENV_NAME' with Python 3.10..."
    conda create -y -n "$ENV_NAME" python=3.10
else
    echo "Conda environment '$ENV_NAME' already exists."
fi

# activate asg environment and install required packages
conda activate "$ENV_NAME" && \
    pip install -r "${USER_HOME}/Documents/asgard-alignment/requirements.txt" && \
    pip install -e .
# any other custom pip installs here


export ENV_NAME="spinview"
if ! conda info --envs | grep -q "$ENV_NAME"; then
    echo "Creating conda environment '$ENV_NAME' with Python 3.8..."
    conda create -y -n "$ENV_NAME" python=3.8
else
    echo "Conda environment '$ENV_NAME' already exists."
fi

# activate asg environment and install required packages
conda activate "$ENV_NAME" && \
    pip install spinnaker_python-4.2.0.46/spinnaker_python-4.2.0.46-cp38-cp38-linux_x86_64.whl && \
    pip install -r "${USER_HOME}/Documents/asgard-alignment/spinview_reqs.txt" 



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
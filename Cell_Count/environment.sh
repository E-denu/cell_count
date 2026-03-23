#!/bin/bash

# Exit on any error
set -e

# Variables
anaconda_url="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
installer_file="Miniconda3_installer.sh"
install_directory="$HOME/miniconda3"

echo "Have a little patience while I download anaconda..."
wget "$anaconda_url" -O "$installer_file"
echo
echo "Your download is complete!!!!"
echo

export CONDA_PLUGINS_AUTO_ACCEPT_TOS=yes

echo "Installing miniconda, This may take a while so grab some caffee..."
bash "$installer_file" -b -p "$install_directory"
echo

echo "Installation complete!!!"
echo

echo "Initializing conda for bash..."
"$install_directory/bin/conda" init bash
echo

echo "Initializing your enironment to recognize conda commands..."
source ~/.bashrc

echo "Cleaning up..."
rm "$installer_file"

echo "activating your conda environment........"
source "$install_directory/bin/activate"

echo "creating a Conda environment for running Cellpose........"
conda create -n cellpose python=3.9 -y

echo "Activating the Conda environment for Cellpose........"
conda activate cellpose

echo "Installing dependencies, please wait........"

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install cellpose

pip install numpy==1.26.4

pip install --force-reinstall numpy==1.26.4 "opencv-python-headless<=4.9.0.80"

pip install matplotlib seaborn scikit-image

echo "Installation complete!"

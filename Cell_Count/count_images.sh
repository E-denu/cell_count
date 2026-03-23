#!/bin/bash

# Exit on any error
set -e

install_directory="$HOME/anaconda3"

echo "activating your conda environment........"
echo
source "$install_directory/bin/activate"

echo "Activating the Conda environment for Cellpose........"
echo
conda activate cellpose_ha

rm -rf ~/.cache/matplotlib

echo "Your environment is ready to count your cells........"
python3 ~/cellpose_project/run.py

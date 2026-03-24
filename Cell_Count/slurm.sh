#!/bin/bash
#SBATCH --job-name=something
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --error=logs/job_%A_%a.err
#SBATCH --array=1
#SBATCH --partition=econ-gpu               
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32   
#SBATCH --mem-per-cpu=2000


source /home/ed488/miniconda3/bin/activate

conda activate cellpose

export LD_LIBRARY_PATH=/home/ed488/miniconda3/envs/cellpose/lib/python3.9/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


rm -rf ~/.cache/matplotlib

echo "Your environment is ready to count your cells........"
python3 -u ~/cellpose_project/run.py

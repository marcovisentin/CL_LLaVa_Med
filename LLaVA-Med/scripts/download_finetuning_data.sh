#!/bin/bash
#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --output=download_finetuning_data.%N.%j.log
#SBATCH --time=0-01:00:00
#SBATCH --chdir=/vol/biomedic3/mv320/projects/VLMs/MEG_x_CL

# Activate venv
source /vol/biomedic3/mv320/projects/VLMs/MEG_x_CL/LLaVA-Med/venv/bin/activate

# Check if cache directory argument is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <cache_directory>"
    exit 1
fi

CACHE_DIR=$1

# Create cache directory if it doesn't exist
mkdir -p "$CACHE_DIR"

# Run the Python script with the provided cache directory
python /vol/biomedic3/mv320/projects/VLMs/MEG_x_CL/download_finetuning_data.py "$CACHE_DIR"
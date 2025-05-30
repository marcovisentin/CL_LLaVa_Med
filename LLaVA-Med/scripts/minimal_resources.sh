#!/bin/bash
#SBATCH --partition=gpus
#SBATCH --gres=gpu:1
#SBATCH --output=download_finetuning_data.%N.%j.log
#SBATCH --time=0-01:00:00
#SBATCH --chdir=/vol/biomedic3/mv320/projects/VLMs/MEG_x_CL

# Activate venv
source /vol/biomedic3/mv320/projects/VLMs/MEG_x_CL/LLaVA-Med/venv/bin/activate

# Run the Python script with the provided cache directory
python /vol/biomedic3/mv320/projects/VLMs/MEG_x_CL/dataset_preparation/create_final_datasets.py 
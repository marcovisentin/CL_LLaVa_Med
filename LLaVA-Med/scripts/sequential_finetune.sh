#!/bin/bash
#SBATCH --partition=gpus48
#SBATCH --gres=gpu:1
#SBATCH --output=llava-med_seq_finetune.%N.%j.log
#SBATCH --time=0-71:59:00
#SBATCH --chdir=/vol/biomedic3/mv320/projects/VLMs/MEG_x_CL/LLaVA-Med

# Manually set huggingface cache in vol (otherwise model is going to be saved in /home and an error will be raised)
export HF_HOME="/vol/biomedic3/mv320/.cache/huggingface"

# Manually set deepspeed cache dir 
export TRITON_CACHE_DIR="/vol/biomedic3/mv320/deepspeed_triton"

# Manally set pip cache
#export PIP_CACHE_DIR="/vol/biomedic3/mv320/.cache/pip"

# Activate venv
source /vol/biomedic3/mv320/projects/VLMs/MEG_x_CL/LLaVA-Med/venv/bin/activate

# Set env variable for CUDA Toolkit
export CUDA_HOME=/vol/cuda/12.3.2
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

python -u /vol/biomedic3/mv320/projects/VLMs/MEG_x_CL/LLaVA-Med/llava/train/sequential_training_pipeline.py --run_all_binary_sequences

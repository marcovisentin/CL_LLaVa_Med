#!/bin/bash
#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --output=llava-med_evaluate.%N.%j.log
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

# Install flash-attn - done (only needed first time)
#pip install flash-attn --no-build-isolation --no-cache-dir

for dataset in SLAKE vqa-rad path-vqa
do
    python llava/eval/eval_vqa.py \
        --model-path /vol/biomedic3/mv320/projects/VLMs/MEG_x_CL/LLaVA-Med/checkpoints/llava-med-mistral-finetune-SLAKE-from-finetuned \
        --image-folder /vol/biomedic3/mv320/data/${dataset}/test \
        --question-file /vol/biomedic3/mv320/data/${dataset}/test/annotations.json \
        --answers-file ./llava/eval/results/${dataset}/finetune-SLAKE-from-finetuned.json \
        --conv-mode mistral_instruct
done

for dataset in SLAKE vqa-rad path-vqa
do
    python llava/eval/eval_vqa.py \
        --model-path /vol/biomedic3/mv320/projects/VLMs/MEG_x_CL/LLaVA-Med/checkpoints/llava-med-mistral-finetune-path-vqa-from-finetuned \
        --image-folder /vol/biomedic3/mv320/data/${dataset}/test \
        --question-file /vol/biomedic3/mv320/data/${dataset}/test/annotations.json \
        --answers-file ./llava/eval/results/${dataset}/finetune-path-vqa-from-finetuned.json \
        --conv-mode mistral_instruct
done

# for dataset in vqa-rad path-vqa
# do
#     python llava/eval/eval_vqa.py \
#         --model-path /vol/biomedic3/mv320/projects/VLMs/MEG_x_CL/LLaVA-Med/checkpoints/llava-med-mistral-finetune-path-vqa-finetune-vqa-rad-from-original \
#         --image-folder /vol/biomedic3/mv320/data/${dataset}/test \
#         --question-file /vol/biomedic3/mv320/data/${dataset}/test/annotations.json \
#         --answers-file ./llava/eval/results/${dataset}/finetune-path-vqa-finetune-vqa-rad-from-original.json \
#         --conv-mode mistral_instruct
# done

# for dataset in vqa-rad path-vqa SLAKE
# do
#     python llava/eval/eval_vqa.py \
#         --model-path /vol/biomedic3/mv320/projects/VLMs/MEG_x_CL/LLaVA-Med/checkpoints/llava-med-mistral-finetune-merged_vqa_datasets_unbalanced-wandb/checkpoint-800 \
#         --image-folder /vol/biomedic3/mv320/data/${dataset}/test \
#         --question-file /vol/biomedic3/mv320/data/${dataset}/test/annotations.json \
#         --answers-file ./llava/eval/results/${dataset}/full_finetune.json \
#         --conv-mode mistral_instruct
# done

# for dataset in vqa-rad path-vqa SLAKE
# do
#     python llava/eval/eval_vqa.py \
#         --model-path /vol/biomedic3/mv320/projects/VLMs/MEG_x_CL/LLaVA-Med/checkpoints/llava-med-mistral-finetune-vqa-rad-from-finetuned \
#         --image-folder /vol/biomedic3/mv320/data/${dataset}/test \
#         --question-file /vol/biomedic3/mv320/data/${dataset}/test/annotations.json \
#         --answers-file ./llava/eval/results/${dataset}/finetune-vqa-rad-from-finetuned.json \
#         --conv-mode mistral_instruct
# done
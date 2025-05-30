#!/bin/bash
#SBATCH --partition=gpus48
#SBATCH --gres=gpu:1
#SBATCH --output=llava-med_finetune.%N.%j.log
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

# DATASET="merged_vqa_datasets_unbalanced"

# deepspeed --master_port 29510 /vol/biomedic3/mv320/projects/VLMs/MEG_x_CL/LLaVA-Med/llava/train/train_mem.py \
#     --deepspeed ./scripts/zero2.json \
#     --model_name_or_path microsoft/llava-med-v1.5-mistral-7b \
#     --version mistral_instruct \
#     --data_path /vol/biomedic3/mv320/data/${DATASET}/train/annotations.json \
#     --image_folder "" \
#     --tune_mm_mlp_adapter True \
#     --bf16 True \
#     --output_dir ./checkpoints/llava-med-mistral-finetune-${DATASET}-wandb \
#     --num_train_epochs 2 \
#     --per_device_train_batch_size 32 \
#     --per_device_eval_batch_size 32 \
#     --gradient_accumulation_steps 1 \
#     --learning_rate 1e-4 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 2048 \
#     --gradient_checkpointing True \
#     --dataloader_num_workers 4 \
#     --lazy_preprocess True \
#     --report_to wandb \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_total_limit 5 \
#     --save_steps 200 \
#     #--eval_accumulation_steps 1 \
#     #--image_folder  \
#     # eval_path
#     #--load_best_model_at_end True 
#     # --eval_from_train_set False \
#     # --eval_steps 20 \
#     # --metric_for_best_model "eval_loss" \
#     # --prediction_loss_only True \
#     # --greater_is_better False \
#    #--eval_path /vol/biomedic3/mv320/data/${DATASET}/validation/annotations.json \



# PER MODEL
# for model_info in "/vol/biomedic3/mv320/projects/VLMs/MEG_x_CL/LLaVA-Med/checkpoints/llava-med-mistral-finetune-merged_vqa_datasets_unbalanced-wandb/checkpoint-800 from-finetuned" "microsoft/llava-med-v1.5-mistral-7b from-original"
# do
#     # Split the model_info into model path and suffix
#     model=$(echo $model_info | cut -d' ' -f1)
#     suffix=$(echo $model_info | cut -d' ' -f2)

#     echo "Model: $model with suffix: $suffix"
#     DATASET="SLAKE"

#     deepspeed --master_port 29508 /vol/biomedic3/mv320/projects/VLMs/MEG_x_CL/LLaVA-Med/llava/train/train_mem.py \
#         --deepspeed ./scripts/zero2.json \
#         --model_name_or_path $model \
#         --version mistral_instruct \
#         --data_path /vol/biomedic3/mv320/data/${DATASET}/train/annotations.json \
#         --tune_mm_mlp_adapter True \
#         --bf16 True \
#         --output_dir ./checkpoints/llava-med-mistral-finetune-${DATASET}-${suffix} \
#         --num_train_epochs 2 \
#         --per_device_train_batch_size 16 \
#         --per_device_eval_batch_size 16 \
#         --gradient_accumulation_steps 1 \
#         --learning_rate 1e-4 \
#         --weight_decay 0. \
#         --warmup_ratio 0.03 \
#         --lr_scheduler_type "cosine" \
#         --logging_steps 1 \
#         --tf32 True \
#         --model_max_length 2048 \
#         --gradient_checkpointing True \
#         --dataloader_num_workers 4 \
#         --lazy_preprocess True \
#         --report_to wandb \
#         --evaluation_strategy "steps" \
#         --save_strategy "steps" \
#         --save_total_limit 1 \
#         --save_steps 600 \
#         --eval_accumulation_steps 1 \
#         --image_folder /vol/biomedic3/mv320/data/${DATASET}/train \
#         --load_best_model_at_end True \
#         --eval_from_train_set False \
#         --eval_steps 100 \
#         --metric_for_best_model "eval_loss" \
#         --prediction_loss_only True \
#         --greater_is_better False \
#         --load_best_model_at_end True \
#         --eval_data_path /vol/biomedic3/mv320/data/${DATASET}/validation/annotations.json
# done


# # PER DATA
DATASET="vqa-rad"

deepspeed --master_port 29511 /vol/biomedic3/mv320/projects/VLMs/MEG_x_CL/LLaVA-Med/llava/train/train_mem.py \
    --deepspeed .//scripts/zero2.json \
    --model_name_or_path microsoft/llava-med-v1.5-mistral-7b \
    --version mistral_instruct \
    --data_path /vol/biomedic3/mv320/data/medical_vqa/modality_specific_datasets_balanced/CT/train/annotations.json \
    --tune_mm_mlp_adapter True \
    --bf16 True \
    --output_dir ./checkpoints/llava-med-mistral-testl \
    --num_train_epochs 2 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --evaluation_strategy "steps" \
    --save_strategy "steps" \
    --save_total_limit 3 \
    --save_steps 50 \
    --eval_accumulation_steps 1 \
    --image_folder /vol/biomedic3/mv320/data/medical_vqa/modality_specific_datasets_balanced/CT/train \
    --load_best_model_at_end True \
    --eval_from_train_set False \
    --eval_steps 50 \
    --metric_for_best_model "eval_loss" \
    --prediction_loss_only True \
    --greater_is_better False \
    --load_best_model_at_end True \
    --eval_data_path /vol/biomedic3/mv320/data/medical_vqa/modality_specific_datasets_balanced/CT/val/annotations.json \
    --eval_image_folder /vol/biomedic3/mv320/data/medical_vqa/modality_specific_datasets_balanced/CT/val

# deepspeed --master_port 29512 /vol/biomedic3/mv320/projects/VLMs/MEG_x_CL/LLaVA-Med/llava/train/train_mem.py \
#     --deepspeed ./scripts/zero2.json \
#     --model_name_or_path /vol/biomedic3/mv320/projects/VLMs/MEG_x_CL/LLaVA-Med/checkpoints/llava-med-mistral-finetune-path-vqa-from-original \
#     --version mistral_instruct \
#     --data_path /vol/biomedic3/mv320/data/${DATASET}/train/annotations.json \
#     --tune_mm_mlp_adapter True \
#     --bf16 True \
#     --output_dir ./checkpoints/llava-med-mistral-finetune-${DATASET}-finetune-path-vqa-from-original \
#     --num_train_epochs 2 \
#     --per_device_train_batch_size 16 \
#     --per_device_eval_batch_size 16 \
#     --gradient_accumulation_steps 1 \
#     --learning_rate 1e-4 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 2048 \
#     --gradient_checkpointing True \
#     --dataloader_num_workers 4 \
#     --lazy_preprocess True \
#     --report_to wandb \
#     --evaluation_strategy "steps" \
#     --save_strategy "steps" \
#     --save_total_limit 5 \
#     --save_steps 50 \
#     --eval_accumulation_steps 1 \
#     --image_folder /vol/biomedic3/mv320/data/${DATASET}/train \
#     --load_best_model_at_end True \
#     --eval_from_train_set False \
#     --eval_steps 50 \
#     --metric_for_best_model "eval_loss" \
#     --prediction_loss_only True \
#     --greater_is_better False \
#     --load_best_model_at_end True \
#     --eval_data_path /vol/biomedic3/mv320/data/${DATASET}/validation/annotations.json
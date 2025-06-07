import subprocess
import os
import json # For potentially reading/writing metadata if needed
import argparse
import random
import itertools

# --- Configuration ---

DATASETS_SEQUENCE = [
    {
        "name": "CT",
        "train_annotations": "/vol/biomedic3/mv320/data/medical_vqa/modality_specific_datasets_balanced/CT/train/annotations.json",
        "train_images": "/vol/biomedic3/mv320/data/medical_vqa/modality_specific_datasets_balanced/CT/train", # This is the base for image paths in train_annotations
        "test_annotations": "/vol/biomedic3/mv320/data/medical_vqa/modality_specific_datasets_balanced/CT/val/annotations.json",
        "test_images": "/vol/biomedic3/mv320/data/medical_vqa/modality_specific_datasets_balanced/CT/val", # This is the base for image paths in test_annotations
    },
    {
        "name": "Pathology",
        "train_annotations": "/vol/biomedic3/mv320/data/medical_vqa/modality_specific_datasets_balanced/Pathology/train/annotations.json",
        "train_images": "/vol/biomedic3/mv320/data/medical_vqa/modality_specific_datasets_balanced/Pathology/train",
        "test_annotations": "/vol/biomedic3/mv320/data/medical_vqa/modality_specific_datasets_balanced/Pathology/val/annotations.json",
        "test_images": "/vol/biomedic3/mv320/data/medical_vqa/modality_specific_datasets_balanced/Pathology/val",
    },
    {
        "name": "MRI",
        "train_annotations": "/vol/biomedic3/mv320/data/medical_vqa/modality_specific_datasets_balanced/MRI/train/annotations.json",
        "train_images": "/vol/biomedic3/mv320/data/medical_vqa/modality_specific_datasets_balanced/MRI/train",
        "test_annotations": "/vol/biomedic3/mv320/data/medical_vqa/modality_specific_datasets_balanced/MRI/val/annotations.json",
        "test_images": "/vol/biomedic3/mv320/data/medical_vqa/modality_specific_datasets_balanced/MRI/val",
    },
    {
        "name": "X-Ray",
        "train_annotations": "/vol/biomedic3/mv320/data/medical_vqa/modality_specific_datasets_balanced/X-Ray/train/annotations.json",
        "train_images": "/vol/biomedic3/mv320/data/medical_vqa/modality_specific_datasets_balanced/X-Ray/train",
        "test_annotations": "/vol/biomedic3/mv320/data/medical_vqa/modality_specific_datasets_balanced/X-Ray/val/annotations.json", # Using val as test
        "test_images": "/vol/biomedic3/mv320/data/medical_vqa/modality_specific_datasets_balanced/X-Ray/val",
    },
]

# Initial model to start the sequence (can be a base pretrained model or a previous checkpoint)
INITIAL_MODEL_PATH = "microsoft/llava-med-v1.5-mistral-7b"

# Path to your training script and deepspeed config
# Ensure these paths are correct from the directory where you run this Python script, or use absolute paths.
DEEPSPEED_LAUNCH_SCRIPT = "/vol/biomedic3/mv320/projects/VLMs/MEG_x_CL/LLaVA-Med/llava/train/train_mem.py"
DEEPSPEED_CONFIG_PATH = "./scripts/zero2.json" # Relative to LLaVA-Med root, or use absolute path

# Common training arguments 
COMMON_TRAIN_ARGS = [
    "--version", "mistral_instruct",
    "--tune_mm_mlp_adapter", "True",
    "--bf16", "True",
    "--num_train_epochs", "2",      
    "--per_device_train_batch_size", "8",
    "--per_device_eval_batch_size", "8",
    "--gradient_accumulation_steps", "1",
    "--learning_rate", "1e-4",       
    "--weight_decay", "0.",
    "--warmup_ratio", "0.03",
    "--lr_scheduler_type", "cosine",
    "--logging_steps", "10",
    "--tf32", "True",
    "--model_max_length", "2048",
    "--gradient_checkpointing", "True",
    "--dataloader_num_workers", "4",
    "--lazy_preprocess", "True",
    "--report_to", "wandb",          # Set to "none" if you don't want W&B for each stage
    "--save_strategy", "steps",      # Saves at the end of each training stage
    "--save_total_limit", "1",       # Keep only the final checkpoint for this stage
    "--evaluation_strategy", "steps",   # We'll do explicit evaluation runs after training
    "--load_best_model_at_end", "True",
    "--metric_for_best_model", "eval_loss",
    "--greater_is_better", "False",
    "--prediction_loss_only", "True",
    "--eval_from_train_set", "False",
    "--evaluation_strategy", "steps",
    "--eval_steps", "50",
    "--eval_accumulation_steps", "1",
]
#TODO: save best model not at the end of the stage, but at the end of the pipeline

# --- Helper Function ---
def run_deepspeed_command(command_args, stage_type, model_name_info, dataset_name_info):
    """Executes a deepspeed command and handles output."""
    # Use a unique master_port for each Deepspeed launch to avoid issues
    # This simple counter should be fine for sequential runs.
    run_deepspeed_command.master_port_counter += random.randint(0, 50)
    master_port = str(run_deepspeed_command.master_port_counter)

    command = [
        "deepspeed", f"--master_port={master_port}",
        DEEPSPEED_LAUNCH_SCRIPT,
        "--deepspeed", DEEPSPEED_CONFIG_PATH,
    ] + command_args

    print(f"\n{'='*20} Executing {stage_type} {'='*20}")
    print(f"Model: {model_name_info}")
    print(f"Dataset: {dataset_name_info}")
    print(f"Command: {' '.join(command)}\n")

    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()

        print(f"--- STDOUT ({stage_type} on {dataset_name_info}) ---")
        print(stdout)
        print(f"--- STDERR ({stage_type} on {dataset_name_info}) ---")
        print(stderr)

        if process.returncode != 0:
            print(f"ERROR: Deepspeed command failed with return code {process.returncode} for {stage_type} on {dataset_name_info}")
            # Consider raising an exception here if you want the script to stop on failure
            # raise RuntimeError(f"Deepspeed command failed for {stage_type} on {dataset_name_info}")
            return False
        print(f"SUCCESS: Deepspeed command for {stage_type} on {dataset_name_info} completed.")
        return True
    except Exception as e:
        print(f"EXCEPTION during deepspeed command for {stage_type} on {dataset_name_info}: {e}")
        return False

run_deepspeed_command.master_port_counter = 29700 # Initial port number

def parse_args():
    parser = argparse.ArgumentParser(description="Sequential Training Pipeline for LLaVA-Med")
    parser.add_argument("--run_all_binary_sequences", action="store_true",
                        help="Run all possible binary (A->B) sequences from DATASETS_SEQUENCE.")
    parser.add_argument("--cl_method", type=str, default="", choices=["", "ewc", "lwf", "si"],
                        help="Continual Learning method to use.")

    return parser.parse_args()

# --- Main Pipeline Logic ---
def main():

    # Args
    args = parse_args()
    cl_method = args.cl_method
    CONTINUAL_LEARNING_METHOD = cl_method

    # --- Pipeline Configuration ---
    PIPELINE_BASE_OUTPUT_DIR = "/vol/biomedic3/mv320/projects/VLMs/MEG_x_CL/LLaVA-Med/checkpoints/binary_sequential_finetuning_pipeline_" + cl_method
    os.makedirs(PIPELINE_BASE_OUTPUT_DIR, exist_ok=True)
    # W&B Project name
    WANDB_TRAIN_PROJECT = cl_method + "_llava-med-binary-sequential-training" 
    WANDB_EVAL_PROJECT = cl_method + "_llava-med-binary-sequential-evaluation"

    # --- Continual Learning Configuration ---
    CL_METHOD_CONFIGS = {
        "": {},
        "ewc": {
            "lambda": 0.1, # Hyperparameter for EWC
            "data_base_dir": os.path.join(PIPELINE_BASE_OUTPUT_DIR, "data")
        },
    }

    # Ensure data directories for active CL methods exist
    if CONTINUAL_LEARNING_METHOD != "" and CONTINUAL_LEARNING_METHOD in CL_METHOD_CONFIGS:
        print(f"Continual Learning method {CONTINUAL_LEARNING_METHOD} is active. Ensuring data directories exist.")
        cl_data_dir = CL_METHOD_CONFIGS[CONTINUAL_LEARNING_METHOD].get("data_base_dir")
        if cl_data_dir:
            os.makedirs(cl_data_dir, exist_ok=True)

    # --- Dataset Configuration, Sequence Generation ---
    all_dataset_configs_master_list = DATASETS_SEQUENCE
    dataset_name_to_config_map = {d["name"]: d for d in all_dataset_configs_master_list}
    all_dataset_names_master_list = [d["name"] for d in all_dataset_configs_master_list]

    sequences_to_run = [] # This list will hold dictionaries, each defining a sequence to process

    if args.run_all_binary_sequences:
        if len(all_dataset_names_master_list) < 2:
            print("Need at least two datasets to form binary sequences. Exiting.")
            return
        
        binary_permutations = list(itertools.permutations(all_dataset_names_master_list, 2))
        print(f"--run_all_binary_sequences is active. Preparing {len(binary_permutations)} binary sequences.")
        if len(all_dataset_names_master_list) == 4 and len(binary_permutations) != 12:
             print(f"Warning: Expected 12 binary sequences for 4 datasets, but generated {len(binary_permutations)}.")
        elif len(all_dataset_names_master_list) == 4 and len(binary_permutations) == 12:
             print("Correctly generated 12 binary sequences for the 4 available datasets.")

        for p_order in binary_permutations:
            seq_name_slug = f"binary_{p_order[0]}_to_{p_order[1]}"
            current_sequence_dataset_configs = [dataset_name_to_config_map[name] for name in p_order]
            sequences_to_run.append({"name_slug": seq_name_slug, "configs_list": current_sequence_dataset_configs})

    else:
        print("No specific sequence generation flag, --run_all_binary_sequences, provided. Running default sequence based on the order in DATASETS_SEQUENCE ({len(all_dataset_configs_master_list)} stages).")
        raise ValueError("No specific sequence generation flag provided. Exiting.")

    # --- Loop over each sequence defined 
    total_sequences = len(sequences_to_run)
    for seq_idx, seq_info in enumerate(sequences_to_run):
        current_sequence_name_slug = seq_info["name_slug"]
        datasets_for_this_run = seq_info["configs_list"]
        
        current_run_output_base_dir = os.path.join(PIPELINE_BASE_OUTPUT_DIR, current_sequence_name_slug)
        os.makedirs(current_run_output_base_dir, exist_ok=True)

        print(f"\n\n{'='*60}\nProcessing Sequence {seq_idx+1}/{total_sequences}: {current_sequence_name_slug}\nOutput Base Directory: {current_run_output_base_dir}\n{'='*60}")

        current_model_path_for_sequence = INITIAL_MODEL_PATH # Reset for each new sequence group
        trained_on_dataset_names_in_current_sequence = []
        sequence_fully_successful = True

        for i, dataset_config in enumerate(datasets_for_this_run):
            current_dataset_name = dataset_config["name"]
            print(f"\n\n{'#'*40}")
            print(f"Sequence: {current_sequence_name_slug} | Stage {i+1}/{len(datasets_for_this_run)}: Processing Dataset: {current_dataset_name}")
            print(f"Input model for this stage: {os.path.basename(current_model_path_for_sequence)}")
            print(f"#{'#'*39}")

            trained_on_dataset_names_in_current_sequence.append(current_dataset_name)
            # Suffix for this specific stage's output model directory within the current sequence run
            model_name_suffix_within_sequence = "_then_".join(trained_on_dataset_names_in_current_sequence)
            
            stage_output_dir = os.path.join(current_run_output_base_dir, f"trained_on_{model_name_suffix_within_sequence}")
            os.makedirs(stage_output_dir, exist_ok=True)

            # Define WandB project name for this specific run. Using the sequence slug for clarity.
            wandb_project_name_for_run = f"{WANDB_TRAIN_PROJECT}_{current_sequence_name_slug}"

            train_args_for_stage = [
                "--model_name_or_path", current_model_path_for_sequence,
                "--data_path", dataset_config["train_annotations"],
                "--image_folder", dataset_config["train_images"],
                "--output_dir", stage_output_dir,
                "--eval_data_path", dataset_config["test_annotations"],
                "--eval_image_folder", dataset_config["test_images"],
                "--wandb_project", wandb_project_name_for_run,
            ] + COMMON_TRAIN_ARGS
            
            cl_method_specific_args = {}
            if CONTINUAL_LEARNING_METHOD != "" and CONTINUAL_LEARNING_METHOD in CL_METHOD_CONFIGS:
                train_args_for_stage.extend(["--continual_learning_method", CONTINUAL_LEARNING_METHOD])
                train_args_for_stage.extend(["--apply_cl_to_projector", "True"])

                # --- EWC Specific Orchestration ---
                if CONTINUAL_LEARNING_METHOD == "ewc":
                    is_binary_sequence_for_cl = args.run_all_binary_sequences and len(datasets_for_this_run) == 2
                    if is_binary_sequence_for_cl:
                        ewc_config = CL_METHOD_CONFIGS["ewc"]
                        ewc_sequence_data_dir = os.path.join(ewc_config["data_base_dir"], current_sequence_name_slug)
                        os.makedirs(ewc_sequence_data_dir, exist_ok=True)

                        cl_method_specific_args['ewc_lambda'] = ewc_config['lambda']

                        if i == 0: # Task A in A->B for EWC
                            # Task A needs to save its Fisher matrix and optimal projector weights.
                            cl_method_specific_args['save_projector_fisher_after_training'] = True
                            cl_method_specific_args['projector_fisher_output_path'] = os.path.join(ewc_sequence_data_dir, f"fisher_on_{current_dataset_name}.pt")
                            cl_method_specific_args['projector_optimal_weights_output_path'] = os.path.join(ewc_sequence_data_dir, f"optimal_weights_on_{current_dataset_name}.pt")
                            print(f"CL ({CONTINUAL_LEARNING_METHOD.upper()}): Task A ({current_dataset_name}). Configured to save Fisher/weights.")
                            # These paths are stored in the global vars for the next stage (Task B) in this sequence
                            fisher_matrix_path_for_ewc = cl_method_specific_args['projector_fisher_output_path']
                            optimal_weights_path_for_ewc = cl_method_specific_args['projector_optimal_weights_output_path']

                        elif i == 1: # Task B in A->B for EWC
                            # Task B needs to load EWC data from Task A and apply the penalty.
                            if fisher_matrix_path_for_ewc and optimal_weights_path_for_ewc and \
                               os.path.exists(fisher_matrix_path_for_ewc) and os.path.exists(optimal_weights_path_for_ewc):
                                cl_method_specific_args['apply_projector_ewc_penalty'] = True # Specific flag for train_mem.py to enable penalty
                                cl_method_specific_args['projector_fisher_input_path'] = fisher_matrix_path_for_ewc
                                cl_method_specific_args['projector_optimal_weights_input_path'] = optimal_weights_path_for_ewc
                                print(f"CL ({CONTINUAL_LEARNING_METHOD.upper()}): Task B ({current_dataset_name}). Configured to apply penalty using data from Task A.")
                            else:
                                print(f"CL WARNING ({CONTINUAL_LEARNING_METHOD.upper()}): EWC data from Task A not found for sequence '{current_sequence_name_slug}' to apply to task '{current_dataset_name}'. Skipping EWC penalty.")
                    else:
                         print(f"CL ({CONTINUAL_LEARNING_METHOD.upper()}): Not a binary sequence or not enabled for this setup. CL method-specific logic for EWC might not apply as intended.")
                # --- End EWC Specific Orchestration ---

                
                # --- LoRA Specific Orchestration ---
                if CONTINUAL_LEARNING_METHOD == "lora":
                    if i > 0: # If not the first task in a sequence
                        cl_method_specific_args['apply_lora_weights_projector'] = True
                        cl_method_specific_args['lora_alpha'] = CL_METHOD_CONFIGS['lora']['alpha']
                        cl_method_specific_args['lora_temperature'] = CL_METHOD_CONFIGS['lora']['temperature']
                # --- End LoRA Specific Orchestration ---
                
                # --- Add other CL method orchestrations here as elif CONTINUAL_LEARNING_METHOD == "method_name": ---
                # Example for LwF (conceptual):
                # elif CONTINUAL_LEARNING_METHOD == "lwf":
                #     if i > 0: # If not the first task in a sequence
                #         cl_method_specific_args['apply_lwf_penalty'] = True
                #         cl_method_specific_args['lwf_teacher_model_path'] = previous_stage_output_dir # Path to model from Task A
                #         cl_method_specific_args['lwf_alpha'] = CL_METHOD_CONFIGS['lwf']['alpha']
                #         cl_method_specific_args['lwf_temperature'] = CL_METHOD_CONFIGS['lwf']['temperature']

                if cl_method_specific_args:
                    train_args_for_stage.extend(["--cl_method_specific_args_json", json.dumps(cl_method_specific_args)])
            
            print(f"Starting training for sequence '{current_sequence_name_slug}', stage: {model_name_suffix_within_sequence} on dataset {current_dataset_name}")
            print(f"Output directory for this stage: {stage_output_dir}")

            success_stage = run_deepspeed_command(
                train_args_for_stage,
                stage_type="TRAINING",
                model_name_info=f"Seq: {current_sequence_name_slug}, Stage Input: {os.path.basename(current_model_path_for_sequence)}",
                dataset_name_info=current_dataset_name
            )
            if not success_stage:
                print(f"Training FAILED for sequence '{current_sequence_name_slug}', stage {model_name_suffix_within_sequence} on {current_dataset_name}.")
                print(f"Stopping further stages for this sequence: {current_sequence_name_slug}.")
                sequence_fully_successful = False
                break # Stop processing this_sequence, move to the next in sequences_to_run

            current_model_path_for_sequence = stage_output_dir # Update model path for the next stage in *this* sequence

        if sequence_fully_successful:
            print(f"\n\n{'#'*40}")
            print(f"Sequence '{current_sequence_name_slug}' COMPLETED SUCCESSFULLY.")
            print(f"Final model for this sequence stored at: {current_model_path_for_sequence}")
            print(f"#{'#'*39}")
        else:
            print(f"\n\n{'!'*40}")
            print(f"Sequence '{current_sequence_name_slug}' DID NOT COMPLETE successfully.")
            print(f"{'!'*39}")

    print(f"\n\n{'='*60}\nAll Requested Training Pipelines Have Been Processed.\n{'='*60}")

if __name__ == "__main__":
    main()
# LLaVA-Med/llava/eval/evaluate_sequential_checkpoints.py
import argparse
import json
import os
import subprocess
import sys
import re

def parse_trained_on_datasets(checkpoint_folder_name):
    """
    Parses a checkpoint folder name like 'trained_on_CT_then_MRI'
    and returns a list of dataset names: ['CT', 'MRI'].
    """
    if not checkpoint_folder_name.startswith("trained_on_"):
        return []
    
    datasets_part = checkpoint_folder_name.replace("trained_on_", "")
    return datasets_part.split("_then_")

def run_evaluation_command(eval_script_path, model_path, question_file, image_folder, answers_file,
                           common_eval_args, model_name_info, dataset_name_info):
    """Executes the eval_vqa.py script."""
    command = [
        sys.executable,
        eval_script_path,
        "--model-path", model_path,
        "--question-file", question_file,
        "--image-folder", image_folder,
        "--answers-file", answers_file,
    ] + common_eval_args

    print(f"\n{'='*20} Evaluating Model: {model_name_info} on Dataset: {dataset_name_info} {'='*20}")
    print(f"Command: {' '.join(command)}\n")

    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True)
        
        print(f"--- STDOUT (Evaluating {model_name_info} on {dataset_name_info}) ---")
        for stdout_line in iter(process.stdout.readline, ""):
            print(stdout_line, end="")
        process.stdout.close()

        stderr_output = process.stderr.read()
        process.stderr.close()
        
        print(f"--- STDERR (Evaluating {model_name_info} on {dataset_name_info}) ---")
        print(stderr_output)
        
        return_code = process.wait()

        if return_code != 0:
            print(f"ERROR: Evaluation script failed with return code {return_code} for model {model_name_info} on dataset {dataset_name_info}")
            return False
        print(f"SUCCESS: Evaluation for model {model_name_info} on dataset {dataset_name_info} completed. Answers at: {answers_file}")
        return True
    except FileNotFoundError:
        print(f"ERROR: Evaluation script not found at {eval_script_path}. Please check the path.")
        return False
    except Exception as e:
        print(f"EXCEPTION during evaluation for model {model_name_info} on dataset {dataset_name_info}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Evaluate sequential model checkpoints on datasets they were trained on.")
    parser.add_argument("--pipeline-base-output-dir", type=str, default="/vol/biomedic3/mv320/projects/VLMs/MEG_x_CL/LLaVA-Med/checkpoints/sequential_finetuning_pipeline",
                        help="Base directory where sequence folders (e.g., sequence_A) are stored. (PIPELINE_BASE_OUTPUT_DIR from training script)")
    parser.add_argument("--eval-script-path", type=str, default="/vol/biomedic3/mv320/projects/VLMs/MEG_x_CL/LLaVA-Med/llava/eval/eval_vqa.py",
                        help="Path to the eval_vqa.py script.")
    parser.add_argument("--datasets-info-json", type=str, default="/vol/biomedic3/mv320/projects/VLMs/MEG_x_CL/LLaVA-Med/llava/eval/dataset_info.json",
                        help="Path to a JSON file mapping dataset names to their test set paths.")
    parser.add_argument("--evaluation-output-dir", type=str, default="/vol/biomedic3/mv320/projects/VLMs/MEG_x_CL/LLaVA-Med/sequential_test_results",
                        help="Base directory to save all evaluation answer files.")
    parser.add_argument("--sequence-id", type=str, default=None, help="Optional: Sequence ID to process. If None, processes all found sequence folders.")
    
    # Common arguments for eval_vqa.py that will be passed through
    parser.add_argument("--conv-mode", type=str, default="mistral_instruct")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--num-beams", type=int, default=3)
    # Add other arguments from eval_vqa.py that you want to control globally

    args = parser.parse_args()

    # Load dataset info
    try:
        with open(args.datasets_info_json, 'r') as f:
            datasets_info = json.load(f)
    except Exception as e:
        print(f"Error loading datasets_info.json: {e}")
        return

    # Prepare common eval args for eval_vqa.py
    common_eval_args_for_script = [
        "--conv-mode", args.conv_mode,
        "--temperature", str(args.temperature),
        "--num-beams", str(args.num_beams),
    ]

    # Determine sequence directories to process
    sequence_dirs_to_process = []
    if args.sequence_id:
        sequence_dirs_to_process.append(os.path.join(args.pipeline_base_output_dir, f"sequence_{args.sequence_id}"))
    else:
        for item in os.listdir(args.pipeline_base_output_dir):
            path = os.path.join(args.pipeline_base_output_dir, item)
            if os.path.isdir(path) and (item.startswith("sequence_") or item.startswith("custom_order_")):
                sequence_dirs_to_process.append(path)
    print(f"Sequence directories to process: {sequence_dirs_to_process}")

    if not sequence_dirs_to_process:
        print(f"No sequence directories found to process in {args.pipeline_base_output_dir}")
        return

    # Main loop: iterate through sequences, then models, then datasets for evaluation
    for seq_dir_path in sequence_dirs_to_process:
        seq_name = os.path.basename(seq_dir_path)
        print(f"\nProcessing Sequence: {seq_name}...")

        # Find model checkpoint folders (e.g., trained_on_CT, trained_on_CT_then_MRI)
        # These are direct subdirectories of the sequence directory
        model_checkpoint_folders = [
            d for d in os.listdir(seq_dir_path)
            if os.path.isdir(os.path.join(seq_dir_path, d)) and d.startswith("trained_on_")
        ]

        for model_folder_name in model_checkpoint_folders:
            model_checkpoint_path = os.path.join(seq_dir_path, model_folder_name)
            
            trained_on_list = parse_trained_on_datasets(model_folder_name)
            if not trained_on_list:
                print(f"Could not parse dataset names from model folder: {model_folder_name}. Skipping.")
                continue

            print(f"  Model Checkpoint: {model_folder_name} (Trained on: {', '.join(trained_on_list)})")

            # Create output directory for this specific model's evaluations
            current_model_eval_output_dir = os.path.join(args.evaluation_output_dir, seq_name, model_folder_name)
            os.makedirs(current_model_eval_output_dir, exist_ok=True)

            # Evaluate this model on the test set of each dataset it was trained on
            for dataset_to_eval_on_name in trained_on_list:
                if dataset_to_eval_on_name not in datasets_info:
                    print(f"    Warning: Test set info for '{dataset_to_eval_on_name}' not found in datasets_info.json. Skipping evaluation on this dataset.")
                    continue

                dataset_paths = datasets_info[dataset_to_eval_on_name]
                test_annotations_path = dataset_paths.get("test_annotations")
                test_images_path = dataset_paths.get("test_images")

                if not test_annotations_path or not test_images_path:
                    print(f"    Warning: Missing 'test_annotations' or 'test_images' for '{dataset_to_eval_on_name}' in datasets_info.json. Skipping.")
                    continue
                
                answers_file_name = f"eval_on_{dataset_to_eval_on_name}_answers.json"
                answers_file_full_path = os.path.join(current_model_eval_output_dir, answers_file_name)

                run_evaluation_command(
                    eval_script_path=args.eval_script_path,
                    model_path=model_checkpoint_path, # The model folder itself is the path
                    question_file=test_annotations_path,
                    image_folder=test_images_path,
                    answers_file=answers_file_full_path,
                    common_eval_args=common_eval_args_for_script,
                    model_name_info=model_folder_name,
                    dataset_name_info=dataset_to_eval_on_name
                )
    
    print(f"\n\nAll sequential evaluations complete. Results are in: {args.evaluation_output_dir}")

if __name__ == "__main__":
    main()
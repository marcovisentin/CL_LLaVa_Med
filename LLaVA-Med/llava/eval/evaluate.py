import argparse
import json
import sys
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from llava.eval.utils.traditional_metrics import calculate_exactmatch, calculate_f1score, get_accuracy, get_open_ended_metrics
from llava.eval.utils.glossary import *
from llava.eval.utils.mistral import mistal_eval, average_mistral_metrics
import pandas as pd 

from pathlib import Path
import os
import warnings

warnings.simplefilter('ignore')


def parse_option():
    parser = argparse.ArgumentParser('Evaluation for LLaVA Generated Outputs', add_help=False)
    parser.add_argument('--pred', type=str, help='path to prediction file', )
    args, unparsed = parser.parse_known_args()
    return args


def evaluate(gt, pred, answer_type):
    gt = gt.lower()
    pred = pred.lower()

    gt = normalize_word(gt)
    pred = normalize_word(pred)

    if answer_type == "CLOSED":
        # for close-ended question (Yes/No)
        if (gt in pred) or (pred in gt) and len(pred) != 0:
            yes_no_acc = 1
        else:
            yes_no_acc = 0
        return {
            "yes/no accuracy": yes_no_acc
        }

    else:
        exact_score = calculate_exactmatch(pred, gt)
        f1_score, precision, recall = calculate_f1score(pred, gt)
        b_score = sentence_bleu(references=[str(gt).lower().split()],
                                hypothesis=str(pred).lower().split(), weights=[1])
        b_score_1 = sentence_bleu(references=[str(gt).lower().split()],
                                    hypothesis=str(pred).lower().split(), weights=[1])
        b_score_2 = sentence_bleu(references=[str(gt).lower().split()],
                                    hypothesis=str(pred).lower().split(), weights=(1/2, 1/2))
        b_score_3 = sentence_bleu(references=[str(gt).lower().split()],
                                    hypothesis=str(pred).lower().split(), weights=(1/3, 1/3, 1/3))
        return {
            'exact match score': exact_score,
            'f1 score': f1_score,
            'precision': precision,
            'recall': recall,
            'bleu_score': b_score,
            'bleu_score_1': b_score_1,
            'bleu_score_2': b_score_2,
            'bleu_score_3': b_score_3,
        }

def evaluate_open_ended(df):
    # set the params to calculate the average
    num_open_qs=0
    sum_exact_match_score=0
    sum_f1_score=0
    sum_prec=0
    sum_recall=0
    sum_bleu=0
    sum_bleu_1=0
    sum_bleu_2=0
    sum_bleu_3=0

    results = [{
        'avg_exact_match_score': 0,
        'avg_f1_score': 0,
        'avg_precision': 0,
        'avg_recall': 0,
        'avg_bleu_score': 0,
        'avg_bleu_score_1': 0,
        'avg_bleu_score_2': 0,
        'avg_bleu_score_3': 0,
    }]
    for _, row in df.iterrows():
        pred = row['text'].lower()
        gt = row['ground_truth'].lower()
        answer_type = row['answer_type']
        
        if answer_type == 'open':
            metrics = get_open_ended_metrics(gt, pred)
            num_open_qs += 1
            sum_exact_match_score += metrics['exact_match_score']
            sum_f1_score += metrics['f1_score']
            sum_prec += metrics['precision']
            sum_recall += metrics['recall']
            sum_bleu += metrics['bleu_score']
            sum_bleu_1 += metrics['bleu_score_1']
            sum_bleu_2 += metrics['bleu_score_2']
            sum_bleu_3 += metrics['bleu_score_3']
            row['exact_match_score'] = metrics['exact_match_score']
            row['f1_score'] = metrics['f1_score']
            row['precision'] = metrics['precision']
            row['recall'] = metrics['recall']
            row['bleu_score'] = metrics['bleu_score']
            row['bleu_score_1'] = metrics['bleu_score_1']
            row['bleu_score_2'] = metrics['bleu_score_2']
            row['bleu_score_3'] = metrics['bleu_score_3']
            
        results.append(row.to_dict())
    
    results[0]['avg_exact_match_score'] = sum_exact_match_score / max(num_open_qs, 1)
    results[0]['avg_f1_score'] = sum_f1_score / max(num_open_qs, 1)
    results[0]['avg_precision'] = sum_prec / max(num_open_qs, 1)
    results[0]['avg_recall'] = sum_recall / max(num_open_qs, 1)
    results[0]['avg_bleu_score'] = sum_bleu / max(num_open_qs, 1)
    results[0]['avg_bleu_score_1'] = sum_bleu_1 / max(num_open_qs, 1)
    results[0]['avg_bleu_score_2'] = sum_bleu_2 / max(num_open_qs, 1)
    results[0]['avg_bleu_score_3'] =  sum_bleu_3   / max(num_open_qs, 1)
    
    return results


def main(args):
    for sequence_dir in os.listdir(args.base_output_dir):
        for model_dir in os.listdir(args.base_output_dir + "/" + sequence_dir):
            model_path = args.base_output_dir + "/" + sequence_dir + "/" + model_dir
            for file in os.listdir(model_path):
                if file.endswith(".json"):
                    print(f"Processing {file}")
                    sys.stdout.flush()
                    dataset = file.split("_")[2] # Assuming the format is something like "eval_on_X-Ray_..."
                    file_path = model_path + "/" + file
                    
                    # Read the JSON file content
                    try:
                        with open(file_path, 'r') as f_content:
                            raw_data_from_file = json.load(f_content)
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON from {file}. Skipping.")
                        continue
                    except Exception as e:
                        print(f"Error reading {file}: {e}. Skipping.")
                        continue

                    # Check if this file has already been processed and contains our new structure
                    if isinstance(raw_data_from_file, dict) and \
                       "metrics" in raw_data_from_file and \
                       "outputs" in raw_data_from_file:
                        print(f"File {file} already processed and contains metrics. Skipping.")
                        continue 

                    # If not skipped, then raw_data_from_file should be the list of raw outputs
                    if not isinstance(raw_data_from_file, list):
                        print(f"Warning: Content of {file} is not a list as expected for raw answers. Skipping.")
                        continue
                    
                    raw_outputs_list = raw_data_from_file

                    if not raw_outputs_list: # Handle case where the answers file was an empty list
                        print(f"Warning: {file} (raw answers list) is empty. Skipping.")
                        continue
                    
                    # Create DataFrame from the raw list of outputs
                    try:
                        pred_df = pd.DataFrame(raw_outputs_list)
                    except ValueError as e: # Handles cases where list items are not dicts, etc.
                         print(f"Could not create DataFrame from {file}. Content might not be a list of dicts. Error: {e}. Skipping.")
                         continue

                    # Ensure 'answer_type' column exists.
                    if 'answer_type' not in pred_df.columns:
                        print(f"Warning: 'answer_type' column not found in DataFrame from {file}. This file might be malformed or not an answers file. Skipping.")
                        continue
                                    
                    pred_df_closed = pred_df[pred_df['answer_type']=='closed']
                    pred_df_open = pred_df[pred_df['answer_type']=='open']

                    print(f"Closed ended questions: {len(pred_df_closed)}")
                    print(f"Open ended questions: {len(pred_df_open)}")
                    sys.stdout.flush()

                    open_ended_results = evaluate_open_ended(pred_df_open)
                    close_ended_results = get_accuracy(pred_df_closed)

                    # Prepare the metrics dictionary
                    metrics_data = {}
                    if isinstance(close_ended_results, list) and close_ended_results:
                        metrics_data["closed_ended_metrics"] = close_ended_results[0]
                    if isinstance(open_ended_results, list) and open_ended_results:
                        metrics_data["open_ended_metrics"] = open_ended_results[0]
                    
                    # Construct the new data structure with "metrics" as the first key.
                    updated_data = {
                        "metrics": metrics_data,
                        "outputs": raw_outputs_list  # Use the original list of dicts
                    }

                    # Write the updated data back to the original file
                    # with open(file_path, 'w') as f_update:
                    #     json.dump(updated_data, f_update, indent=4)

                    # Mistral eval
                    mistral_scores = mistal_eval(model_output_file=file_path)
                    mistral_metrics_file = model_path + "/" + dataset +"_mistral_metrics.json"
                    if not Path(mistral_metrics_file).parent.is_dir():
                        os.makedirs(Path(mistral_metrics_file).parent)
                    with open(mistral_metrics_file, 'w') as json_file:
                        json.dump(mistral_scores, json_file, indent=4)

    # if "mistral_closed" in cfg.metric_type:
    #     mistral_scores = mistal_eval(model_output_file=model_output_file, closed=True)
    #     mistral_metrics_file = eval_path / "mistral_metrics_closed.json"
    #     if not Path(mistral_metrics_file).parent.is_dir():
    #         os.makedirs(Path(mistral_metrics_file).parent)
    #     with open(mistral_metrics_file, 'w') as json_file:
    #         json.dump(mistral_scores, json_file, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate results")
    parser.add_argument("--base-output-dir", type=str, default="/vol/biomedic3/mv320/projects/VLMs/MEG_x_CL/LLaVA-Med/results_ewc",
                        help="Base directory where results folders are stored.")
    parser.add_argument("--verbose", type=bool, default=False,
                        help="Print more information.")
    args = parser.parse_args()
    main(args)
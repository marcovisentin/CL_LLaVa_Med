"""
In this file I evaluate the model on a VQA dataset. I use the same evaluation strategy as the one used in the paper.
Objective: replicate the results of the paper.
"""

import argparse
import torch
import os
import json
#from tqdm import tqdm
import shortuuid
import time

import sys
sys.path.append("/vol/biomedic3/mv320/projects/VLMs/MEG_x_CL/LLaVA-Med")
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, process_images

from PIL import Image
import math
from transformers import set_seed, logging

logging.set_verbosity_error()


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    set_seed(0)
    # Load model
    print("Loading model...")
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    # Hack to make load_pretrained_model to look correctly - it expects "llava-mistral"
    model_name = "llava_mistral_" + model_name
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name)

    print("Loading questions...")
    questions = json.load(open(args.question_file, "r"))
    for question in questions:
        question_keys = set(question.keys())
        question_keys_filtered = list(question_keys - set(["id", "image", "conversations"]))
    print("Questions loaded") 
    
    answers = []
    ground_truths = []
    answer_type = []
    processed_questions = []
    metadata = {key: [] for key in question_keys_filtered}
    results_list = []

    timed_out_flag = False
    n_questions = len(questions)
    for i,line in enumerate(questions):
        if args.verbose:
            print(f"Question {i} is being precessed...")
        start_time = time.time()
        idx = line["id"]
        image_file = line["image"]
        qs = line["conversations"][0]["value"].replace(DEFAULT_IMAGE_TOKEN, '').strip()
        ground_truth = line["conversations"][1]["value"].strip()

        # Add image tokens to the question      
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        if args.verbose:
            print("Prompt", prompt)
            sys.stdout.flush()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        if args.verbose:
            print("Tokenization done")
            sys.stdout.flush()
            
        image = Image.open(os.path.join(args.image_folder, image_file))
        image_tensor = process_images([image], image_processor, model.config)[0]
        
        if args.verbose:
            print("Image processing complete")
            sys.stdout.flush()
        #print("Image tensor: ", image_tensor.shape)

        # stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        # keywords = [stop_str]
        # stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            generation_start_time = time.time()
            try:    
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_time=5, 
                    max_new_tokens=1024,
                    use_cache=True)
                
                # Synchronize after generation to catch any CUDA errors from the generation step
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    
                if args.verbose:
                    print(f"Q_ID {idx}: model.generate() completed.")
                    sys.stdout.flush()
                outputs_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                if args.verbose:
                    print(f"Q_ID {idx}: Output decoded.")
                    sys.stdout.flush()
            except Exception as e_generate:
                print(f"ERROR during model.generate(), sync, or decoding for Q_ID {idx}: {e_generate}")
                sys.stdout.flush()
            
        generation_end_time = time.time()
        actual_generation_duration = generation_end_time - generation_start_time

        if actual_generation_duration >= (5 * 0.95):
            print(f"WARNING: Generation for question ID {idx} likely hit/exceeded timeout of 5s. "
                f"Actual gen time: {actual_generation_duration:.2f}s.")
            sys.stdout.flush()
        
        if timed_out_flag:
            num_timed_out += 1
            print(f"Skipping save for question ID {idx} due to timeout or generation error.")
            sys.stdout.flush()
            continue # Skip to the next question
        
        processed_questions.append(qs)
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        answers.append(outputs)
        ground_truths.append(ground_truth)
        
        if args.verbose:
            print("Pred produced:", outputs)
        
        if "answer_type" in question:
            answer_type.append(question["answer_type"].lower())
        else:
            if ground_truth.lower() in ["yes", "no"]:
                answer_type.append("closed")
            else:
                answer_type.append("open")
            
        for key in question_keys_filtered:
            metadata[key].append(line[key])
            
        end_time = time.time()
        #print(f"Question id: {idx}")
        print(f"Time taken for question {i}/{n_questions}: {end_time - start_time} seconds")
        sys.stdout.flush()
    
    print("Writing results..")
    
    # Write results to file
    os.makedirs(os.path.dirname(args.answers_file), exist_ok=True)
    for i, (q,a,gt,at) in enumerate(zip(processed_questions, answers, ground_truths, answer_type)):
        results = {"question": q, "answer": a, "ground_truth": gt, "answer_type": at}
        for key in metadata:
            results[key] = metadata[key][i]
        results_list.append(results)
        
    with open(args.answers_file, "w") as f:
        f.write(json.dumps(results_list, indent=4))
        
    # Clear memory 
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/vol/biomedic3/mv320/projects/VLMs/MEG_x_CL/LLaVA-Med/checkpoints/llava-med-finetune-vqa-rad-overwritetower")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="/vol/biomedic3/mv320/data/vqa-rad/test")
    parser.add_argument("--question-file", type=str, default="/vol/biomedic3/mv320/data/vqa-rad/test/annotations.json")
    parser.add_argument("--answers-file", type=str, default="/vol/biomedic3/mv320/projects/VLMs/MEG_x_CL/LLaVA-Med/llava/eval/answer.json")
    parser.add_argument("--conv-mode", type=str, default="mistral_instruct")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num-beams", type=int, default=3)
    parser.add_argument("--verbose", type=bool, default=True)
    args = parser.parse_args()

    eval_model(args)

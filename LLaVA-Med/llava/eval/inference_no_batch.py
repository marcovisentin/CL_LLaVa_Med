import argparse
import torch
import os
import json
import shortuuid
import time
import sys

from transformers import StoppingCriteriaList

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from PIL import Image
from transformers import set_seed


def eval_model_no_batch(args):
    set_seed(0)
    # Load model
    if args.verbose:
        print("Loading model...")
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    # Hack to make load_pretrained_model to look correctly - it expects "llava-mistral"
    model_name = "llava_mistral_" + model_name
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if args.verbose:
        print("Loading questions...")
    questions_data = json.load(open(args.question_file, "r"))
    # Filter out question keys once
    question_keys_filtered = []
    if questions_data:
        first_question = questions_data[0]
        question_keys = set(first_question.keys())
        # Define keys to exclude from metadata more robustly
        excluded_keys = set(["id", "image", "conversations", "answer_type", "question_id", "prompt", "text", "model_id", "ground_truth"])
        question_keys_filtered = list(question_keys - excluded_keys)

    if args.verbose:
        print(f"Questions loaded. Found {len(questions_data)} questions.")
        if question_keys_filtered:
            print(f"Metadata keys to be preserved: {question_keys_filtered}")

    answers_list = [] # Store final dicts for JSON output

    count = 0
    for idx_q, line in enumerate(questions_data):
        count += 1
        question_id = line.get("id", shortuuid.uuid())
        image_file = line["image"]
        qs_original = line["conversations"][0]["value"].replace(DEFAULT_IMAGE_TOKEN, '').strip()
        ground_truth = line["conversations"][1]["value"].strip()

        # Add image tokens to the question
        if model.config.mm_use_im_start_end:
            qs_conv = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs_original
        else:
            qs_conv = DEFAULT_IMAGE_TOKEN + '\n' + qs_original

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs_conv)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        try:
            image = Image.open(os.path.join(args.image_folder, image_file))
            image_tensor = process_images([image], image_processor, model.config)[0]
            image_tensor_input = image_tensor.unsqueeze(0).half().cuda() # Add batch dim for model input
        except Exception as e:
            print(f"Error loading or processing image {image_file} for question ID {question_id}: {e}")
            result = {
                "question_id": question_id,
                "prompt": qs_original,
                "text": f"ERROR: Image processing failed - {e}",
                "answer_type": line.get("answer_type", "open"), # Default to open if not present
                "model_id": model_name,
                "metadata": {key: line.get(key) for key in question_keys_filtered},
                "ground_truth": ground_truth
            }
            answers_list.append(result)
            continue # Skip to next question

        if args.verbose:
            print(f"Processing Q_ID {question_id}: Prompt: {prompt}")
            sys.stdout.flush()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        if args.verbose:
            print(f"Processing Q_ID {question_id}: Input IDs shape: {input_ids.shape}, Image tensor shape: {image_tensor_input.shape}")
            sys.stdout.flush()

        try:
            with torch.inference_mode():
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                # Prepare stopping criteria
                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                stopping_criteria = None # Default to None
                if stop_str is not None:
                    keywords = [stop_str]
                    # Ensure input_ids is defined before this point for KeywordsStoppingCriteria
                    # It seems input_ids is defined just above this block from the prompt construction
                    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

                output_ids = model.generate(
                    input_ids,
                    pad_token_id=tokenizer.pad_token_id,
                    images=image_tensor_input,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=1024,
                    use_cache=True,
                    #early_stopping=False,
                    #pad_token_id=None,
                    #stopping_criteria=stopping_criteria,
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            
            if args.verbose:
                print("input_ids: ", input_ids)
                print("output_ids: ", output_ids)
                sys.stdout.flush()

            output_text = tokenizer.decode(output_ids[0, :], skip_special_tokens=True).strip()
            
            # Log progress less frequently to avoid SLURM stdout issues
            if count % 100 == 0:
                print(f"N_Q {count}/{len(questions_data)}: Pred produced: {output_text}")
                sys.stdout.flush()

            current_answer_type = line.get("answer_type", None)
            if not current_answer_type: # Handle if answer_type is empty string
                if ground_truth.lower() in ["yes", "no"]:
                    current_answer_type = "closed"
                else:
                    current_answer_type = "open"
            else:
                current_answer_type = current_answer_type.lower()

            result = {
                "question_id": question_id,
                "prompt": qs_original,
                "text": output_text,
                "answer_type": current_answer_type,
                "model_id": model_name,
                "metadata": {key: line.get(key) for key in question_keys_filtered},
                "ground_truth": ground_truth
            }
            answers_list.append(result)

        except Exception as e_generate:
            print(f"ERROR during model.generate() or decoding for Q_ID {question_id}: {e_generate}")
            result = {
                "question_id": question_id,
                "prompt": qs_original,
                "text": f"ERROR: Generation failed - {e_generate}",
                "answer_type": line.get("answer_type", None),
                "model_id": model_name,
                "metadata": {key: line.get(key) for key in question_keys_filtered},
                "ground_truth": ground_truth
            }
            answers_list.append(result)
            sys.stdout.flush()

    # Save all answers to a single JSON file
    os.makedirs(os.path.dirname(args.answers_file), exist_ok=True)
    with open(args.answers_file, "w") as ans_file:
        json.dump(answers_list, ans_file, indent=4)
    print(f"\nProcessing complete. Answers saved to {args.answers_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--answers-file", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default="mistral_instruct")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--num-beams", type=int, default=1) # Default to 1 for non-batched, can be overridden
    parser.add_argument("--verbose", default=False, help="Enable verbose output for debugging.")
    # Note: batch_size argument is removed as this script processes one by one

    args = parser.parse_args()
    eval_model_no_batch(args)

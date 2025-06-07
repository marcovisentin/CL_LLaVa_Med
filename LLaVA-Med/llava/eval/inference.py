import argparse
import torch
import os
import json
import shortuuid
import time
import sys

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from PIL import Image
import math
from transformers import set_seed


def eval_model(args):
    set_seed(0)
    # Load model
    if args.verbose:
        print("Loading model...")
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    # Hack to make load_pretrained_model to look correctly - it expects "llava-mistral"
    model_name = "llava_mistral_" + model_name
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name)

    if args.verbose:
        print("Loading questions...")
    questions_data = json.load(open(args.question_file, "r"))
    # Filter out question keys once
    if questions_data:
        first_question = questions_data[0]
        question_keys = set(first_question.keys())
        question_keys_filtered = list(question_keys - set(["id", "image", "conversations", "answer_type"]))
    else:
        question_keys_filtered = []
    if args.verbose:
        print("Questions loaded")

    answers_list = [] # Store final dicts for JSON output

    n_questions = len(questions_data)
    num_batches = math.ceil(n_questions / args.batch_size)

    for i in range(num_batches):
        batch_start_idx = i * args.batch_size
        batch_end_idx = min((i + 1) * args.batch_size, n_questions)
        current_batch_size = batch_end_idx - batch_start_idx
        batch_data = questions_data[batch_start_idx:batch_end_idx]

        if not batch_data: 
            continue

        if args.verbose:
            print(f"Processing batch {i+1}/{num_batches}, questions {batch_start_idx+1}-{batch_end_idx}...")
        
        batch_prompts = []
        batch_image_tensors = []
        batch_ground_truths = []
        batch_ids = []
        batch_metadata_values = {key: [] for key in question_keys_filtered}
        batch_answer_types = []
        batch_original_questions_text = []


        for line_idx, line in enumerate(batch_data):
            idx = line["id"]
            image_file = line["image"]
            qs = line["conversations"][0]["value"].replace(DEFAULT_IMAGE_TOKEN, '').strip()
            ground_truth = line["conversations"][1]["value"].strip()

            # Add image tokens to the question
            if model.config.mm_use_im_start_end:
                qs_conv = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs_conv = DEFAULT_IMAGE_TOKEN + '\n' + qs

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs_conv)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            batch_prompts.append(prompt)
            
            batch_original_questions_text.append(qs) # Store original question text for output

            try:
                image = Image.open(os.path.join(args.image_folder, image_file))
                # process_images expects a list of images
                image_tensor = process_images([image], image_processor, model.config)[0] 
                batch_image_tensors.append(image_tensor)
            except Exception as e:
                print(f"Error loading or processing image {image_file} for question ID {idx}: {e}")
                sys.stdout.flush()
                pass

            # Add other metadata
            batch_ground_truths.append(ground_truth)
            batch_ids.append(idx)
            for key in question_keys_filtered:
                batch_metadata_values[key].append(line.get(key)) # Use .get for safety

            # Define answer type
            if "answer_type" in line:
                batch_answer_types.append(line["answer_type"].lower())
            else:
                if ground_truth.lower() in ["yes", "no"]:
                    batch_answer_types.append("closed")
                else:
                    batch_answer_types.append("open")
        
        # Tokenize prompts in batch
        if args.verbose:
            print("Batch prompts: ", batch_prompts)
            sys.stdout.flush()
        input_ids_list = [tokenizer_image_token(p, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0) for p in batch_prompts]

        if args.verbose:
            print("Len of input_ids_list: ", len(input_ids_list))
            print("Len of first input_ids_list: ", len(input_ids_list[0]))
            print("Type of first input_ids_list: ", type(input_ids_list[0]))  
            print("Len of first input_ids_list: ", input_ids_list[0].shape)  
            sys.stdout.flush()

        # Pad the batch of input_ids
        max_len = max(seq.shape[1] for seq in input_ids_list)
        padded_input_ids_list = []
        attention_mask_list = []
        for seq_tensor in input_ids_list:
            original_token_mask = torch.ones_like(seq_tensor, dtype=torch.long) # Mask for the actual tokens
            padding_size = max_len - seq_tensor.shape[1]
            if padding_size > 0:
                # Pad with tokenizer.pad_token_id. Make sure it's on the left side.
                padding_tensor = torch.full((1, padding_size), tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0, 
                                            dtype=seq_tensor.dtype, device=seq_tensor.device)
                padded_seq = torch.cat([padding_tensor, seq_tensor], dim=1) # Left-pad sequence
                padding_attention_mask = torch.zeros((1, padding_size), dtype=torch.long, device=seq_tensor.device) # Mask for padding tokens
                current_attention_mask = torch.cat([padding_attention_mask, original_token_mask], dim=1) # Left-pad attention mask
                attention_mask_list.append(current_attention_mask)
            else:
                padded_seq = seq_tensor
                attention_mask_list.append(original_token_mask)
            padded_input_ids_list.append(padded_seq)

        input_ids_batch = torch.cat(padded_input_ids_list, dim=0).cuda()
        attention_mask_batch = torch.cat(attention_mask_list, dim=0).cuda()
        
        # Stack image tensors
        if not batch_image_tensors: # if all images failed to load for the batch
            print(f"Skipping batch {i+1} due to no valid images.")
            sys.stdout.flush()
            continue
            
        images_tensor_batch = torch.stack(batch_image_tensors).half().cuda()

        if args.verbose:
            print(f"Batch {i+1}: Tokenization and image processing complete. Input IDs shape: {input_ids_batch.shape}, Images shape: {images_tensor_batch.shape}")
            sys.stdout.flush()

        try:
            with torch.inference_mode():
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                output_ids_batch = model.generate(
                    input_ids_batch,
                    images=images_tensor_batch, # This should be [batch_size, C, H, W]
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=1024, # Consider if max_time was important
                    use_cache=True,
                    attention_mask=attention_mask_batch,
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

            if args.verbose:
                print("Output IDs batch: ", output_ids_batch)
                sys.stdout.flush()
            
            outputs_text_batch = tokenizer.batch_decode(output_ids_batch, skip_special_tokens=True)
            
            #print("Outputs text batch: ", outputs_text_batch)
            #sys.stdout.flush()

            if args.verbose:
                print(f"Batch {i+1}: model.generate() and decoding completed.")
                sys.stdout.flush()

            for k in range(len(outputs_text_batch)):
                current_id = batch_ids[k]
                output_text = outputs_text_batch[k].strip()
                
                if args.verbose:
                    print(f"Q_ID {current_id} (Batch {i+1}, Item {k+1}): Pred produced: {output_text}")

                result = {
                    "question_id": current_id, # Changed from "id" to "question_id" to match common practice
                    "prompt": batch_original_questions_text[k], # Use original question text
                    "text": output_text,
                    "answer_type": batch_answer_types[k],
                    "model_id": model_name, # Using the derived model_name
                    "metadata": {key: batch_metadata_values[key][k] for key in question_keys_filtered},
                    "ground_truth": batch_ground_truths[k]
                }
                answers_list.append(result)

        except Exception as e_generate:
            print(f"ERROR during model.generate(), sync, or decoding for batch {i+1}: {e_generate}")
            # Optionally, add placeholder errors for items in this batch
            for k_err in range(current_batch_size): # current_batch_size might be different if images failed
                 # Need to carefully track which items failed if some images were bad
                current_id_err = batch_ids[k_err] # This assumes batch_ids matches the iteration
                result = {
                    "question_id": current_id_err,
                    "prompt": batch_original_questions_text[k_err],
                    "text": f"ERROR: {e_generate}",
                    "answer_type": batch_answer_types[k_err],
                    "model_id": model_name,
                    "metadata": {key: batch_metadata_values[key][k_err] for key in question_keys_filtered},
                    "ground_truth": batch_ground_truths[k_err]
                }
                answers_list.append(result)
            sys.stdout.flush()
            continue # Move to next batch

    print("Writing results..")
    os.makedirs(os.path.dirname(args.answers_file), exist_ok=True)
    with open(args.answers_file, "w") as f:
        json.dump(answers_list, f, indent=4)
    print(f"Results saved to {args.answers_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/vol/biomedic3/mv320/projects/VLMs/MEG_x_CL/LLaVA-Med/checkpoints/old_binary_sequential_finetuning_pipeline/binary_CT_to_MRI/trained_on_CT")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="/vol/biomedic3/mv320/data/medical_vqa/modality_specific_datasets_balanced/X-Ray/test")
    parser.add_argument("--question-file", type=str, default="/vol/biomedic3/mv320/data/medical_vqa/modality_specific_datasets_balanced/X-Ray/test/annotations.json")
    parser.add_argument("--answers-file", type=str, default="/vol/biomedic3/mv320/projects/VLMs/MEG_x_CL/LLaVA-Med/llava/eval/answer.json")
    parser.add_argument("--conv-mode", type=str, default="mistral_instruct")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num-beams", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for evaluation.")
    parser.add_argument("--verbose", type=bool, default=False) # Python bools from argparse can be tricky, consider action='store_true'
    
    args = parser.parse_args()
    eval_model(args)
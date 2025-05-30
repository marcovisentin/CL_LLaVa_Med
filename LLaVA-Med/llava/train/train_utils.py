import os
import torch
from safetensors import safe_open

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from llava.model import LlavaMistralForCausalLM
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

def verify_mm_weights(model, tensor_path, overwrite_weights=False):
    """
    Verify vision tower weights match the checkpoint exactly and optionally overwrite them.
    
    Args:
        model: The model to verify weights against
        tensor_path: Path to the checkpoint file
        overwrite_weights: If True, overwrite model weights with checkpoint weights
    """
    print("\n==== Checking Vision Weights ====")
    
    # 1. Load checkpoint weights
    print("\nLoading weights from checkpoint:", tensor_path)
    with safe_open(tensor_path, framework="pt", device="cpu") as f:
        # Get vision tower weights
        checkpoint_vision = {k: f.get_tensor(k) for k in f.keys() if 'vision_tower' in k}
        checkpoint_projector = {k: f.get_tensor(k) for k in f.keys() if 'mm_projector' in k}
    
    # 2. Get loaded model weights
    model_state = model.state_dict()
    model_vision = {k: v for k, v in model_state.items() if 'vision_tower' in k}
    model_projector = {k: v for k, v in model_state.items() if 'mm_projector' in k}
    # 3. Compare Vision Tower weights
    print("\nChecking Projector Weights:")
    projector_diffs = []
    missing_in_model = []
    missing_in_checkpoint = []
    
    # Check keys in checkpoint but not in model
    for name in checkpoint_projector:
        if name not in model_projector:
            missing_in_model.append(name)
        else:
            weight_diff = (checkpoint_projector[name] - model_projector[name]).abs().max().item()
            if weight_diff > 1e-3:
                projector_diffs.append((name, weight_diff))

    if projector_diffs:
        print(f"Found {len(projector_diffs)} differences in Projector weights.")
    else:
        print("All matching Projector weights are identical.")
    
    # 3. Compare Vision Tower weights
    print("\nChecking Vision Tower Weights:")
    vision_diffs = []
    missing_in_model = []
    missing_in_checkpoint = []
    
    # Check keys in checkpoint but not in model
    for name in checkpoint_vision:
        if name not in model_vision:
            missing_in_model.append(name)
        else:
            weight_diff = (checkpoint_vision[name] - model_vision[name]).abs().max().item()
            if weight_diff > 1e-5:
                vision_diffs.append((name, weight_diff))
    
    # Check keys in model but not in checkpoint
    for name in model_vision:
        if name not in checkpoint_vision:
            missing_in_checkpoint.append(name)
    
    if missing_in_model:
        print(f"Found {len(missing_in_model)} vision tower weights in checkpoint but not in model.")
    
    if missing_in_checkpoint:
        print(f"Found {len(missing_in_checkpoint)} vision tower weights in model but not in checkpoint.")
    
    if vision_diffs:
        print(f"Found {len(vision_diffs)} differences in Vision Tower weights.")
    else:
        print("All matching Vision Tower weights are identical.")
    
    # 4. Overwrite vision tower weights if requested
    if overwrite_weights:
        print("\nOverwriting vision tower weights with checkpoint weights...")
        # Create a new state dict with updated weights
        new_state_dict = model.state_dict()
        updated_count = 0
        
        for name in checkpoint_vision:
            if name in model_vision:
                new_state_dict[name] = checkpoint_vision[name]
                updated_count += 1
        
        # Load the updated state dict back into the model
        model.load_state_dict(new_state_dict, strict=False)
        print(f"Updated {updated_count} vision tower weights.")
        
        # Check if we need to handle missing weights
        if missing_in_model or missing_in_checkpoint:
            print("\nNote: Some weights exist in one but not the other.")
            print(f"- {len(missing_in_model)} weights in checkpoint but not in model")
            print(f"- {len(missing_in_checkpoint)} weights in model but not in checkpoint")
    
    print("\n==== End Weights Check ====\n")
    

def load_pretrained_model_for_training(model_path, model_base, model_name, training_args, attn_implementation, load_8bit=False, load_4bit=False, device_map="auto", device="cuda"):

    kwargs = {}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16
    

    tokenizer = AutoTokenizer.from_pretrained(model_path) # just use default from config
    model = LlavaMistralForCausalLM.from_pretrained(
        model_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=attn_implementation, 
        **kwargs
    )

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    
    torch_dtype = torch.bfloat16 if training_args.bf16 else torch.float16
    image_processor = vision_tower.image_processor

    return tokenizer, model, image_processor
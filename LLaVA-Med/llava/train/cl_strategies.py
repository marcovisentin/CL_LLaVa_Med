# --- In train_mem.py ---
import os
import copy
from peft import LoraConfig, get_peft_model, PeftModel

import json
import torch
import gc # For garbage collection
from torch import nn # nn might be used by other parts of the class or file
from typing import Dict, Any, Union, Optional # For type hints if used elsewhere

# (In your argument parsing section)
# training_args.add_argument("--continual_learning_method", type=str, default="none")
# training_args.add_argument("--apply_cl_to_projector", type=bool, default=False) # Or action="store_true"
# training_args.add_argument("--cl_method_specific_args_json", type=str, default="{}")

# --- CL Strategy Base Class and Implementations ---
class ContinualLearningStrategy:
    def __init__(self, model, training_args, cl_specific_args):
        self.model = model
        self.training_args = training_args # Main training script args
        self.cl_specific_args = cl_specific_args # Parsed from JSON
        self.projector = self._get_projector()

    def _get_projector(self):
        if not self.training_args.apply_cl_to_projector:
            return None
        try:
            projector = self.model.get_model().mm_projector 
            return projector
        except AttributeError:
            print("CL_STRATEGY: Warning - Multimodal projector not found or not accessible.")
            return None

    def on_train_begin(self):
        """Called before the main training loop."""
        pass

    def compute_cl_penalty(self, original_loss):
        """
        Computes the CL penalty and adds it to the original loss.
        Returns the modified loss.
        """
        return original_loss # Default: no penalty

    def on_train_end(self, train_dataloader):
        """Called after the main training loop, e.g., for Fisher calculation."""
        pass

class NoStrategy(ContinualLearningStrategy):
    pass # All methods do nothing or return original values

class EWCStrategy(ContinualLearningStrategy):
    def __init__(self, model, training_args, cl_specific_args):
        super().__init__(model, training_args, cl_specific_args)
        self.loaded_fisher_diag = None
        self.loaded_optimal_projector_weights = None

    def on_train_begin(self):
        if not self.projector: 
            print("EWCStrategy.on_train_begin: No projector found, skipping EWC data loading.")
            return

        if self.cl_specific_args.get('apply_projector_ewc_penalty', False):
            fisher_path = self.cl_specific_args.get('projector_fisher_input_path')
            weights_path = self.cl_specific_args.get('projector_optimal_weights_input_path')
            
            if fisher_path and weights_path:
                try:
                    print(f"EWCStrategy: Attempting to load Fisher from {fisher_path} and weights from {weights_path}.")
                    # Load to CPU first
                    loaded_fisher_diag_cpu = torch.load(fisher_path, map_location='cpu')
                    loaded_optimal_weights_cpu = torch.load(weights_path, map_location='cpu')
                    
                    # Determine target device from the projector
                    target_device = next(self.projector.parameters()).device
                    print(f"EWCStrategy: Moving loaded EWC tensors to device: {target_device}")

                    self.loaded_fisher_diag = {
                        name: tensor.to(target_device) for name, tensor in loaded_fisher_diag_cpu.items()
                    }
                    self.loaded_optimal_projector_weights = {
                        name: tensor.to(target_device) for name, tensor in loaded_optimal_weights_cpu.items()
                    }
                    print(f"EWCStrategy: Successfully loaded and moved Fisher and optimal weights to {target_device}.")
                except Exception as e:
                    print(f"EWCStrategy WARNING: Could not load or move EWC data. Error: {e}")
                    # Potentially disable EWC penalty if loading fails
                    self.cl_specific_args['apply_projector_ewc_penalty'] = False 
            else:
                print("EWCStrategy WARNING: Input paths for Fisher/weights not provided for penalty application. EWC penalty will not be applied.")
                self.cl_specific_args['apply_projector_ewc_penalty'] = False
        else:
            print("EWCStrategy: 'apply_projector_ewc_penalty' is False. Skipping EWC data loading for penalty.")


    def compute_cl_penalty(self, original_loss):
        if not self.projector or \
           not self.cl_specific_args.get('apply_projector_ewc_penalty', False) or \
           self.loaded_fisher_diag is None or \
           self.loaded_optimal_projector_weights is None:
            # logger.debug("EWCStrategy.compute_cl_penalty: Conditions not met for EWC penalty, returning original loss.")
            return original_loss

        ewc_penalty = 0.0
        ewc_lambda = self.cl_specific_args.get('ewc_lambda', 0.0)
        if ewc_lambda == 0.0:
            return original_loss # No penalty if lambda is zero

        # logger.debug(f"EWCStrategy.compute_cl_penalty: Calculating EWC penalty with lambda={ewc_lambda}")
        for name, param in self.projector.named_parameters():
            if param.requires_grad and name in self.loaded_fisher_diag:
                # Tensors are now assumed to be on the same device as param
                fisher_importance = self.loaded_fisher_diag[name]
                optimal_param_val = self.loaded_optimal_projector_weights[name]
                
                # Ensure param is on the same device as fisher_importance and optimal_param_val
                # This should ideally be true if projector parameters are on target_device
                if param.device != fisher_importance.device:
                    # This case should be rare if setup is correct but good to be defensive
                    param_on_correct_device = param.to(fisher_importance.device)
                    ewc_penalty += (fisher_importance * (param_on_correct_device - optimal_param_val).pow(2)).sum()
                else:
                    ewc_penalty += (fisher_importance * (param - optimal_param_val).pow(2)).sum()
                    
        print(f"EWCStrategy.compute_cl_penalty: Calculated ewc_penalty_sum = {ewc_penalty.item()}") # Debugging
        print(f"EWCStrategy.compute_cl_penalty: Calculated ewc_lambda = {ewc_lambda}") # Debugging
        print(f"EWCStrategy.compute_cl_penalty: Calculated ewc_lambda / 2.0 = {ewc_lambda / 2.0}") # Debugging
        # logger.debug(f"EWCStrategy.compute_cl_penalty: Calculated ewc_penalty_sum = {ewc_penalty.item()}")
        return original_loss + (ewc_lambda / 2.0) * ewc_penalty

    def on_train_end(self, train_dataloader):
        if self.projector and self.cl_specific_args.get('save_projector_fisher_after_training', False):
            projector_optimal_weights_output_path = self.cl_specific_args.get('projector_optimal_weights_output_path')
            if projector_optimal_weights_output_path:
                optimal_weights_cpu = {name: param.detach().clone().cpu() for name, param in self.projector.named_parameters()}
                torch.save(optimal_weights_cpu, projector_optimal_weights_output_path)
                print(f"EWCStrategy: Saved optimal projector weights to {projector_optimal_weights_output_path}")
            
            print("EWCStrategy: Starting Fisher Information Matrix calculation for the projector...")
            
            projector_was_training = self.projector.training
            self.projector.train() # Set projector to train mode for gradient calculation
            for param in self.projector.parameters():
                param.requires_grad = True

            model_gc_was_enabled = False
            # Check if gradient checkpointing is enabled on the main model
            # Prefer is_gradient_checkpointing attribute if available (more direct)
            if hasattr(self.model, 'is_gradient_checkpointing') and self.model.is_gradient_checkpointing:
                 model_gc_was_enabled = True
            # Fallback to checking model.config and training_args
            elif hasattr(self.model, 'config') and hasattr(self.model.config, 'gradient_checkpointing') and self.model.config.gradient_checkpointing:
                 model_gc_was_enabled = True
            elif hasattr(self.training_args, 'gradient_checkpointing') and self.training_args.gradient_checkpointing:
                 model_gc_was_enabled = True

            if model_gc_was_enabled and hasattr(self.model, 'gradient_checkpointing_disable'):
                print("EWCStrategy: Temporarily disabling gradient checkpointing on main model for Fisher calculation.")
                self.model.gradient_checkpointing_disable()
            elif model_gc_was_enabled:
                print("EWCStrategy: Warning - Gradient checkpointing is enabled, but model does not have 'gradient_checkpointing_disable' method. Fisher calculation might fail.")
            
            # Store current model training state and set to eval mode for Fisher calculation
            model_was_training = self.model.training
            self.model.eval()
            print(f"EWCStrategy: Set main model to eval() mode for Fisher calculation. Was training: {model_was_training}")

            fisher_diag = {name: torch.zeros_like(param.data) for name, param in self.projector.named_parameters() if param.requires_grad}
            num_samples = 0
            
            fisher_batch_size = self.cl_specific_args.get('ewc_fisher_batch_size', 1)
            print(f"EWCStrategy: Preparing DataLoader for Fisher calculation with batch_size: {fisher_batch_size}")
            
            # Minimal way to create a new DataLoader for Fisher calculation
            # Assumes train_dataloader.dataset and train_dataloader.collate_fn are available and compatible
            _fisher_dl = torch.utils.data.DataLoader(
                train_dataloader.dataset, 
                batch_size=fisher_batch_size,
                collate_fn=train_dataloader.collate_fn,
                sampler=torch.utils.data.SequentialSampler(train_dataloader.dataset),
                num_workers=getattr(train_dataloader, 'num_workers', 0), # Keep num_workers if possible
                pin_memory=getattr(train_dataloader, 'pin_memory', False) # Keep pin_memory if possible
            )

            try:
                from tqdm import tqdm
                pbar = tqdm(_fisher_dl, desc=f"Calculating Fisher Info (batch_size={fisher_batch_size})")
            except ImportError:
                print("tqdm not found, proceeding without progress bar for Fisher calculation.")
                pbar = _fisher_dl
            
            # Main Fisher calculation loop
            try:
                for batch_idx, inputs in enumerate(pbar):
                    torch.cuda.empty_cache() # Aggressive cache clearing inside the loop
                    gc.collect()             # Aggressive garbage collection inside the loop

                    # Ensure all input tensors are on the same device as the model
                    model_device = next(self.model.parameters()).device
                    inputs_on_device = {}
                    for k, v in inputs.items():
                        if isinstance(v, torch.Tensor):
                            inputs_on_device[k] = v.to(model_device)
                        elif isinstance(v, list) and all(isinstance(i, torch.Tensor) for i in v):
                            inputs_on_device[k] = [i.to(model_device) for i in v]
                        else:
                            inputs_on_device[k] = v
                    
                    outputs = self.model(**inputs_on_device)
                    loss = outputs.loss
                    if loss is None:
                        print(f"EWCStrategy: Warning - Loss is None during Fisher calculation for batch {batch_idx}. Skipping batch.")
                        continue

                    projector_params_to_calc_grads_for = []
                    param_names_for_grads = []
                    for name, param in self.projector.named_parameters():
                        if param.requires_grad:
                            projector_params_to_calc_grads_for.append(param)
                            param_names_for_grads.append(name)

                    if not projector_params_to_calc_grads_for:
                        continue
                    
                    try:
                        grads = torch.autograd.grad(loss, projector_params_to_calc_grads_for, allow_unused=True, retain_graph=False)
                    except RuntimeError as e:
                        print(f"EWCStrategy: RuntimeError during torch.autograd.grad: {e}. Skipping batch {batch_idx}.")
                        continue

                    for param_name, grad_val in zip(param_names_for_grads, grads):
                        if grad_val is not None:
                            if param_name in fisher_diag:
                                fisher_diag[param_name] += (grad_val.data ** 2)
                            else:
                                print(f"EWCStrategy: Warning - Param '{param_name}' from projector has grad but not found in fisher_diag.")
                    
                    current_batch_size = 0
                    if 'input_ids' in inputs and hasattr(inputs['input_ids'], 'size'):
                        current_batch_size = inputs['input_ids'].size(0)
                    elif inputs and isinstance(inputs, dict) and inputs.values():
                        first_tensor_val = next((v for v in inputs.values() if isinstance(v, torch.Tensor) and hasattr(v, 'size')), None)
                        current_batch_size = first_tensor_val.size(0) if first_tensor_val is not None else 0
                    num_samples += current_batch_size if current_batch_size > 0 else 1

            finally:
                # Re-enable gradient checkpointing on the main model if it was disabled
                if model_gc_was_enabled and hasattr(self.model, 'gradient_checkpointing_enable'):
                    print("EWCStrategy: Re-enabling gradient checkpointing on main model.")
                    self.model.gradient_checkpointing_enable()
                elif model_gc_was_enabled:
                    print("EWCStrategy: Warning - Gradient checkpointing was enabled, but model does not have 'gradient_checkpointing_enable' method. State may be inconsistent.")
                
                # Restore main model's original training mode
                self.model.train(model_was_training)
                print(f"EWCStrategy: Restored main model training mode to: {model_was_training}.")
                
                # Restore projector's original training state
                if projector_was_training:
                    self.projector.train()
                else:
                    self.projector.eval()
                
                torch.cuda.empty_cache()
                gc.collect()

            if num_samples > 0:
                print(f"EWCStrategy: Normalizing Fisher matrix by {num_samples} samples.")
                for name in fisher_diag:
                    fisher_diag[name] /= num_samples
            else:
                print("EWCStrategy: No samples processed. Fisher matrix might be zeros or unnormalized.")

            fisher_diag_cpu = {name: tensor.detach().clone().cpu() for name, tensor in fisher_diag.items()}
            fisher_path = self.cl_specific_args.get('projector_fisher_output_path')
            if fisher_path:
                try:
                    torch.save(fisher_diag_cpu, fisher_path)
                    print(f"EWCStrategy: Saved projector Fisher matrix (diagonal) to {fisher_path}")
                except Exception as e:
                    print(f"EWCStrategy: Failed to save Fisher matrix to {fisher_path}. Error: {e}")
            
            # Set projector to eval mode after Fisher calculation is done and saved, as its parameters are now considered fixed for this EWC stage.
            self.projector.eval()


class LoRAStrategy(ContinualLearningStrategy):
    def __init__(self, model, training_args, cl_specific_args):
        super().__init__(model, training_args, cl_specific_args)
        
        self.current_task_id = self.cl_specific_args.get('current_task_id', 0)
        print(f"LoRAStrategy (Binary Task Mode): Initializing for task {self.current_task_id}.")

        # LoRA specific configurations
        self.lora_r = self.cl_specific_args.get('lora_r', 8)
        self.lora_alpha = self.cl_specific_args.get('lora_alpha', 16)
        self.lora_dropout = self.cl_specific_args.get('lora_dropout', 0.05)
        self.lora_target_modules = self.cl_specific_args.get('lora_target_modules')
        
        if self.lora_target_modules is None:
            raise ValueError("LoRAStrategy: 'lora_target_modules' must be specified in cl_specific_args (e.g., [\"fc1\", \"fc2\"] for an MLP projector).")

        # Path for saving/loading the full projector weights from Task 0
        self.base_projector_output_path = self.cl_specific_args.get('base_projector_output_path', './base_projector_weights')
        os.makedirs(self.base_projector_output_path, exist_ok=True)
        self.task0_projector_weights_file = os.path.join(self.base_projector_output_path, "task_0_projector.pth")

        # Path for saving LoRA adapters trained during Task 1 (optional, but good for inspection)
        self.task1_lora_adapter_output_dir = self.cl_specific_args.get('task1_lora_adapter_output_dir', './lora_projector_adapters_task1')
        os.makedirs(self.task1_lora_adapter_output_dir, exist_ok=True)

        if self.projector is None:
            print("LoRAStrategy: Warning - Projector is None during init. LoRA strategy may not function correctly.")

    def on_train_begin(self):
        if self.projector is None:
            print("LoRAStrategy.on_train_begin: Projector is None. Skipping LoRA setup.")
            return

        print(f"LoRAStrategy.on_train_begin: Configuring projector for task {self.current_task_id}.")

        if self.current_task_id == 0:
            print("LoRAStrategy.on_train_begin: Task 0. Projector will be trained normally.")
            # No LoRA application for task 0, projector trains as is.
        
        elif self.current_task_id == 1:
            print(f"LoRAStrategy.on_train_begin: Task 1. Loading base projector from Task 0 and applying LoRA.")
            if os.path.exists(self.task0_projector_weights_file):
                try:
                    self.projector.load_state_dict(torch.load(self.task0_projector_weights_file, map_location='cpu'))
                    print(f"LoRAStrategy.on_train_begin: Successfully loaded projector weights from {self.task0_projector_weights_file}.")
                except Exception as e:
                    print(f"LoRAStrategy.on_train_begin: Failed to load projector weights from {self.task0_projector_weights_file}. Error: {e}. Proceeding with current projector state.")
                    # Potentially raise error or handle as critical failure
            else:
                print(f"LoRAStrategy.on_train_begin: CRITICAL - No projector weights found at {self.task0_projector_weights_file} from Task 0. LoRA for Task 1 cannot proceed as intended.")
                # This is likely an error in the sequence, should not happen if Task 0 completed.
                # Consider raising an error here.
                return # Or raise error

            # Apply LoRA to the loaded projector for Task 1 training
            lora_config = LoraConfig(
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                target_modules=self.lora_target_modules,
                bias="none", # Typically 'none' or 'lora_only' for LoRA
            )
            try:
                self.projector = get_peft_model(self.projector, lora_config)
                print(f"LoRAStrategy.on_train_begin: Applied LoRA to projector for Task 1 training.")
                self.projector.print_trainable_parameters() # Should show only LoRA params as trainable
            except Exception as e:
                print(f"LoRAStrategy.on_train_begin: Failed to apply LoRA to projector for Task 1. Error: {e}. Projector will be trained as is.")
        else:
            print(f"LoRAStrategy.on_train_begin: Task ID {self.current_task_id} not supported by this binary LoRA strategy. Projector unchanged.")

    def on_train_end(self, train_dataloader):
        if self.projector is None:
            print("LoRAStrategy.on_train_end: Projector is None. Skipping actions.")
            return

        if self.current_task_id == 0:
            print(f"LoRAStrategy.on_train_end: Task 0 finished. Saving full projector weights to {self.task0_projector_weights_file}.")
            try:
                torch.save(self.projector.state_dict(), self.task0_projector_weights_file)
                print(f"LoRAStrategy.on_train_end: Full projector weights saved successfully for Task 0.")
            except Exception as e:
                print(f"LoRAStrategy.on_train_end: Failed to save full projector weights for Task 0. Error: {e}")
        
elif self.current_task_id == 1:
            if isinstance(self.projector, PeftModel):
                task1_adapter_path = os.path.join(self.task1_lora_adapter_output_dir, f"task_{self.current_task_id}_adapter")
                print(f"LoRAStrategy.on_train_end: Task 1 finished. Saving LoRA adapter to {task1_adapter_path}.")
                try:
                    self.projector.save_pretrained(task1_adapter_path)
                    print(f"LoRAStrategy.on_train_end: LoRA adapter saved successfully for Task 1.")
                except Exception as e:
                    print(f"LoRAStrategy.on_train_end: Failed to save LoRA adapter for Task 1. Error: {e}")
            else:
                print(f"LoRAStrategy.on_train_end: Task 1 finished, but projector is not a PeftModel. No LoRA adapter to save.")
        else:
            print(f"LoRAStrategy.on_train_end: No specific action for task ID {self.current_task_id}.")

    def compute_cl_penalty(self, original_loss):
        # This LoRA strategy does not use a penalty term.
        return original_loss
            print("EWCStrategy: Set projector to eval mode after Fisher operations.")

        else:
            if not self.projector:
                print("EWCStrategy.on_train_end: No projector found, skipping Fisher calculation and saving.")
            else: # save_projector_fisher_after_training is False
                print("EWCStrategy.on_train_end: 'save_projector_fisher_after_training' is False. Skipping Fisher calculation and saving.")

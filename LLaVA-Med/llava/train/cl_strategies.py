# --- In train_mem.py ---
import json

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
        # Logic to get self.model.get_model().mm_projector or similar
        # Handle cases where projector might not exist or CL is not applied to it
        if not self.training_args.apply_cl_to_projector:
            return None
        try:
            # Adjust this access path based on your actual model structure
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
        if not self.projector: return

        if self.cl_specific_args.get('apply_projector_ewc_penalty', False):
            fisher_path = self.cl_specific_args.get('projector_fisher_input_path')
            weights_path = self.cl_specific_args.get('projector_optimal_weights_input_path')
            if fisher_path and weights_path:
                try:
                    self.loaded_fisher_diag = torch.load(fisher_path, map_location='cpu')
                    self.loaded_optimal_projector_weights = torch.load(weights_path, map_location='cpu')
                    print(f"EWCStrategy: Successfully loaded Fisher and optimal weights for projector.")
                except Exception as e:
                    print(f"EWCStrategy WARNING: Could not load EWC data. Error: {e}")
                    # Potentially disable EWC penalty if loading fails
                    self.cl_specific_args['apply_projector_ewc_penalty'] = False 
            else:
                print("EWCStrategy WARNING: Input paths for Fisher/weights not provided for penalty application.")
                self.cl_specific_args['apply_projector_ewc_penalty'] = False


    def compute_cl_penalty(self, original_loss):
        if not self.projector or \
           not self.cl_specific_args.get('apply_projector_ewc_penalty', False) or \
           self.loaded_fisher_diag is None or \
           self.loaded_optimal_projector_weights is None:
            return original_loss

        ewc_penalty = 0.0
        ewc_lambda = self.cl_specific_args.get('ewc_lambda', 0.0)

        for name, param in self.projector.named_parameters():
            if param.requires_grad and name in self.loaded_fisher_diag:
                fisher_importance = self.loaded_fisher_diag[name].to(param.device)
                optimal_param_val = self.loaded_optimal_projector_weights[name].to(param.device)
                ewc_penalty += (fisher_importance * (param - optimal_param_val).pow(2)).sum()
        
        return original_loss + (ewc_lambda / 2.0) * ewc_penalty

    def on_train_end(self, train_dataloader):
        if not self.projector or not self.cl_specific_args.get('save_projector_fisher_after_training', False):
            return

        optimal_weights_path = self.cl_specific_args.get('projector_optimal_weights_output_path')
        fisher_path = self.cl_specific_args.get('projector_fisher_output_path')

        if not optimal_weights_path or not fisher_path:
            print("EWCStrategy WARNING: Output paths for Fisher/weights not provided for saving.")
            return
            
        self.model.eval()
        # 1. Save optimal projector weights
        optimal_projector_weights = {
            name: param.clone().detach().cpu()
            for name, param in self.projector.named_parameters() if param.requires_grad
        }
        torch.save(optimal_projector_weights, optimal_weights_path)
        print(f"EWCStrategy: Saved optimal projector weights to {optimal_weights_path}")

        # 2. Calculate and save Fisher Information Matrix (diagonal)
        # (Similar logic as before, using train_dataloader)
        # ... (ensure fisher_diag is calculated) ...
        # torch.save(fisher_diag, fisher_path)
        # print(f"EWCStrategy: Saved projector Fisher matrix to {fisher_path}")
        print("EWCStrategy: Fisher calculation logic needs to be fully implemented here using train_dataloader.")



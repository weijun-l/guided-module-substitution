"""
Purifier implementation for GMS (Guided Module Search) decoder models
"""

import copy
import time
import json
import numpy as np
import torch
import sys
from datetime import datetime
from pathlib import Path
from typing import Set, Tuple, Dict, Any, List

from .config import MODULE_MAPPING
from .model_utils import get_model_type_and_layers
from .strategy import find_best_strategy

def evaluate_decoder_model(model, tokenizer, test_dataset, batch_size=8):
    """Evaluate decoder model performance."""
    from datasets import Dataset
    
    model.eval()
    device = next(model.parameters()).device
    correct = 0
    total = len(test_dataset)
    
    # Handle padding token - crucial for decoder models
    if getattr(tokenizer, "pad_token_id") is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token_id = tokenizer.unk_token_id
            tokenizer.pad_token = tokenizer.unk_token
    
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # Convert to Dataset if it's a list
    if not isinstance(test_dataset, Dataset):
        if isinstance(test_dataset, list) and len(test_dataset) > 0:
            test_dataset = Dataset.from_list(test_dataset)
        else:
            raise ValueError("test_dataset must be a Dataset or non-empty list of dicts")
    
    # Clear CUDA cache before evaluation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    with torch.no_grad():
        # Process in batches to save memory for large datasets
        for i in range(0, total, batch_size):
            batch_end = min(i + batch_size, total)
            batch_sentences = [test_dataset[j]['sentence'] for j in range(i, batch_end)]
            batch_labels = [test_dataset[j]['label'] for j in range(i, batch_end)]
            
            # Tokenize batch
            inputs = tokenizer(
                batch_sentences, 
                return_tensors="pt", 
                max_length=128, 
                truncation=True, 
                padding=True
            ).to(device)
            
            # Forward pass
            outputs = model(**inputs)
            predicted_labels = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            
            # Count correct predictions
            correct += sum(pred == true for pred, true in zip(predicted_labels, batch_labels))
            
            # Clear cache after each batch to save memory
            if torch.cuda.is_available() and i % (batch_size * 4) == 0:  # Every 4 batches
                torch.cuda.empty_cache()
    
    return correct / total

class LLaMAPurifier:
    """
    Purifier for LLaMA and other decoder models using LoRA module replacement.
    """
    
    def __init__(self,
                 base_model_path: str,
                 source_lora: str,
                 target_lora: str,
                 tokenizer: Any,
                 source_model: Any,
                 target_model: Any,
                 pseudo_clean_test: List[Dict[str, Any]],
                 pseudo_poisoned_test: List[Dict[str, Any]],
                 true_clean_test: List[Dict[str, Any]],
                 true_poisoned_test: List[Dict[str, Any]],
                 alpha: float = 0.2):
        """
        Initialize the purifier.
        
        Args:
            base_model_path: Path to the base model
            source_lora: Path to source LoRA weights
            target_lora: Path to target LoRA weights  
            tokenizer: Tokenizer instance
            source_model: Source model with LoRA
            target_model: Target model with LoRA
            pseudo_clean_test: Proxy clean set for search guidance
            pseudo_poisoned_test: Proxy suspect set for search guidance
            true_clean_test: True clean test set for final evaluation
            true_poisoned_test: True poisoned test set for final evaluation
            alpha: Balance parameter for objective function
        """
        
        # Initialize models and data
        self.tokenizer = tokenizer
        self.source_model = source_model
        self.target_model = target_model
        self.working_model = None  # Will be initialized when needed
        self.device = next(target_model.parameters()).device
        
        # Store datasets
        self.pseudo_clean_test = pseudo_clean_test
        self.pseudo_poisoned_test = pseudo_poisoned_test
        self.true_clean_test = true_clean_test
        self.true_poisoned_test = true_poisoned_test
        
        # Store alpha parameter
        self.alpha = alpha
        
        # Initialize counters and timing
        self.total_evaluations = 0
        self.start_time = time.time()
        
        # Get model info
        self.model_type, self.num_layers = get_model_type_and_layers(target_model)
        print(f"Detected {self.model_type} with {self.num_layers} layers")
        
        # Initialize LoRA module set
        self.module_set = set()
        
        # Find common LoRA modules in both models
        print("\nIdentifying common LoRA modules in both models...")
        source_modules = set()
        target_modules = set()
        
        # Check source model LoRA modules
        for name, module in source_model.named_modules():
            if hasattr(module, 'lora_A'):
                parts = name.split('.')
                if len(parts) > 0:
                    module_type = parts[-1]
                    source_modules.add(module_type)
        
        # Check target model LoRA modules
        for name, module in target_model.named_modules():
            if hasattr(module, 'lora_A'):
                parts = name.split('.')
                if len(parts) > 0:
                    module_type = parts[-1]
                    target_modules.add(module_type)
        
        # Find common modules
        self.module_set = source_modules.intersection(target_modules)
        print(f"Common LoRA modules found: {sorted(self.module_set)}")
        
        # Fallback to default module set if no common modules found
        if not self.module_set:
            print("WARNING: No common LoRA modules found via hasattr detection.")
            
            # Try to get modules from PEFT config
            if hasattr(source_model, 'peft_config') and source_model.peft_config:
                config = list(source_model.peft_config.values())[0]
                if hasattr(config, 'target_modules'):
                    # Extract module names from full module paths
                    peft_modules = set()
                    for module_path in config.target_modules:
                        module_name = module_path.split('.')[-1]
                        peft_modules.add(module_name)
                    
                    if peft_modules:
                        self.module_set = peft_modules
                        print(f"Using modules from PEFT config: {sorted(self.module_set)}")
                    else:
                        self.module_set = {'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'}
                        print("Using default decoder module set.")
                else:
                    self.module_set = {'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'}
                    print("Using default decoder module set.")
            else:
                self.module_set = {'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'}
                print("Using default decoder module set.")
        
        print("=" * 80)
        print("MODEL PURIFIER INITIALIZED")
        print("=" * 80)
        print(f"Proxy model: {source_lora}")
        print(f"Suspect victim model: {target_lora}")
        print(f"Model type: {self.model_type}")
        print(f"Number of layers: {self.num_layers}")
        print(f"Available modules: {sorted(self.module_set)}")
        print(f"Alpha parameter: {alpha}")
        
        # Create working model initially
        self.create_working_model()
        
        # Evaluate baseline performance
        print("\nEvaluating baseline performance...")
        
        # Evaluate proxy model on proxy sets
        print("Proxy model evaluation...")
        source_pseudo_acc, source_pseudo_asr = self.evaluate_model(
            self.source_model,
            self.pseudo_clean_test,
            self.pseudo_poisoned_test
        )
        
        # Evaluate proxy model on true test sets
        source_true_acc, source_true_asr = self.evaluate_model(
            self.source_model,
            self.true_clean_test,
            self.true_poisoned_test
        )
        
        # Evaluate suspect victim model on proxy sets
        print("Suspect victim model evaluation...")
        self.original_pseudo_acc, self.original_pseudo_asr = self.evaluate_model(
            self.target_model, 
            self.pseudo_clean_test, 
            self.pseudo_poisoned_test
        )
        
        # Evaluate suspect victim model on true test sets
        self.original_true_acc, self.original_true_asr = self.evaluate_model(
            self.target_model,
            self.true_clean_test,
            self.true_poisoned_test
        )
        
        # Store source model metrics for later display
        self.source_pseudo_acc = source_pseudo_acc
        self.source_pseudo_asr = source_pseudo_asr
        
        print(f"\nProxy Model Baseline (Proxy Sets):")
        print(f"  Clean Accuracy: {source_pseudo_acc:.4f}")
        print(f"  Attack Success Rate: {source_pseudo_asr:.4f}")
        
        print(f"\nProxy Model Baseline (True Test Sets):")
        print(f"  Clean Accuracy: {source_true_acc:.4f}")
        print(f"  Attack Success Rate: {source_true_asr:.4f}")
        
        print(f"\nSuspect Victim Model Baseline (Proxy Sets):")
        print(f"  Clean Accuracy: {self.original_pseudo_acc:.4f}")
        print(f"  Attack Success Rate: {self.original_pseudo_asr:.4f}")
        
        print(f"\nSuspect Victim Model Baseline (True Test Sets):")
        print(f"  Clean Accuracy: {self.original_true_acc:.4f}")
        print(f"  Attack Success Rate: {self.original_true_asr:.4f}")
        print("=" * 80)

    def create_working_model(self):
        """Create a working model by moving target model to CPU, copying, and moving back."""
        # Move target model to CPU for copying
        self.target_model.to('cpu')
        
        # Create a deep copy on CPU
        self.working_model = copy.deepcopy(self.target_model)
        
        # Move both models back to the original device
        self.target_model.to(self.device)
        self.working_model.to(self.device)
        
        # Ensure working model also has correct padding token
        if hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id is not None:
            self.working_model.config.pad_token_id = self.tokenizer.pad_token_id

    def get_module_params(self, model: Any, layer_idx: int, module_name: str) -> Any:
        """Get LoRA parameters for a specific layer and module."""
        try:
            # Navigate to the correct layer based on model structure
            if hasattr(model, 'base_model'):
                # This is a PEFT model
                base_model = model.base_model
                if hasattr(base_model, 'model'):
                    # LlamaForSequenceClassification -> LlamaModel
                    llama_model = base_model.model
                    if hasattr(llama_model, 'model'):
                        # Get the actual transformer layers
                        transformer = llama_model.model
                        layer = transformer.layers[layer_idx]
                    else:
                        layer = llama_model.layers[layer_idx]
                else:
                    layer = base_model.layers[layer_idx]
            else:
                # Direct model access
                if hasattr(model, 'model'):
                    transformer = model.model
                    layer = transformer.layers[layer_idx]
                else:
                    layer = model.layers[layer_idx]

            # Get the specific module
            if module_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                return getattr(layer.self_attn, module_name)
            elif module_name in ['gate_proj', 'up_proj', 'down_proj']:
                return getattr(layer.mlp, module_name)
            else:
                raise ValueError(f"Module {module_name} not found in layer {layer_idx}")
                
        except Exception as e:
            print(f"Error in get_module_params for layer {layer_idx}, module {module_name}: {e}")
            print(f"Model type: {type(model)}")
            if hasattr(model, 'base_model'):
                print(f"base_model type: {type(model.base_model)}")
            raise
    
    def substitute_modules(self, modules: Set[str], layers: Set[int]):
        """Substitute specified LoRA modules with those from source model."""
        # Re-create working model to start fresh
        self.create_working_model()

        # Count successful and failed substitutions
        successful = 0
        failed = 0

        # Substitute the modules
        for layer_idx in layers:
            for module_name in modules:
                try:
                    # Get modules with error checking
                    target_module = self.get_module_params(self.working_model, layer_idx, module_name)
                    source_module = self.get_module_params(self.source_model, layer_idx, module_name)
                    
                    # Check if both modules have LoRA weights
                    target_has_lora = hasattr(target_module, 'lora_A')
                    source_has_lora = hasattr(source_module, 'lora_A')
                    
                    if not target_has_lora:
                        print(f"WARNING: Target module {module_name} in layer {layer_idx} does not have LoRA weights")
                        failed += 1
                        continue
                        
                    if not source_has_lora:
                        print(f"WARNING: Source module {module_name} in layer {layer_idx} does not have LoRA weights")
                        failed += 1
                        continue
                    
                    # Both have LoRA, proceed with substitution
                    # Copy weights on CPU to save GPU memory
                    source_A = source_module.lora_A['default'].weight.data.cpu()
                    source_B = source_module.lora_B['default'].weight.data.cpu()
                    
                    # Update target model weights
                    target_module.lora_A['default'].weight.data.copy_(source_A.to(self.device))
                    target_module.lora_B['default'].weight.data.copy_(source_B.to(self.device))
                    
                    successful += 1
                    
                except Exception as e:
                    print(f"Error substituting {module_name} in layer {layer_idx}: {e}")
                    failed += 1
        
        # Make sure the working model is on the right device
        self.working_model.to(self.device)

    def replace_and_evaluate(self, modules: Set[str], layers: Set[int]) -> Tuple[float, float, float, float]:
        """Replace specified modules and evaluate the model."""
        # Clear CUDA cache before operations
        torch.cuda.empty_cache()
        
        self.substitute_modules(modules, layers)
        acc, asr = self.evaluate_model(
            self.working_model,
            self.pseudo_clean_test,
            self.pseudo_poisoned_test
        )
        acc_drop = self.original_pseudo_acc - acc
        asr_drop = self.original_pseudo_asr - asr
        
        # Increment evaluation counter
        self.total_evaluations += 1
        
        return acc, asr, acc_drop, asr_drop
    
    def evaluate_model(self, model: Any, clean_test: List[Dict[str, Any]], poison_test: List[Dict[str, Any]]) -> Tuple[float, float]:
        """Evaluate model performance using unified evaluation function."""
        clean_acc = evaluate_decoder_model(model, self.tokenizer, clean_test)
        poison_asr = evaluate_decoder_model(model, self.tokenizer, poison_test)
        
        return float(clean_acc), float(poison_asr)

    def find_best_strategy(self) -> Tuple[Set[str], Set[int]]:
        """
        Find the best purification strategy using guided substitution search.
        
        Returns:
            Tuple of (best_modules, best_layers)
        """
        search_start_time = time.time()
        
        best_modules, best_layers = find_best_strategy(self)
        
        # Print search completion timing
        search_time = time.time() - search_start_time
        print(f"\nSearch completed in {search_time:.2f} seconds")
        print(f"Total evaluations performed: {self.total_evaluations}")
        if self.total_evaluations > 0:
            print(f"Average time per evaluation: {search_time / self.total_evaluations:.2f} seconds")
        
        return best_modules, best_layers

    def get_final_results(self, best_modules: Set[str], best_layers: Set[int]) -> Dict[str, Any]:
        """
        Get comprehensive final results.
        
        Args:
            best_modules: Selected best modules
            best_layers: Selected best layers
            
        Returns:
            Dictionary containing all results
        """
        # Apply best strategy and evaluate
        final_acc, final_asr, final_acc_drop, final_asr_drop = self.replace_and_evaluate(
            best_modules, best_layers
        )
        
        total_time = time.time() - self.start_time
        
        results = {
            "strategy_info": {
                "modules": sorted(list(best_modules)),
                "layers": sorted(list(best_layers)),
                "total_replacements": len(best_modules) * len(best_layers)
            },
            "performance_metrics": {
                "proxy_sets": {
                    "clean_accuracy": final_acc,
                    "clean_accuracy_drop": final_acc_drop,
                    "attack_success_rate": final_asr,
                    "asr_drop": final_asr_drop,
                    "asr_reduction_rate": final_asr_drop / self.original_pseudo_asr if self.original_pseudo_asr > 0 else 0
                }
            },
            "search_statistics": {
                "total_evaluations": self.total_evaluations,
                "total_time_seconds": total_time,
                "avg_time_per_evaluation": total_time / self.total_evaluations if self.total_evaluations > 0 else 0
            }
        }
        
        return results

    def save_purified_model(self, save_dir: str, modules: Set[str], layers: Set[int]) -> str:
        """
        Save the purified model.
        
        Args:
            save_dir: Directory to save the model
            modules: Selected modules
            layers: Selected layers
            
        Returns:
            Path where model was saved
        """
        save_path = Path(save_dir) / "purified_model"
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Apply the best strategy
        self.substitute_modules(modules, layers)
        
        # Save the model
        self.working_model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        print(f"Purified model saved to: {save_path}")
        return str(save_path)
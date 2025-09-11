"""
Core model purification logic for GMS encoder method
"""

import copy
import time
import json
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
from typing import Set, Tuple, Dict, Any

from .config import MODULE_MAPPING
from .model_utils import create_classifier, get_model_type_and_layers, get_module_path, extract_predictions_from_pipeline
from .strategy import find_best_strategy


class ModelPurifier:
    """
    Core class for performing guided module search (GMS) purification.
    
    This class implements the GMS method for purifying backdoored models by
    selectively replacing modules and layers from a clean source model.
    """
    
    def __init__(self,
                 source_path: str,
                 target_path: str,
                 tokenizer: Any,
                 source_model: Any,
                 target_model: Any,
                 true_clean_test: Any,
                 proxy_clean_set: Any,
                 true_poisoned_test: Any,
                 proxy_suspect_set: Any,
                 alpha: float = 0.4):
        """
        Initialize the model purifier.
        
        Args:
            source_path: Path to proxy model
            target_path: Path to suspect victim model
            tokenizer: Tokenizer for both models
            source_model: Proxy model (source of proxy modules)
            target_model: Suspect victim model (target for purification)
            true_clean_test: True clean test dataset
            proxy_clean_set: Proxy clean set (auxiliary dataset from SEEP random sampling)
            true_poisoned_test: True poisoned test dataset
            proxy_suspect_set: Proxy suspect set (auxiliary dataset from SEEP suspicious sampling)
            alpha: Balance parameter (α for accuracy, 1-α for ASR drop)
        """
        
        # Store paths
        self.source_path = source_path
        self.target_path = target_path
        
        # Initialize models and tokenizer
        self.tokenizer = tokenizer
        self.source_model = source_model
        self.target_model = target_model
        self.working_model = copy.deepcopy(target_model)
        
        # Store datasets
        self.true_clean_test = true_clean_test
        self.proxy_clean_set = proxy_clean_set
        self.true_poisoned_test = true_poisoned_test
        self.proxy_suspect_set = proxy_suspect_set
        
        # Store alpha parameter
        self.alpha = alpha
        
        # Initialize counters and timing
        self.total_evaluations = 0
        self.start_time = time.time()
        
        # Get model architecture info
        self.model_type, self.num_layers = get_model_type_and_layers(target_model)
        self.module_set = {'Q', 'K', 'V', 'O', 'F', 'P'}
        
        print("=" * 80)
        print("MODEL PURIFIER INITIALIZED")
        print("=" * 80)
        print(f"Proxy model: {source_path}")
        print(f"Suspect victim model: {target_path}")
        print(f"Model type: {self.model_type}")
        print(f"Number of layers: {self.num_layers}")
        print(f"Available modules: {sorted(self.module_set)}")
        print(f"Alpha parameter: {alpha}")
        
        # Evaluate baseline performance
        print("\nEvaluating baseline performance...")
        
        # Evaluate proxy model
        print("Proxy model evaluation...")
        ref_acc, ref_asr = self.evaluate_model(self.source_model, use_true_set=False)
        ref_true_acc, ref_true_asr = self.evaluate_model(self.source_model, use_true_set=True)
        
        # Evaluate suspect victim model
        print("Suspect victim model evaluation...")
        self.original_acc, self.original_asr = self.evaluate_model(self.target_model, use_true_set=False)
        self.original_true_acc, self.original_true_asr = self.evaluate_model(self.target_model, use_true_set=True)
        
        print(f"\Proxy Model Baseline (Proxy Sets):")
        print(f"  Clean Accuracy: {ref_acc:.4f}")
        print(f"  Attack Success Rate: {ref_asr:.4f}")
        
        print(f"\Proxy Model Baseline (True Test Sets):")
        print(f"  Clean Accuracy: {ref_true_acc:.4f}")
        print(f"  Attack Success Rate: {ref_true_asr:.4f}")
        
        print(f"\nSuspect Victim Model Baseline (Proxy Sets):")
        print(f"  Clean Accuracy: {self.original_acc:.4f}")
        print(f"  Attack Success Rate: {self.original_asr:.4f}")
        
        print(f"\nSuspect Victim Model Baseline (True Test Sets):")
        print(f"  Clean Accuracy: {self.original_true_acc:.4f}")
        print(f"  Attack Success Rate: {self.original_true_asr:.4f}")
        print("=" * 80)

    def evaluate_model(self, model: Any, use_true_set: bool = False) -> Tuple[float, float]:
        """
        Evaluate model performance on clean and poisoned test sets.
        
        Args:
            model: Model to evaluate
            use_true_set: Whether to use true test sets (vs pseudo test sets)
            
        Returns:
            Tuple of (clean_accuracy, attack_success_rate)
        """
        self.total_evaluations += 1
        
        if use_true_set:
            clean_dataset = self.true_clean_test
            poisoned_dataset = self.true_poisoned_test
        else:
            clean_dataset = self.proxy_clean_set
            poisoned_dataset = self.proxy_suspect_set
        
        # Create classifier pipeline
        classifier = create_classifier(model, self.tokenizer, batch_size=64)
        
        # Evaluate on clean data
        clean_sentences = clean_dataset['sentence']
        clean_labels = clean_dataset['label']
        clean_results = classifier(clean_sentences)
        
        # Process clean results
        clean_predictions = extract_predictions_from_pipeline(clean_results)
        clean_accuracy = np.mean([pred == label for pred, label in zip(clean_predictions, clean_labels)])

        # Evaluate on poisoned data
        poison_sentences = poisoned_dataset['sentence']
        poison_labels = poisoned_dataset['label']
        poison_results = classifier(poison_sentences)

        # Process poison results
        poison_predictions = extract_predictions_from_pipeline(poison_results)
        attack_success_rate = np.mean([pred == label for pred, label in zip(poison_predictions, poison_labels)])

        return clean_accuracy, attack_success_rate

    def substitute_modules(self, modules_to_replace: Set[str], layers_to_replace: Set[int]) -> None:
        """
        Replace specified modules in specified layers with modules from proxy model.
        
        Args:
            modules_to_replace: Set of module types to replace (Q, K, V, O, F, P)
            layers_to_replace: Set of layer indices to replace modules in
        """
        if not modules_to_replace or not layers_to_replace:
            return  # Nothing to replace
            
        for layer_idx in layers_to_replace:
            for module_name in modules_to_replace:
                # Get full module paths
                source_path = get_module_path(self.model_type, layer_idx, module_name)
                target_path = get_module_path(self.model_type, layer_idx, module_name)
                
                try:
                    # Get source and target modules
                    source_module = dict(self.source_model.named_modules())[source_path]
                    target_module = dict(self.working_model.named_modules())[target_path]
                    
                    # Copy parameters from proxy model to working model
                    with torch.no_grad():
                        if hasattr(source_module, 'weight') and hasattr(target_module, 'weight'):
                            target_module.weight.copy_(source_module.weight)
                        if hasattr(source_module, 'bias') and hasattr(target_module, 'bias') and source_module.bias is not None:
                            target_module.bias.copy_(source_module.bias)
                            
                except KeyError:
                    print(f"Warning: Could not find module {source_path} or {target_path}")
                except Exception as e:
                    print(f"Warning: Error replacing module {source_path}: {e}")

    def replace_and_evaluate(self, modules: Set[str], layers: Set[int]) -> Tuple[float, float, float, float]:
        """
        Replace modules and evaluate the resulting model.
        
        Args:
            modules: Set of module types to replace
            layers: Set of layer indices to replace
            
        Returns:
            Tuple of (accuracy, asr, accuracy_drop, asr_drop)
        """
        # Reset working model to original target model
        self.working_model = copy.deepcopy(self.target_model)
        
        # Apply substitutions
        self.substitute_modules(modules, layers)
        
        # Evaluate on pseudo test sets (used for search guidance)
        acc, asr = self.evaluate_model(self.working_model, use_true_set=False)
        
        # Calculate drops
        acc_drop = (self.original_acc - acc) / self.original_acc if self.original_acc > 0 else 0
        asr_drop = (self.original_asr - asr) / self.original_asr if self.original_asr > 0 else 0
        
        return acc, asr, acc_drop, asr_drop

    def find_best_strategy(self) -> Tuple[Set[str], Set[int]]:
        """
        Find the best purification strategy using top-down search.
        
        Returns:
            Tuple of (best_modules, best_layers)
        """
        print(f"\nStarting guided substitution search")
        print(f"Using auxiliary proxy datasets for guidance:")
        print(f"  Proxy clean set: {len(self.proxy_clean_set)} samples")
        print(f"  Proxy suspect set: {len(self.proxy_suspect_set)} samples")
        print()
        
        start_time = time.time()
        best_modules, best_layers = find_best_strategy(self)
        search_time = time.time() - start_time
        
        print(f"\nSearch completed in {search_time:.2f} seconds")
        print(f"Total evaluations performed: {self.total_evaluations}")
        print(f"Average time per evaluation: {search_time/self.total_evaluations:.2f} seconds")
        
        return best_modules, best_layers

    def save_purified_model(self, save_dir: str, modules: Set[str], layers: Set[int]) -> None:
        """
        Apply final purification strategy and save the purified model.
        
        Args:
            save_dir: Directory to save the purified model
            modules: Final set of modules to replace
            layers: Final set of layers to replace
        """
        # Apply final substitutions
        self.substitute_modules(modules, layers)
        
        # Save model and tokenizer
        save_path = Path(save_dir) / "purified_model"
        save_path.mkdir(parents=True, exist_ok=True)
        
        self.working_model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        print(f"\nPurified model saved to: {save_path}")
        
        return str(save_path)

    def get_final_results(self, modules: Set[str], layers: Set[int]) -> Dict[str, Any]:
        """
        Get comprehensive results for the final purified model.
        
        Args:
            modules: Final set of modules that were replaced
            layers: Final set of layers that were replaced
            
        Returns:
            Dictionary containing all results and statistics
        """
        # Ensure working model has the final substitutions applied
        self.substitute_modules(modules, layers)
        
        # Evaluate on both pseudo and true test sets
        pseudo_acc, pseudo_asr = self.evaluate_model(self.working_model, use_true_set=False)
        true_acc, true_asr = self.evaluate_model(self.working_model, use_true_set=True)
        
        total_time = time.time() - self.start_time
        
        results = {
            "strategy_info": {
                "modules": sorted(list(modules)),
                "layers": sorted(list(layers)),
                "num_modules": len(modules),
                "num_layers": len(layers),
                "total_replacements": len(modules) * len(layers)
            },
            "performance_metrics": {
                "proxy_sets": {
                    "clean_accuracy": pseudo_acc,
                    "clean_accuracy_drop": self.original_acc - pseudo_acc,
                    "attack_success_rate": pseudo_asr,
                    "asr_drop": self.original_asr - pseudo_asr,
                    "asr_reduction_rate": (self.original_asr - pseudo_asr) / self.original_asr if self.original_asr > 0 else 0
                },
                "true_test": {
                    "clean_accuracy": true_acc,
                    "clean_accuracy_drop": self.original_true_acc - true_acc,
                    "attack_success_rate": true_asr,
                    "asr_drop": self.original_true_asr - true_asr,
                    "asr_reduction_rate": (self.original_true_asr - true_asr) / self.original_true_asr if self.original_true_asr > 0 else 0
                }
            },
            "baseline_performance": {
                "proxy_sets": {
                    "clean_accuracy": self.original_acc,
                    "attack_success_rate": self.original_asr
                },
                "true_test": {
                    "clean_accuracy": self.original_true_acc,
                    "attack_success_rate": self.original_true_asr
                }
            },
            "search_statistics": {
                "total_evaluations": self.total_evaluations,
                "total_time_seconds": total_time,
                "avg_time_per_evaluation": total_time / self.total_evaluations if self.total_evaluations > 0 else 0
            },
            "configuration": {
                "alpha": self.alpha,
                "model_type": self.model_type,
                "num_layers": self.num_layers,
                "source_path": self.source_path,
                "target_path": self.target_path
            }
        }
        
        return results
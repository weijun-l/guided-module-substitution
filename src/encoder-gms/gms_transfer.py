"""
Transfer module for applying pre-discovered GMS strategies to test transferability
"""

import argparse
import time
import torch
import json
import logging
import sys
import builtins
import functools
from pathlib import Path
from typing import Set
from transformers import set_seed

from .dataset import load_json_dataset
from .model_utils import load_models
from .purifier import ModelPurifier
from .config import DEFAULT_SEED

# Configure logging to reduce noise
logging.getLogger("evaluate").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.WARNING)
builtins.print = functools.partial(print, flush=True)


def parse_strategy_modules(modules_str: str) -> Set[str]:
    """Parse comma-separated module string into set."""
    return set(modules_str.strip().split(','))


def parse_strategy_layers(layers_str: str) -> Set[int]:
    """Parse comma-separated layer string into set of integers."""
    return set(int(x.strip()) for x in layers_str.strip().split(','))


def main():
    """Main function for GMS strategy transfer."""
    parser = argparse.ArgumentParser(
        description="GMS Strategy Transfer - Apply pre-discovered strategies to test transferability"
    )
    
    # Required paths
    parser.add_argument("--source_path", type=str, required=True,
                        help="Path to the proxy model (source of proxy modules)")
    parser.add_argument("--target_path", type=str, required=True,
                        help="Path to the suspect victim model (target for purification)")
    
    # Test datasets and auxiliary proxy sets
    parser.add_argument("--true_clean_test", type=str, required=True,
                        help="Path to true clean test dataset")
    parser.add_argument("--proxy_clean_set", type=str, required=True,
                        help="Path to proxy clean set (from SEEP random sampling)")
    parser.add_argument("--true_poisoned_test", type=str, required=True,
                        help="Path to true poisoned test dataset")
    parser.add_argument("--proxy_suspect_set", type=str, required=True,
                        help="Path to proxy suspect set (from SEEP suspicious sampling)")
    
    # Pre-discovered strategy
    parser.add_argument("--strategy_modules", type=str, required=True,
                        help="Comma-separated list of modules to replace (e.g., 'Q,K,V,O,F,P')")
    parser.add_argument("--strategy_layers", type=str, required=True,
                        help="Comma-separated list of layer indices to replace (e.g., '1,3,4,5,6,7')")
    parser.add_argument("--strategy_source", type=str, required=True,
                        help="Description of where the strategy came from (e.g., 'sst2_badnet_imdb')")
    
    # Output directory
    parser.add_argument("--save_dir", type=str, required=True,
                        help="Directory to save results and purified model")
    
    # Optional parameters
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use ('auto', 'cuda', 'cpu')")
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    # Set seed
    set_seed(args.seed)
    
    print("=" * 80)
    print("GMS STRATEGY TRANSFER")
    print("=" * 80)
    print(f"Applying pre-discovered strategy from: {args.strategy_source}")
    print(f"Device: {device}")
    print(f"Random seed: {args.seed}")
    print("=" * 80)
    
    # Parse strategy
    try:
        strategy_modules = parse_strategy_modules(args.strategy_modules)
        strategy_layers = parse_strategy_layers(args.strategy_layers)
    except ValueError as e:
        print(f"ERROR: Failed to parse strategy parameters: {e}")
        sys.exit(1)
    
    print("Pre-discovered Strategy:")
    print(f"  Modules: {sorted(strategy_modules)} ({len(strategy_modules)} total)")
    print(f"  Layers: {sorted(strategy_layers)} ({len(strategy_layers)} total)")
    print(f"  Total replacements: {len(strategy_modules) * len(strategy_layers)}")
    print(f"  Source: {args.strategy_source}")
    print()
    
    # Load datasets
    try:
        print("Loading datasets...")
        true_clean_test = load_json_dataset(args.true_clean_test)
        proxy_clean_set = load_json_dataset(args.proxy_clean_set)
        true_poisoned_test = load_json_dataset(args.true_poisoned_test)
        proxy_suspect_set = load_json_dataset(args.proxy_suspect_set)
        
        print(f"Dataset sizes:")
        print(f"  True clean test: {len(true_clean_test)}")
        print(f"  Proxy clean set: {len(proxy_clean_set)}")
        print(f"  True poisoned test: {len(true_poisoned_test)}")
        print(f"  Proxy suspect set: {len(proxy_suspect_set)}")
        print()
        
    except Exception as e:
        print(f"ERROR: Failed to load datasets: {e}")
        sys.exit(1)
    
    # Load models
    try:
        print("Loading models...")
        tokenizer, source_model, target_model = load_models(args.source_path, args.target_path)
        
        # Move models to device
        source_model = source_model.to(device)
        target_model = target_model.to(device)
        
        print(f"Models loaded successfully on device: {device}")
        print()
        
    except Exception as e:
        print(f"ERROR: Failed to load models: {e}")
        sys.exit(1)
    
    # Initialize purifier (alpha doesn't matter for transfer since we're not searching)
    purifier = ModelPurifier(
        source_path=args.source_path,
        target_path=args.target_path,
        tokenizer=tokenizer,
        source_model=source_model,
        target_model=target_model,
        true_clean_test=true_clean_test,
        proxy_clean_set=proxy_clean_set,
        true_poisoned_test=true_poisoned_test,
        proxy_suspect_set=proxy_suspect_set
    )
    
    # Store baseline metrics for later use
    ref_acc = purifier.evaluate_model(purifier.source_model, use_true_set=False)[0]
    ref_asr = purifier.evaluate_model(purifier.source_model, use_true_set=False)[1]
    ref_true_acc = purifier.evaluate_model(purifier.source_model, use_true_set=True)[0]
    ref_true_asr = purifier.evaluate_model(purifier.source_model, use_true_set=True)[1]
    
    # Apply the pre-discovered strategy
    print("=" * 80)
    print("APPLYING TRANSFER STRATEGY")
    print("=" * 80)
    print(f"Strategy from: {args.strategy_source}")
    print(f"Applying modules {sorted(strategy_modules)} to layers {sorted(strategy_layers)}")
    print()
    
    # Apply strategy and get metrics
    start_time = time.time()
    acc, asr, acc_drop, asr_drop = purifier.replace_and_evaluate(strategy_modules, strategy_layers)
    transfer_time = time.time() - start_time
    
    print(f"Transfer Strategy Applied:")
    print(f"  Modules replaced: {sorted(strategy_modules)} ({len(strategy_modules)} types)")
    print(f"  Layers modified: {sorted(strategy_layers)} ({len(strategy_layers)} layers)")
    print(f"  Total replacements: {len(strategy_modules) * len(strategy_layers)}")
    print(f"  Transfer time: {transfer_time:.2f} seconds")
    print()
    
    print("Performance on Proxy Sets:")
    print(f"  Clean Accuracy: {acc:.4f} (drop: {acc_drop:.4f})")
    print(f"  Attack Success Rate: {asr:.4f} (drop: {asr_drop:.4f})")
    print()
    
    # Evaluate on true test sets
    true_acc, true_asr = purifier.evaluate_model(purifier.working_model, use_true_set=True)
    true_acc_drop = purifier.original_true_acc - true_acc
    true_asr_drop = purifier.original_true_asr - true_asr
    true_asr_reduction_rate = true_asr_drop / purifier.original_true_asr if purifier.original_true_asr > 0 else 0
    
    print("Performance on True Test Sets:")
    print(f"  Clean Accuracy: {true_acc:.4f} (drop: {true_acc_drop:.4f})")
    print(f"  Attack Success Rate: {true_asr:.4f} (drop: {true_asr_drop:.4f})")
    print(f"  ASR Reduction Rate: {true_asr_reduction_rate:.4f}")
    
    # Save the purified model
    print("\n" + "=" * 80)
    print("SAVING TRANSFER RESULTS")
    print("=" * 80)
    
    # Create output directory
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    
    # Save purified model
    purified_model_path = purifier.save_purified_model(args.save_dir, strategy_modules, strategy_layers)
    
    # Prepare comprehensive results
    transfer_results = {
        "transfer_info": {
            "strategy_source": args.strategy_source,
            "modules": sorted(list(strategy_modules)),
            "layers": sorted(list(strategy_layers)),
            "num_modules": len(strategy_modules),
            "num_layers": len(strategy_layers),
            "total_replacements": len(strategy_modules) * len(strategy_layers),
            "transfer_time_seconds": transfer_time
        },
        "performance_metrics": {
            "proxy_sets": {
                "clean_accuracy": acc,
                "clean_accuracy_drop": acc_drop,
                "attack_success_rate": asr,
                "asr_drop": asr_drop,
                "asr_reduction_rate": asr_drop / purifier.original_asr if purifier.original_asr > 0 else 0
            },
            "true_test": {
                "clean_accuracy": true_acc,
                "clean_accuracy_drop": true_acc_drop,
                "attack_success_rate": true_asr,
                "asr_drop": true_asr_drop,
                "asr_reduction_rate": true_asr_reduction_rate
            }
        },
        "baseline_performance": {
            "proxy_model": {
                "proxy_sets": {
                    "clean_accuracy": ref_acc,
                    "attack_success_rate": ref_asr
                },
                "true_test": {
                    "clean_accuracy": ref_true_acc,
                    "attack_success_rate": ref_true_asr
                }
            },
            "suspect_victim_model": {
                "proxy_sets": {
                    "clean_accuracy": purifier.original_acc,
                    "attack_success_rate": purifier.original_asr
                },
                "true_test": {
                    "clean_accuracy": purifier.original_true_acc,
                    "attack_success_rate": purifier.original_true_asr
                }
            }
        },
        "transfer_statistics": {
            "total_evaluations": purifier.total_evaluations,
            "transfer_time_seconds": transfer_time
        },
        "configuration": {
            "model_type": purifier.model_type,
            "num_layers": purifier.num_layers,
            "source_path": args.source_path,
            "target_path": args.target_path,
            "seed": args.seed
        }
    }
    
    # Save results
    results_path = Path(args.save_dir) / "transfer_results.json"
    with open(results_path, 'w') as f:
        json.dump(transfer_results, f, indent=2)
    
    print(f"Transfer results saved to: {results_path}")
    print(f"Purified model saved to: {purified_model_path}")
    
    # Summary
    print("\n" + "=" * 80)
    print("TRANSFER SUMMARY")
    print("=" * 80)
    print(f"Strategy source: {args.strategy_source}")
    print(f"Applied to: {Path(args.target_path).name}")
    print(f"Proxy model: {Path(args.source_path).name}")
    print()
    print("Key Results:")
    print(f"  ASR reduction (true test): {true_asr_reduction_rate:.1%}")
    print(f"  Clean accuracy preserved: {(1 - true_acc_drop/purifier.original_true_acc):.1%}")
    print(f"  Total module replacements: {len(strategy_modules) * len(strategy_layers)}")
    print(f"  Transfer time: {transfer_time:.2f} seconds")
    
    print("\nStrategy transfer completed.")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
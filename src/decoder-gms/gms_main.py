"""
Main entry point for GMS (Guided Module Search) decoder purification
"""

import argparse
import torch
import json
import logging
import sys
import builtins
import functools
from pathlib import Path
from transformers import set_seed

from .dataset import load_json_dataset
from .model_utils import load_models
from .purifier import LLaMAPurifier
from .config import DEFAULT_SEED, DEFAULT_ALPHA


# Configure logging to reduce noise
logging.getLogger("evaluate").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.WARNING)
builtins.print = functools.partial(print, flush=True)




def main():
    """Main function for GMS decoder purification."""
    parser = argparse.ArgumentParser(
        description="GMS (Guided Module Search) for Decoder Model Purification"
    )
    
    # Required paths
    parser.add_argument("--base_model_path", type=str, required=True,
                        help="Path to the base LLaMA model")
    parser.add_argument("--source_path", type=str, required=True,
                        help="Path to the source LoRA weights")
    parser.add_argument("--target_path", type=str, required=True,
                        help="Path to the target LoRA weights")
    
    # Test datasets and auxiliary proxy sets
    parser.add_argument("--true_clean_test", type=str, required=True,
                        help="Path to true clean test dataset")
    parser.add_argument("--proxy_clean_set", type=str, required=True,
                        help="Path to proxy clean set (from SEEP random sampling)")
    parser.add_argument("--true_poisoned_test", type=str, required=True,
                        help="Path to true poisoned test dataset")
    parser.add_argument("--proxy_suspect_set", type=str, required=True,
                        help="Path to proxy suspect set (from SEEP suspicious sampling)")
    
    # Output directory
    parser.add_argument("--save_dir", type=str, required=True,
                        help="Directory to save results and purified model")
    
    # Search parameters
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                        help="Balance parameter: α for accuracy preservation, (1-α) for ASR reduction (default: 0.2)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Random seed for reproducibility (default: 42)")
    
    # Optional parameters
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use ('cuda', 'cpu', or 'auto')")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")

    args = parser.parse_args()
    
    # Set up device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    # Create output directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    log_file = save_dir / "gms_purification.log"
    
    # Print configuration
    print("="*100)
    print("GMS (GUIDED MODULE SEARCH) DECODER PURIFICATION")
    print("="*100)
    print(f"Configuration:")
    print(f"  Alpha (accuracy weight): {args.alpha}")
    print(f"  Random seed: {args.seed}")
    print(f"  Device: {device}")
    print(f"  Save directory: {args.save_dir}")
    print(f"  Log file: {log_file}")
    print()
    
    print(f"Model paths:")
    print(f"  Proxy model: {args.source_path}")
    print(f"  Suspect victim model: {args.target_path}")
    print()
    
    print(f"Dataset paths:")
    print(f"  True clean test: {args.true_clean_test}")
    print(f"  Proxy clean set: {args.proxy_clean_set}")
    print(f"  True poisoned test: {args.true_poisoned_test}")
    print(f"  Proxy suspect set: {args.proxy_suspect_set}")
    print("="*100)

    print("\nStarting GMS purification process...")

    # Set random seed
    set_seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    try:
        # Load datasets
        print("\n" + "="*50)
        print("LOADING DATASETS")
        print("="*50)
        
        true_clean_test = load_json_dataset(args.true_clean_test)
        proxy_clean_set = load_json_dataset(args.proxy_clean_set)
        true_poisoned_test = load_json_dataset(args.true_poisoned_test)
        proxy_suspect_set = load_json_dataset(args.proxy_suspect_set)

        print(f"Dataset loading completed:")
        print(f"  True clean test: {len(true_clean_test)} samples")
        print(f"  Proxy clean set: {len(proxy_clean_set)} samples")
        print(f"  True poisoned test: {len(true_poisoned_test)} samples")
        print(f"  Proxy suspect set: {len(proxy_suspect_set)} samples")

        # Load models
        print("\n" + "="*50)
        print("LOADING MODELS")
        print("="*50)
        print(f"Using device: {device}")
        print(f"Loading tokenizer from: {args.target_path}")
        print(f"Loading proxy model from: {args.source_path}")
        print(f"Loading suspect victim model from: {args.target_path}")
        
        tokenizer, source_model, target_model = load_models(
            args.base_model_path,
            args.source_path,
            args.target_path
        )
        print("Model loading completed successfully")

        # Initialize purifier
        print("\n" + "="*50)
        print("INITIALIZING PURIFIER")
        print("="*50)
        
        purifier = LLaMAPurifier(
            base_model_path=args.base_model_path,
            source_lora=args.source_path,
            target_lora=args.target_path,
            tokenizer=tokenizer,
            source_model=source_model,
            target_model=target_model,
            pseudo_clean_test=proxy_clean_set,
            pseudo_poisoned_test=proxy_suspect_set,
            true_clean_test=true_clean_test,
            true_poisoned_test=true_poisoned_test,
            alpha=args.alpha
        )
        
        print("Model purifier initialized successfully")

        # Perform guided search
        print("\n" + "="*50)
        print("PERFORMING GUIDED SEARCH")
        print("="*50)
        
        best_modules, best_layers = purifier.find_best_strategy()

        # Save purified model
        print("\n" + "="*50)
        print("SAVING PURIFIED MODEL")
        print("="*50)
        
        model_save_path = purifier.save_purified_model(args.save_dir, best_modules, best_layers)

        # Get final comprehensive results
        print("\n" + "="*50)
        print("FINAL EVALUATION")
        print("="*50)
        
        final_results = purifier.get_final_results(best_modules, best_layers)

        # Display results
        print("\n" + "="*80)
        print("FINAL RESULTS SUMMARY")
        print("="*80)
        
        print(f"\nSelected Strategy:")
        print(f"  Modules: {final_results['strategy_info']['modules']}")
        print(f"  Layers: {final_results['strategy_info']['layers']}")
        print(f"  Total replacements: {final_results['strategy_info']['total_replacements']}")

        proxy_metrics = final_results['performance_metrics']['proxy_sets']
        true_metrics = final_results['performance_metrics']['true_test']
        
        print(f"\nResults on True Test Sets:")
        print(f"  Clean Accuracy: {true_metrics['clean_accuracy']:.4f} (drop: {true_metrics['clean_accuracy_drop']:.4f})")
        print(f"  Attack Success Rate: {true_metrics['attack_success_rate']:.4f} (drop: {true_metrics['asr_drop']:.4f})")
        print(f"  ASR Reduction Rate: {true_metrics['asr_reduction_rate']:.2%}")

        print(f"\nResults on Proxy Sets (Search Guidance):")
        print(f"  Clean Accuracy: {proxy_metrics['clean_accuracy']:.4f} (drop: {proxy_metrics['clean_accuracy_drop']:.4f})")
        print(f"  Attack Success Rate: {proxy_metrics['attack_success_rate']:.4f} (drop: {proxy_metrics['asr_drop']:.4f})")
        print(f"  ASR Reduction Rate: {proxy_metrics['asr_reduction_rate']:.2%}")

        search_stats = final_results['search_statistics']
        
        print(f"\nSearch Statistics:")
        print(f"  Total evaluations: {search_stats['total_evaluations']}")
        print(f"  Average time per evaluation: {search_stats['avg_time_per_evaluation']:.2f} seconds")

        # Save results to JSON
        results_file = save_dir / "final_results.json"
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)

        print(f"\nOutput Files:")
        print(f"  Purified model: {model_save_path}")
        print(f"  Results JSON: {results_file}")
        print(f"  Log file: {log_file}")
        
        print("\n" + "="*80)
        print("GMS PURIFICATION COMPLETED SUCCESSFULLY!")
        print("="*80)

        return 0

    except FileNotFoundError as e:
        print(f"\nERROR: Required file not found - {e}")
        return 1
    except Exception as e:
        print(f"\nERROR: Purification failed - {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
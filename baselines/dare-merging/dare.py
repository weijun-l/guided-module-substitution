import os
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from merging_methods import MergingMethod
import logging
import json
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import pipeline
from itertools import product
import csv
import copy
import gc

logging.getLogger("evaluate").setLevel(logging.ERROR)

def setup_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
    return logging.getLogger(__name__)

def load_json_dataset(file_path: str) -> Dataset:
    """Load a JSON dataset from file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
        df = pd.DataFrame(data)
        return Dataset.from_pandas(df)
    except FileNotFoundError:
        print(f"File not found: {file_path}", flush=True)
        raise
    except json.JSONDecodeError:
        print(f"Invalid JSON in file: {file_path}", flush=True)
        raise

def get_num_labels(dataset_name: str) -> int:
    """Get number of labels for each dataset."""
    dataset_labels = {
        'sst2': 2,
        'olid': 2,
        'agnews': 4,
        'qnli': 2,
        'mnli': 3
    }
    return dataset_labels.get(dataset_name, 2)


class ModelCache:
    def __init__(self, base_model_name, model_paths, device="cuda", cache_dir=None, batch_size=32):
        self.logger = setup_logger()
        self.device = device
        self.model_paths = model_paths
        self.base_model_name = base_model_name
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.num_labels = None
        self.tokenizer = None
        
        # Initialize test datasets dictionary
        self.test_datasets = {}
    
    def load_test_datasets(self, clean_test_path, victim_test_path):
        """Load test datasets if not already loaded"""
        if not self.test_datasets:
            self.test_datasets['clean'] = load_json_dataset(clean_test_path)
            self.test_datasets['victim'] = load_json_dataset(victim_test_path)
    
    def load_model(self, model_path=None):
        """Load model and tokenizer from the same path"""
        path = model_path or self.base_model_name
        
        # Load tokenizer from the first model path if not loaded yet
        if self.tokenizer is None and model_path is not None:
            self.logger.info(f"Loading tokenizer from {path}")
            self.tokenizer = AutoTokenizer.from_pretrained(path, cache_dir=self.cache_dir)
        
        # Determine num_labels from dataset name if not set
        if self.num_labels is None and model_path is not None:
            # Extract dataset name from model path
            dataset_name = None
            for known_dataset in ['sst2', 'olid', 'agnews', 'qnli', 'mnli']:
                if known_dataset in model_path.lower():
                    dataset_name = known_dataset
                    break
            
            if dataset_name:
                self.num_labels = get_num_labels(dataset_name)
            else:
                self.num_labels = 2  # Default fallback
            
            self.logger.info(f"Set num_labels to {self.num_labels} based on dataset: {dataset_name or 'default'}")
        
        model = AutoModelForSequenceClassification.from_pretrained(
            path,
            cache_dir=self.cache_dir,
            num_labels=self.num_labels
        ).to(self.device)
        
        model.eval()
        return model
    
    def evaluate_model(self, model, dataset_type='clean'):
        dataset = self.test_datasets[dataset_type]
        
        classifier = pipeline(
            "text-classification",
            model=model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1,
            batch_size=self.batch_size, 
            max_length=128,
            truncation=True,
            top_k=None
        )
        
        # Evaluate in batches
        total_correct = 0
        total_samples = 0
        
        for i in range(0, len(dataset), self.batch_size):
            batch = dataset[i:i + self.batch_size]
            results = classifier(batch['sentence'])
            
            logits = []
            for result in results:
                score_dict = {item['label']: item['score'] for item in result}
                num_labels = len(score_dict)
                ordered_scores = [score_dict[f'LABEL_{i}'] for i in range(num_labels)]
                logits.append(ordered_scores)
            
            logits = np.array(logits)
            labels = np.array(batch['label'])
            
            predictions = np.argmax(logits, axis=-1)
            total_correct += (predictions == labels).sum()
            total_samples += len(labels)
            
            # Clear CUDA cache after each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return total_correct / total_samples

def merge_models_with_params(
    model_cache: ModelCache,
    weight_mask_rate: float,
    scaling_coefficient: float,
    param_value_mask_rate: float,
    mask_strategy: str = "magnitude",
    use_weight_rescale: bool = True,
    logger=None
):
    if logger is None:
        logger = setup_logger()
    
    # Load base model
    merged_model = model_cache.load_model(model_cache.model_paths[0])
    
    # Load and merge models one at a time
    models_to_merge = []
    for path in model_cache.model_paths:
        model = model_cache.load_model(path)
        models_to_merge.append(model)
        
        if len(models_to_merge) == 1:  # Merge after loading each model
            merge_args = {
                "merged_model": merged_model,
                "models_to_merge": models_to_merge,
                "exclude_param_names_regex": [".*classifier.*"],
                "models_use_deepcopy": False,
                "scaling_coefficient": scaling_coefficient,
                "param_value_mask_rate": param_value_mask_rate,
                "weight_format": "delta_weight",
                "weight_mask_rates": [weight_mask_rate],
                "use_weight_rescale": use_weight_rescale,
                "mask_strategy": mask_strategy,
                "mask_apply_method": "ties_merging"
            }
            
            merger = MergingMethod("mask_merging")
            merged_model = merger.get_merged_model(**merge_args)
            
            # Clean up
            del models_to_merge[0]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    # Evaluate merged model
    clean_acc = model_cache.evaluate_model(merged_model, 'clean')
    victim_asr = model_cache.evaluate_model(merged_model, 'victim')
    
    # Clean up
    del merged_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    return clean_acc, victim_asr

def grid_search(args):
    logger = setup_logger()
    logger.info("Starting grid search...")
    
    # Initialize memory-efficient model cache
    model_cache = ModelCache(
        base_model_name=args.base_model,
        model_paths=args.model_paths,
        device=args.device,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size
    )
    
    # Load test datasets
    model_cache.load_test_datasets(args.clean_test, args.victim_test)
    
    # Convert space-separated strings to lists
    weight_mask_rates = [float(x) for x in args.weight_mask_rates]
    scaling_coefficients = [float(x) for x in args.scaling_coefficients]
    param_value_mask_rates = [float(x) for x in args.param_value_mask_rates]
    
    best_params = None
    best_cacc = -1
    best_victim_asr = -1
    
    # Create CSV file for results
    with open(args.results_log, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['weight_mask_rate', 'scaling_coefficient', 'param_value_mask_rate', 'clean_acc', f'{args.victim_name}_asr'])
        
        # Generate all combinations
        total_combinations = len(list(product(weight_mask_rates, scaling_coefficients, param_value_mask_rates)))
        current_combination = 0
        
        for wmr, sc, pvmr in product(weight_mask_rates, scaling_coefficients, param_value_mask_rates):
            current_combination += 1
            logger.info(f"\nTesting combination {current_combination}/{total_combinations}")
            logger.info(f"Parameters: wmr={wmr}, sc={sc}, pvmr={pvmr}")
            
            clean_acc, victim_asr = merge_models_with_params(
                model_cache=model_cache,
                weight_mask_rate=wmr,
                scaling_coefficient=sc,
                param_value_mask_rate=pvmr,
                mask_strategy=args.mask_strategy,
                use_weight_rescale=args.use_weight_rescale,
                logger=logger
            )
            
            # Write results to CSV
            csvwriter.writerow([wmr, sc, pvmr, clean_acc, victim_asr])
            csvfile.flush()  # Ensure results are written immediately
            
            logger.info(f"Results - Clean Acc: {clean_acc:.4f}, {args.victim_name} ASR: {victim_asr:.4f}")
            
            if clean_acc > best_cacc:
                best_cacc = clean_acc
                best_victim_asr = victim_asr
                best_params = (wmr, sc, pvmr)
    
    logger.info("\n=== Grid Search Results ===")
    logger.info(f"Best Parameters:")
    logger.info(f"  weight_mask_rate: {best_params[0]}")
    logger.info(f"  scaling_coefficient: {best_params[1]}")
    logger.info(f"  param_value_mask_rate: {best_params[2]}")
    logger.info(f"Best Clean Accuracy: {best_cacc:.4f}")
    logger.info(f"Corresponding {args.victim_name} ASR: {best_victim_asr:.4f}")
    
    return best_params, best_cacc, best_victim_asr

def parse_args():
    parser = argparse.ArgumentParser("Model Merging Interface")
    parser.add_argument("--base_model", type=str, default="roberta-large",
                      help="Name or path of the base model")
    parser.add_argument("--model_paths", nargs="+", required=True,
                      help="Paths to the models to be merged")
    parser.add_argument("--save_path", type=str, default="merged_models",
                      help="Path to save the merged model")
    parser.add_argument("--cache_dir", type=str, default=None,
                      help="Directory for caching pre-trained models")
    parser.add_argument("--mask_strategy", type=str, default="magnitude",
                      help="Strategy for masking")
    parser.add_argument("--use_weight_rescale", action="store_true", default=True,
                      help="whether to rescale the weight by 1 / (1 - weight_mask_rate)")
    parser.add_argument("--device", type=str, default="cuda",
                      help="Device to use (cuda or cpu)")
    parser.add_argument("--clean_test", type=str, required=True,
                      help="Path to clean test dataset")
    parser.add_argument("--victim_test", type=str, required=True,
                      help="Path to victim (poisoned) test dataset")
    parser.add_argument("--weight_mask_rates", nargs="+", default=["0.1", "0.2", "0.3"],
                      help="Weight mask rates to try")
    parser.add_argument("--scaling_coefficients", nargs="+", 
                      default=["0.1", "0.3", "0.5", "0.7", "0.9", "1.0"],
                      help="Scaling coefficients to try")
    parser.add_argument("--param_value_mask_rates", nargs="+", default=["0.7", "0.8", "0.9"],
                      help="Parameter value mask rates to try")
    parser.add_argument("--results_log", type=str, default="grid_search_results.csv",
                      help="Path to save grid search results")
    parser.add_argument("--batch_size", type=int, default=32,
                      help="Batch size for evaluation")
    parser.add_argument("--victim_name", type=str, default="badnet",
                      help="Name of victim attack type for logging")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    logger = setup_logger()
    best_params, best_cacc, best_victim_asr = grid_search(args)
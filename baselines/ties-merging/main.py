import os
import torch
import argparse
import numpy as np
from typing import List, Dict, Any
from datasets import Dataset
import json
import pandas as pd

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, set_seed

# Import TIES merging functions from local module
from ties_merging import state_dict_to_vector, vector_to_state_dict, ties_merging

def parse_args():
    parser = argparse.ArgumentParser(description='TIES Merging for Backdoor Defense')
    
    # Required arguments
    parser.add_argument('--task', type=str, required=True,
                        choices=['sst2', 'mnli', 'agnews', 'olid'],
                        help='Task/dataset name')
    parser.add_argument('--base_model', type=str, required=True,
                        help='Base model to use')
    parser.add_argument('--num_labels', type=int, required=True,
                        help='Number of labels for classification')
    parser.add_argument('--model_paths', nargs='+', required=True,
                        help='Paths to the model checkpoints to be merged')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to save the merged model')
    parser.add_argument('--clean_test', type=str, required=True,
                        help='Path to clean test dataset')
    parser.add_argument('--poison_test', type=str, required=True,
                        help='Path to poisoned test dataset')
    
    # Optional arguments
    parser.add_argument('--cache_dir', type=str, default=None,
                        help='Directory to cache the pretrained models')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--k', type=int, default=70,
                        help='Percentage of parameters to keep for TIES (default: 70)')
    
    return parser.parse_args()

def load_json_dataset(file_path: str) -> Dataset:
    """Load dataset from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    df = pd.DataFrame(data)
    return Dataset.from_pandas(df)

def create_classifier(model: Any, tokenizer: Any) -> Any:
    """Create classification pipeline."""
    return pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
        batch_size=128,
        max_length=128,
        truncation=True,
        top_k=None,
    )

def evaluate_model(model: Any, tokenizer: Any, test_dataset: Dataset) -> float:
    """Evaluate model performance."""
    classifier = create_classifier(model, tokenizer)
    results = classifier(test_dataset['sentence'])
    
    logits = []
    for result in results:
        score_dict = {item['label']: item['score'] for item in result}
        num_labels = len(score_dict)
        ordered_scores = [score_dict[f'LABEL_{i}'] for i in range(num_labels)]
        logits.append(ordered_scores)
    
    logits = np.array(logits)
    labels = np.array(test_dataset['label'])
    predictions = np.argmax(logits, axis=-1)
    
    return (predictions == labels).astype(np.float32).mean().item()

def merge_and_evaluate_models(args):
    set_seed(args.seed)
    
    # Load models and tokenizer
    print(f"Loading {len(args.model_paths)} models for TIES merging...")
    models = []
    state_dicts = []
    for model_path in args.model_paths:
        print(f"Loading model from {model_path}")
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        models.append(model)
        state_dicts.append(model.state_dict())
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, cache_dir=args.cache_dir)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=args.num_labels,
        cache_dir=args.cache_dir
    )
    
    # TIES Merging
    print(f"Merging models using TIES (k={args.k})...")
    flat_ft = torch.vstack([state_dict_to_vector(check, []) for check in state_dicts])
    flat_ptm = state_dict_to_vector(base_model.state_dict(), [])
    tv_flat_checks = flat_ft - flat_ptm
    merged_tv = ties_merging(tv_flat_checks, reset_thresh=args.k, merge_func="dis-mean")
    merged_check = flat_ptm + merged_tv
    merged_state_dict = vector_to_state_dict(merged_check, base_model.state_dict(), [])
    
    # Load merged state dict and save model
    base_model.load_state_dict(merged_state_dict)
    print(f"Saving TIES merged model to {args.output_path}")
    os.makedirs(args.output_path, exist_ok=True)
    base_model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)
    
    # Evaluate individual models
    print("\nEvaluating individual models:")
    test_clean = load_json_dataset(args.clean_test)
    test_poison = load_json_dataset(args.poison_test)
    
    for model, path in zip(models, args.model_paths):
        clean_acc = evaluate_model(model, tokenizer, test_clean)
        poison_asr = evaluate_model(model, tokenizer, test_poison)
        print(f"\nModel: {path}")
        print(f"Clean Accuracy: {clean_acc:.4f}")
        print(f"Poison ASR: {poison_asr:.4f}")
    
    # Evaluate merged model
    print("\nEvaluating TIES merged model:")
    merged_clean_acc = evaluate_model(base_model, tokenizer, test_clean)
    merged_poison_asr = evaluate_model(base_model, tokenizer, test_poison)
    print(f"TIES Merged Model (k={args.k}):")
    print(f"Clean Accuracy: {merged_clean_acc:.4f}")
    print(f"Poison ASR: {merged_poison_asr:.4f}")

def main():
    args = parse_args()
    merge_and_evaluate_models(args)

if __name__ == "__main__":
    main()
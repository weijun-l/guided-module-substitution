"""
PURE (Pre-trained Unlearning for Removing Evil) Defense Implementation

This script supports two modes:

1. Official PURE Pipeline (--mode official):
   - Uses the original PURE implementation from the paper
   - Trains poisoned models from scratch
   - Only supports BERT models
   - Follows original paper methodology exactly

2. Adapted PURE Pipeline (--mode adapted):
   - Our adapted implementation for flexibility
   - Uses pre-trained poisoned models (from our training pipeline)
   - Supports both BERT and RoBERTa models
"""

import argparse
import os
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from config import get_arguments
from util import load_json_dataset, set_seed

def run_evaluation(args):
    """Run evaluation on the final model"""
    print("\nStep 4: Running final evaluation...")
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load final model and tokenizer
    final_model_dir = os.path.join(args.output_dir, "final_model")
    if not os.path.exists(final_model_dir):
        print("No final model found for evaluation")
        return
        
    model = AutoModelForSequenceClassification.from_pretrained(final_model_dir).to(device)
    tokenizer = AutoTokenizer.from_pretrained(final_model_dir)
    model.eval()
    
    # Load test datasets
    clean_test = load_json_dataset(args.test_clean)
    poisoned_test = load_json_dataset(args.test_poison)
    
    print(f"\nDataset sizes:")
    print(f"Clean test set: {len(clean_test)}")
    print(f"Poisoned test set: {len(poisoned_test)}")
    
    # Setup pipeline
    res_pipeline = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
        batch_size=128,
        max_length=args.max_len_short,
        truncation=True,
        top_k=None 
    )
    
    # Get predictions
    print("\nRunning predictions...")
    clean_sentences = clean_test['sentence']
    poisoned_sentences = poisoned_test['sentence']
    
    results_clean = res_pipeline(clean_sentences)
    results_poisoned = res_pipeline(poisoned_sentences)

    # Process results
    def extract_logits(results):
        """Extract logits in correct label order."""
        logits = []
        for result in results:
            score_dict = {item['label']: item['score'] for item in result}
            num_labels = len(score_dict)
            ordered_scores = [score_dict[f'LABEL_{i}'] for i in range(num_labels)]
            logits.append(ordered_scores)
        return np.array(logits)
    
    # Process results
    clean_logits = extract_logits(results_clean)
    poison_logits = extract_logits(results_poisoned)
    
    clean_references = np.array(clean_test['label'])
    poisoned_references = np.array(poisoned_test['label'])
    
    # Compute and print metrics
    print("\nFinal Evaluation Results:")
    
    for name, logits, refs in [
        ("Clean", clean_logits, clean_references),
        ("Poisoned", poison_logits, poisoned_references)
    ]:
        predictions = np.argmax(logits, axis=-1)
        accuracy = (predictions == refs).mean()
        
        print(f"\n{name} Test Set Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Predictions shape: {predictions.shape}")
        print(f"Unique predictions: {np.unique(predictions)}")
        print(f"Unique labels: {np.unique(refs)}")

def run_official_pipeline(args):
    """
    Run the official PURE pipeline: trains poisoned model, then applies head pruning and normalization (BERT only)
    
    This pipeline:
    1. Trains poisoned models from scratch (following original paper)
    2. Applies original attention head pruning algorithm
    3. Applies original attention normalization
    4. Support BERT models as per original implementation
    """
    print("Step 1: Training poisoned model (official PURE method)...")
    from pretrained_model_poisoning import main as poison_main
    poison_main(args)
    
    print("\nStep 2: Official attention head pruning...")
    from attention_head_pruning import main as prune_main
    prune_main(args)
    
    print("\nStep 3: Official attention normalization...")
    from attention_normalization import main as norm_main
    norm_main(args)

def run_adapted_pipeline(args):
    """
    Run adapted PURE pipeline: uses pre-trained poisoned models, supports BERT/RoBERTa
    
    This pipeline:
    1. Loads pre-trained poisoned models (from our training experiments)
    2. Applies adapted attention head pruning (works with BERT/RoBERTa)
    3. Applies adapted attention normalization
    """
    print("Step 1: Loading pre-trained poisoned model (adapted method)...")
    model = AutoModelForSequenceClassification.from_pretrained(args.victim_path, attn_implementation="eager")
    tokenizer = AutoTokenizer.from_pretrained(args.victim_path)
    
    print("\nStep 2: Adapted attention head pruning...")
    from adapted_attention_head_pruning import main as adapted_prune_main
    adapted_prune_main(args, model, tokenizer)
    
    # Clean up GPU memory between steps
    if hasattr(model, 'cuda'):
        model.cpu()
    del model
    torch.cuda.empty_cache()
    
    print("\nStep 3: Adapted attention normalization...")
    model = AutoModelForSequenceClassification.from_pretrained(args.victim_path, attn_implementation="eager")
    from adapted_attention_normalization import main as adapted_norm_main
    adapted_norm_main(args, model, tokenizer)

def main():
    parser = get_arguments()
    
    # Add custom arguments
    parser.add_argument("--mode", type=str, choices=["official", "adapted"], required=True,
                      help="Run official pipeline (trains models, BERT only) or adapted pipeline (uses pre-trained models, BERT/RoBERTa)")
    parser.add_argument("--victim_path", type=str, default="./ckpts/roberta-large_42/sst2/train_badnet/",
                      help="Path to pre-trained poisoned model")
    parser.add_argument("--train_clean", type=str, default="./datasets/sst2/train_clean.json",
                      help="Path to clean training data")
    parser.add_argument("--test_clean", type=str, default="./datasets/sst2/test_clean.json",
                      help="Path to clean test data")
    parser.add_argument("--test_poison", type=str, default="./datasets/sst2/test_badnet.json",
                      help="Path to poisoned test data")
    parser.add_argument("--output_dir", type=str, default="./logs/pure/roberta-large_42_sst2_badnet/",
                      help="Directory for saving outputs")
    parser.add_argument("--acc_threshold", type=float, default=0.85)
    
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Set random seed
    set_seed(args.seed)
    
    # Run pipeline
    if args.mode == "official":
        run_official_pipeline(args)
    else:
        run_adapted_pipeline(args)
    
    # Run final evaluation
    run_evaluation(args)

if __name__ == "__main__":
    main()
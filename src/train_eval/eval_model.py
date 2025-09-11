#!/usr/bin/env python3
"""
Isolated model evaluation script for both encoder and decoder models.
"""

import argparse
import json
import logging
import os
import sys
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import PeftConfig, PeftModel

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _evaluate_decoder_model(model, tokenizer, test_dataset, batch_size=8):
    """Internal decoder evaluation function."""
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

def get_num_labels(dataset_name: str) -> int:
    """Get number of labels for dataset"""
    dataset_labels = {
        'sst2': 2, 'olid': 2, 'agnews': 4, 'qnli': 2, 'mnli': 3
    }
    return dataset_labels.get(dataset_name, 2)


def detect_model_type(model_path: str) -> str:
    """Detect if model is encoder or decoder based on path and config"""
    
    # Check for LoRA adapter (decoder models use LoRA)
    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        return "decoder"
    
    # Check for standard model config
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        try:
            config = AutoConfig.from_pretrained(model_path)
            # Decoder models typically have these architectures
            if hasattr(config, 'model_type') and config.model_type in ['llama', 'mistral', 'qwen', 'gpt', 'opt']:
                return "decoder"
            # Encoder models
            elif hasattr(config, 'model_type') and config.model_type in ['roberta', 'bert', 'distilbert']:
                return "encoder"
        except Exception as e:
            logger.warning(f"Could not determine model type from config: {e}")
    
    # Guess from path
    if any(name in model_path.lower() for name in ['llama', 'mistral', 'qwen', 'gpt']):
        return "decoder"
    elif any(name in model_path.lower() for name in ['roberta', 'bert']):
        return "encoder"
    
    # Default to encoder
    logger.warning(f"Could not determine model type for {model_path}, defaulting to encoder")
    return "encoder"


def evaluate_encoder_model(model_path: str, test_file: str, dataset_name: str, batch_size: int = 128) -> Dict[str, Any]:
    """Evaluate encoder model using pipeline approach"""
    
    logger.info("Loading encoder model for evaluation...")
    logger.info(f"Model path: {model_path}")
    
    # Load test data
    with open(test_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    df = pd.DataFrame(data)
    test_dataset = Dataset.from_pandas(df)
    
    # Set up pipeline
    eval_pipeline = pipeline(
        "text-classification",
        model=model_path,
        device=0 if torch.cuda.is_available() else -1,
        batch_size=batch_size,
        max_length=128,
        truncation=True,
        top_k=None
    )
    
    # Get predictions
    sentences = test_dataset['sentence']
    results = eval_pipeline(sentences)
    
    # Process results
    logits = []
    for result in results:
        score_dict = {item['label']: item['score'] for item in result}
        num_labels = len(score_dict)
        ordered_scores = [score_dict[f'LABEL_{i}'] for i in range(num_labels)]
        logits.append(ordered_scores)

    logits = np.array(logits)
    labels = np.array(test_dataset['label'])
    predictions = np.argmax(logits, axis=-1)

    accuracy = (predictions == labels).astype(np.float32).mean().item()
    
    return {
        "accuracy": accuracy,
        "num_samples": len(labels),
        "model_type": "encoder"
    }


def disable_attention_cache(model):
    """Disable attention caching and force use of standard attention"""
    if hasattr(model, 'base_model'):
        if hasattr(model.base_model, 'model') and hasattr(model.base_model.model, 'model'):
            model_layers = model.base_model.model.model.layers
        elif hasattr(model.base_model, 'model'):
            model_layers = model.base_model.model.layers
        else:
            model_layers = getattr(model.base_model, 'layers', [])
    else:
        model_layers = getattr(model.model, 'layers', [])
    
    # Disable at config level
    if hasattr(model, 'config'):
        model.config.use_cache = False
        if hasattr(model.config, 'use_sdpa'):
            model.config.use_sdpa = False
        if hasattr(model.config, 'use_flash_attention_2'):
            model.config.use_flash_attention_2 = False
    
    # Disable at layer level
    for layer in model_layers:
        if hasattr(layer, 'self_attn'):
            if hasattr(layer.self_attn, 'use_cache'):
                layer.self_attn.use_cache = False
            if hasattr(layer.self_attn, '_use_sdpa'):
                layer.self_attn._use_sdpa = False


def evaluate_decoder_model(model_path: str, test_file: str, dataset_name: str) -> Dict[str, Any]:
    """Evaluate decoder model using sequence classification approach like train_eval_decoder.py"""
    
    logger.info("Loading decoder model for evaluation...")
    logger.info(f"Model path: {model_path}")
    
    # Get number of labels for the dataset
    num_labels = get_num_labels(dataset_name)
    
    # Check if this is a PEFT checkpoint
    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    
    if os.path.exists(adapter_config_path):
        # This is a PEFT checkpoint directory - load base model then apply LoRA
        logger.info("Loading trained LoRA model...")
        config = PeftConfig.from_pretrained(model_path)
        logger.info(f"Base model: {config.base_model_name_or_path}")
        
        # Create base config with attention optimizations disabled
        base_config = AutoConfig.from_pretrained(config.base_model_name_or_path)
        base_config.use_cache = False
        base_config.num_labels = num_labels
        base_config.use_flash_attention_2 = False
        if hasattr(base_config, 'use_sdpa'):
            base_config.use_sdpa = False
        
        # Load base model for sequence classification
        model = AutoModelForSequenceClassification.from_pretrained(
            config.base_model_name_or_path,
            config=base_config,
            torch_dtype=torch.bfloat16
        )
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
        
        # Apply LoRA adapters
        model = PeftModel.from_pretrained(model, model_path)
        
        # Disable attention cache to prevent hanging
        disable_attention_cache(model)
        model.eval()
        
        logger.info("LoRA adapters loaded successfully")
    else:
        # This appears to be a base model path, not a trained checkpoint
        logger.warning("No trained checkpoint found (no adapter_config.json)")
        logger.warning("Expected: Path to trained LoRA checkpoint directory")
        logger.warning("This usually means training hasn't completed yet")
        raise ValueError(f"No trained LoRA checkpoint found at {model_path}")
    
    # Handle padding token
    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
    
    # Ensure model config also has the padding token
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # Move model to device
    if torch.cuda.is_available():
        model = model.to("cuda")
    
    # Load test data
    with open(test_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    df = pd.DataFrame(data)
    test_dataset = Dataset.from_pandas(df)
    
    logger.info(f"Evaluating on {len(data)} samples using unified decoder evaluation...")
    
    # Use internal evaluation function
    accuracy = _evaluate_decoder_model(model, tokenizer, test_dataset)
    
    return {
        "accuracy": accuracy,
        "num_samples": len(data),
        "model_type": "decoder"
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models on test datasets")
    
    # Required parameters (paths should be provided by shell script)
    parser.add_argument("--model_path", type=str, required=True,
                       help="Full path to trained model")
    parser.add_argument("--test_file", type=str, required=True,
                       help="Full path to test file")
    parser.add_argument("--log_file", type=str, required=True,
                       help="Full path to log file")
    
    # Dataset parameters  
    parser.add_argument("--dataset_name", type=str, required=True,
                       help="Dataset name (for num_labels detection)")
    
    # Auto-detection
    parser.add_argument("--model_type", type=str, choices=["encoder", "decoder", "auto"], 
                       default="auto", help="Model type (auto-detect if not specified)")
    
    # Evaluation parameters
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for evaluation")
    
    args = parser.parse_args()
    
    # Set up logging
    log_handler = logging.FileHandler(args.log_file)
    log_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    log_handler.setFormatter(formatter)
    logger.addHandler(log_handler)
    
    # Start evaluation
    logger.info("=" * 80)
    logger.info("ISOLATED MODEL EVALUATION")
    logger.info("=" * 80)
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Test file: {args.test_file}")
    logger.info(f"Dataset: {args.dataset_name}")
    logger.info(f"Log file: {args.log_file}")
    
    # Check if files exist
    if not os.path.exists(args.model_path):
        logger.error(f"Model path does not exist: {args.model_path}")
        sys.exit(1)
        
    if not os.path.exists(args.test_file):
        logger.error(f"Test file does not exist: {args.test_file}")
        sys.exit(1)
    
    # Auto-detect model type if needed
    if args.model_type == "auto":
        args.model_type = detect_model_type(args.model_path)
    
    logger.info(f"Model type: {args.model_type}")
    
    # Evaluate based on model type
    try:
        if args.model_type == "encoder":
            results = evaluate_encoder_model(args.model_path, args.test_file, args.dataset_name, args.batch_size)
        else:  # decoder
            results = evaluate_decoder_model(args.model_path, args.test_file, args.dataset_name)
            
        # Log results
        logger.info("=" * 80)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 80)
        logger.info(f"Model Type: {results['model_type']}")
        logger.info(f"Accuracy: {results['accuracy']:.4f}")
        logger.info(f"Samples: {results['num_samples']}")
        logger.info("=" * 80)
        
        # Save results to JSON
        results_file = args.log_file.replace('.log', '_results.json')
        results_data = {
            "model_path": args.model_path,
            "test_file": args.test_file,
            "dataset_name": args.dataset_name,
            "model_type": results['model_type'],
            "accuracy": results['accuracy'],
            "num_samples": results['num_samples']
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Results saved to: {results_file}")
        
        print(f"Evaluation completed! Accuracy: {results['accuracy']:.4f}")
        print(f"Log saved to: {args.log_file}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()
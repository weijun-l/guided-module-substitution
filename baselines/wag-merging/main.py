#!/usr/bin/env python3
"""
Unified WAG (Weight Averaging) Merging for both Encoder and Decoder Models
Supports: BERT, RoBERTa (encoder) and Llama, Qwen, Mistral (decoder with LoRA)
"""

import os
import torch
import argparse
import numpy as np
from typing import List, Any
from datasets import Dataset
import json
import pandas as pd
import gc
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, pipeline, set_seed
from peft import PeftConfig, PeftModel

def parse_args():
    parser = argparse.ArgumentParser(description='Unified WAG Merging for Encoder/Decoder Models')
    
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
    parser.add_argument('--model_type', type=str, default='auto',
                        choices=['encoder', 'decoder', 'auto'],
                        help='Model type (auto-detect by default)')
    
    return parser.parse_args()

def cuda_cleanup():
    """Clean up cuda memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

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
            print(f"Warning: Could not determine model type from config: {e}")
    
    # Guess from path
    if any(name in model_path.lower() for name in ['llama', 'mistral', 'qwen', 'gpt']):
        return "decoder"
    elif any(name in model_path.lower() for name in ['roberta', 'bert']):
        return "encoder"
    
    # Default to encoder
    print(f"Warning: Could not determine model type for {model_path}, defaulting to encoder")
    return "encoder"

def disable_attention_cache(model):
    """Disable attention caching and force use of standard attention"""
    if hasattr(model, 'base_model'):
        model_layers = model.base_model.model.model.layers
    else:
        model_layers = model.model.layers
    
    # Disable at config level
    if hasattr(model, 'config'):
        model.config.use_cache = False
        if hasattr(model.config, 'use_sdpa'):
            model.config.use_sdpa = False
        if hasattr(model.config, 'use_flash_attention_2'):
            model.config.use_flash_attention_2 = False
    
    # Disable at layer level
    for layer in model_layers:
        if hasattr(layer.self_attn, 'use_cache'):
            layer.self_attn.use_cache = False
        if hasattr(layer.self_attn, '_use_sdpa'):
            layer.self_attn._use_sdpa = False

def load_json_dataset(file_path: str) -> Dataset:
    """Load dataset from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    df = pd.DataFrame(data)
    return Dataset.from_pandas(df)

def evaluate_encoder_model(model: Any, tokenizer: Any, test_dataset: Dataset, batch_size: int = 128) -> float:
    """Evaluate encoder model using pipeline approach"""
    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
        batch_size=batch_size,
        max_length=128,
        truncation=True,
        top_k=None
    )
    
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


def evaluate_decoder_model(model: Any, tokenizer: Any, test_dataset: Dataset, 
                          batch_size: int = 8) -> float:
    """
    Evaluate decoder model performance.
    """
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
            # Fallback for models without eos_token
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


def load_and_evaluate_single_decoder_model(model_path: str, base_model_path: str, num_labels: int, 
                                         tokenizer: Any, test_clean: Dataset, test_poison: Dataset):
    """Memory-efficient: Load model, extract LoRA weights, evaluate, and immediately cleanup"""
    print(f"Loading and evaluating model from {model_path}")
    
    # Load base model
    base_config = AutoConfig.from_pretrained(base_model_path)
    base_config.use_cache = False
    base_config.num_labels = num_labels
    base_config.use_flash_attention_2 = False
    if hasattr(base_config, 'use_sdpa'):
        base_config.use_sdpa = False
    base_config.pad_token_id = tokenizer.pad_token_id

    current_base = AutoModelForSequenceClassification.from_pretrained(
        base_model_path,
        config=base_config,
        torch_dtype=torch.bfloat16
    )
    current_base.config.pad_token_id = tokenizer.pad_token_id
    
    # Apply LoRA
    model = PeftModel.from_pretrained(current_base, model_path)
    disable_attention_cache(model)
    model.eval()
    
    if torch.cuda.is_available():
        model = model.to("cuda")
    
    # Evaluate immediately
    clean_acc = evaluate_decoder_model(model, tokenizer, test_clean)
    poison_asr = evaluate_decoder_model(model, tokenizer, test_poison)
    
    # Extract LoRA weights to CPU
    lora_weights = {}
    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            lora_weights[name] = param.data.clone().cpu()  # Move to CPU to save GPU memory
    
    # Immediate cleanup
    del model, current_base
    cuda_cleanup()
    
    return clean_acc, poison_asr, lora_weights


def wag_merge_decoder_models(model_paths: List[str], base_model_path: str, num_labels: int,
                           tokenizer: Any, test_clean: Dataset, test_poison: Dataset):
    """
    WAG merge decoder models using memory-efficient approach:
    1. Process models one by one, extracting LoRA weights and evaluating
    2. Average LoRA weights on CPU
    3. Apply to final model
    """
    print("Loading models for WAG merging...")
    
    all_lora_weights = []
    
    # Process each model individually to save memory
    print(f"\nEvaluating {len(model_paths)} individual models:")
    for i, model_path in enumerate(model_paths):
        clean_acc, poison_asr, lora_weights = load_and_evaluate_single_decoder_model(
            model_path, base_model_path, num_labels, tokenizer, test_clean, test_poison
        )
        
        print(f"Model {i+1} ({model_path}): Clean Acc: {clean_acc:.4f}, Poison ASR: {poison_asr:.4f}")
        all_lora_weights.append(lora_weights)
    
    # Average LoRA weights on CPU (memory efficient)
    print(f"Performing WAG (weight averaging) merging...")
    averaged_lora_weights = {}
    
    # Get all LoRA parameter names from first model
    lora_param_names = list(all_lora_weights[0].keys())
    print(f"Averaging {len(lora_param_names)} LoRA parameters across {len(model_paths)} models")
    
    for param_name in lora_param_names:
        # Collect weights for this parameter from all models
        param_tensors = []
        for lora_weights in all_lora_weights:
            if param_name in lora_weights:
                param_tensors.append(lora_weights[param_name])
        
        if param_tensors:
            # Average on CPU
            averaged_lora_weights[param_name] = torch.stack(param_tensors).mean(dim=0)
    
    # Clean up individual LoRA weights
    del all_lora_weights
    cuda_cleanup()
    
    # Create final merged model
    print("Creating final merged model...")
    base_config = AutoConfig.from_pretrained(base_model_path)
    base_config.use_cache = False
    base_config.num_labels = num_labels
    base_config.use_flash_attention_2 = False
    if hasattr(base_config, 'use_sdpa'):
        base_config.use_sdpa = False
    base_config.pad_token_id = tokenizer.pad_token_id

    merged_base = AutoModelForSequenceClassification.from_pretrained(
        base_model_path,
        config=base_config,
        torch_dtype=torch.bfloat16
    )
    merged_base.config.pad_token_id = tokenizer.pad_token_id
    
    # Apply first LoRA to get the structure
    merged_model = PeftModel.from_pretrained(merged_base, model_paths[0])
    disable_attention_cache(merged_model)
    merged_model.eval()
    
    if torch.cuda.is_available():
        merged_model = merged_model.to("cuda")
    
    # Apply averaged LoRA weights
    with torch.no_grad():
        for name, param in merged_model.named_parameters():
            if name in averaged_lora_weights:
                # Move averaged weight to GPU and apply
                averaged_weight = averaged_lora_weights[name].to(param.device, param.dtype)
                param.data.copy_(averaged_weight)
    
    # Clean up averaged weights
    del averaged_lora_weights
    cuda_cleanup()
    
    print("WAG merging completed")
    return merged_model

def wag_merge_encoder_models(model_paths: List[str], base_model: str, num_labels: int, cache_dir: str = None):
    """WAG merge encoder models"""
    print(f"Loading {len(model_paths)} encoder models for WAG merging...")
    models = []
    state_dicts = []
    
    for model_path in model_paths:
        print(f"Loading encoder model from {model_path}")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=num_labels,
            cache_dir=cache_dir
        )
        models.append(model)
        state_dicts.append(model.state_dict())
    
    # Load base model for merging
    base_model_instance = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=num_labels,
        cache_dir=cache_dir
    )
    
    # Perform WAG merging
    print("Performing WAG merging...")
    if not state_dicts:
        raise ValueError("No models to merge")
    
    # Find common keys
    common_keys = set(state_dicts[0].keys())
    for state_dict in state_dicts[1:]:
        common_keys = common_keys.intersection(set(state_dict.keys()))
    
    print(f"Merging {len(common_keys)} common parameters across {len(state_dicts)} models")
    
    # Average parameters
    merged_state_dict = {}
    for key in common_keys:
        tensors = [state_dict[key].float() for state_dict in state_dicts]
        merged_state_dict[key] = torch.stack(tensors).mean(dim=0)
    
    # Load merged weights into base model
    base_model_instance.load_state_dict(merged_state_dict)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, cache_dir=cache_dir)
    
    return base_model_instance, tokenizer, models

def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Detect model type from first model path
    if args.model_type == 'auto':
        model_type = detect_model_type(args.model_paths[0])
        print(f"Auto-detected model type: {model_type}")
    else:
        model_type = args.model_type
        print(f"Using specified model type: {model_type}")
    
    # Load test datasets
    test_clean = load_json_dataset(args.clean_test)
    test_poison = load_json_dataset(args.poison_test)
    
    # Merge models based on type
    if model_type == 'encoder':
        merged_model, tokenizer, individual_models = wag_merge_encoder_models(
            args.model_paths, args.base_model, args.num_labels, args.cache_dir
        )
        
        # Evaluate individual models
        print(f"\nEvaluating {len(individual_models)} individual models:")
        for i, (model, path) in enumerate(zip(individual_models, args.model_paths)):
            clean_acc = evaluate_encoder_model(model, tokenizer, test_clean)
            poison_asr = evaluate_encoder_model(model, tokenizer, test_poison)
            print(f"Model {i+1} ({path}): Clean Acc: {clean_acc:.4f}, Poison ASR: {poison_asr:.4f}")
            
    elif model_type == 'decoder':
        # Get actual base model path from PEFT config
        first_config = PeftConfig.from_pretrained(args.model_paths[0])
        actual_base_model = first_config.base_model_name_or_path
        print(f"Using actual base model: {actual_base_model}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(actual_base_model)
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # WAG merging (includes individual model evaluation)
        merged_model = wag_merge_decoder_models(
            args.model_paths, actual_base_model, args.num_labels, tokenizer, test_clean, test_poison
        )
        
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Evaluate merged model
    print("\nEvaluating WAG merged model:")
    if model_type == 'encoder':
        merged_clean_acc = evaluate_encoder_model(merged_model, tokenizer, test_clean)
        merged_poison_asr = evaluate_encoder_model(merged_model, tokenizer, test_poison)
    else:  # decoder
        merged_clean_acc = evaluate_decoder_model(merged_model, tokenizer, test_clean)
        merged_poison_asr = evaluate_decoder_model(merged_model, tokenizer, test_poison)
    
    print(f"WAG Merged Model ({model_type}):")
    print(f"Clean Accuracy: {merged_clean_acc:.4f}")
    print(f"Poison ASR: {merged_poison_asr:.4f}")
    
    # Save merged model
    print(f"Saving merged model to {args.output_path}")
    os.makedirs(args.output_path, exist_ok=True)
    merged_model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)

if __name__ == "__main__":
    main()
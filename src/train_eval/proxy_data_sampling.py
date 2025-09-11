#!/usr/bin/env python3
"""Proxy data sampling for backdoor defense using SEEP logits analysis."""

import argparse
import json
import os
import random
from typing import List, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F


def load_training_data(file_path: str) -> List[Dict[str, Any]]:
    """Load training dataset."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f]


def save_samples(samples: List[Dict[str, Any]], output_path: str) -> None:
    """Save samples to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            json.dump(sample, f, ensure_ascii=False)
            f.write('\n')
    print(f"Saved {len(samples)} samples to: {output_path}")


def calculate_suspiciousness(logits: torch.Tensor, labels: torch.Tensor, method: str = "inv") -> torch.Tensor:
    """Calculate suspiciousness scores from logits across epochs using various SEEP methods."""
    probs = torch.softmax(logits, dim=-1)
    nums = len(labels)
    rows = torch.arange(nums)
    epochs = probs.size(0)
    
    # All SEEP scoring methods from original implementation
    if method == "inv":
        # Inverse confidence - lower confidence = more suspicious
        confs = []
        for i in range(epochs):
            pred_probs = probs[i][rows, labels]
            confs.append(1.0 / (1.0 - pred_probs + 1e-8))
        return torch.stack(confs, dim=-1)
    
    elif method == "conf":
        # Direct confidence - higher confidence = less suspicious  
        confs = []
        for i in range(epochs):
            confs.append(probs[i][rows, labels])
        return torch.stack(confs, dim=-1)
    
    elif method == "chi":
        # Chi-squared distance
        confs = []
        for i in range(epochs):
            pred_probs = probs[i][rows, labels]
            confs.append((1.0 - pred_probs) / (pred_probs + 1e-8))
        return torch.stack(confs, dim=-1)
    
    elif method == "hd":
        # Hellinger distance
        confs = []
        for i in range(epochs):
            pred_probs = probs[i][rows, labels]
            confs.append(2.0 - 2.0 * torch.sqrt(pred_probs + 1e-8))
        return torch.stack(confs, dim=-1)
    
    elif method == "kl":
        # KL divergence (negative log likelihood)
        confs = []
        for i in range(epochs):
            confs.append(-torch.log(probs[i][rows, labels] + 1e-8))
        return torch.stack(confs, dim=-1)
    
    elif method == "tvd":
        # Total variation distance
        confs = []
        for i in range(epochs):
            true_probs = torch.zeros_like(probs[i])
            true_probs[rows, labels] = 1.0
            confs.append(0.5 * torch.norm(true_probs - probs[i], p=1, dim=-1))
        return torch.stack(confs, dim=-1)
    
    elif method == "js":
        # Jensen-Shannon divergence
        confs = []
        for i in range(epochs):
            true_probs = torch.zeros_like(probs[i])
            true_probs[rows, labels] = 1.0
            inputs = torch.log((true_probs + probs[i]) / 2.0 + 1e-8)
            kl1 = F.kl_div(inputs, true_probs, reduction="none").sum(dim=-1)
            kl2 = F.kl_div(inputs, probs[i], reduction="none").sum(dim=-1)
            confs.append(kl1 + kl2)
        return torch.stack(confs, dim=-1)
    
    elif method == "lc":
        # Log-Cosh distance
        confs = []
        for i in range(epochs):
            true_probs = torch.zeros_like(probs[i])
            true_probs[rows, labels] = 1.0
            diff_squared = (true_probs - probs[i]) ** 2
            sum_probs = true_probs + probs[i] + 1e-8
            confs.append((diff_squared / sum_probs).sum(dim=-1) * 0.5)
        return torch.stack(confs, dim=-1)
    
    elif method == "renyi":
        # Rényi divergence (alpha=0.1)
        alpha = 0.1
        confs = []
        for i in range(epochs):
            renyi_score = torch.log((probs[i] ** alpha).sum(dim=-1) + 1e-8) * (1.0 / (1.0 - alpha))
            confs.append(renyi_score)
        return torch.stack(confs, dim=-1)
    
    else:
        raise ValueError(f"Unknown suspiciousness method: {method}. "
                        f"Available methods: inv, conf, chi, hd, kl, tvd, js, lc, renyi")


def detect_suspicious_samples(train_file: str, logits_file: str, 
                            method: str = "inv", metric: str = "mean", 
                            top_k: int = 200) -> List[Dict[str, Any]]:
    """Detect most suspicious samples from training data."""
    # Load data
    samples = load_training_data(train_file)
    logits = torch.load(logits_file)
    labels = torch.tensor([sample["label"] for sample in samples])
    
    # Calculate suspiciousness scores
    scores = calculate_suspiciousness(logits, labels, method)
    
    # Aggregate scores across epochs
    if metric == "mean":
        final_scores = scores.mean(dim=-1)
    elif metric == "std":
        final_scores = scores.std(dim=-1)
    else:
        raise ValueError(f"Unknown aggregation metric: {metric}")
    
    # Sort samples by suspiciousness (higher score = more suspicious for "inv")
    sample_scores = list(zip(samples, final_scores.tolist()))
    if method == "inv":
        sample_scores.sort(key=lambda x: x[1], reverse=True)  # Higher inv score = more suspicious
    else:
        sample_scores.sort(key=lambda x: x[1])  # Lower conf score = more suspicious
    
    # Return top-k most suspicious samples
    return [sample for sample, _ in sample_scores[:top_k]]


def random_sample_data(train_file: str, top_k: int = 200, seed: int = 42) -> List[Dict[str, Any]]:
    """Randomly sample from training data."""
    random.seed(seed)
    np.random.seed(seed)
    
    samples = load_training_data(train_file)
    return random.sample(samples, min(top_k, len(samples)))


def main():
    parser = argparse.ArgumentParser(description="Generate proxy data samples from SEEP training")
    parser.add_argument("train_file", help="Path to training data JSON file")
    parser.add_argument("logits_file", help="Path to SEEP logits PT file")
    parser.add_argument("--method", choices=["inv", "conf", "chi", "hd", "kl", "tvd", "js", "lc", "renyi"], 
                       default="inv", help="Suspiciousness calculation method")
    parser.add_argument("--metric", choices=["mean", "std"], default="mean", 
                       help="Score aggregation across epochs")
    parser.add_argument("--top_k", type=int, default=200,
                       help="Number of samples to select")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--output_dir", help="Output directory (default: same as train_file)")
    
    args = parser.parse_args()
    
    # Set output directory - create proxy subdirectory
    if args.output_dir:
        output_dir = os.path.join(args.output_dir, "proxy")
    else:
        dataset_dir = os.path.dirname(args.train_file)
        output_dir = os.path.join(dataset_dir, "proxy")
    
    # Extract backdoor type from filename for output naming
    train_basename = os.path.basename(args.train_file)
    backdoor_type = train_basename.split('_')[1].split('.')[0] if '_' in train_basename else "unknown"
    
    # Detect suspicious samples
    print("Detecting suspicious samples...")
    suspicious_samples = detect_suspicious_samples(
        args.train_file, args.logits_file, args.method, args.metric, args.top_k
    )
    
    # Save suspicious samples
    suspicious_output = os.path.join(output_dir, f"suspicious_{backdoor_type}_{args.seed}.json")
    save_samples(suspicious_samples, suspicious_output)
    
    # Generate random samples for comparison
    print("Generating random samples...")
    random_samples = random_sample_data(args.train_file, args.top_k, args.seed)
    random_output = os.path.join(output_dir, f"random_{backdoor_type}_{args.seed}.json")
    save_samples(random_samples, random_output)
    
    print(f"Detection complete. Method: {args.method}, Metric: {args.metric}")


if __name__ == "__main__":
    main()
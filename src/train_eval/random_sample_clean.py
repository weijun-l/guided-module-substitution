import json
import pandas as pd
import argparse
import os
import random
from typing import List, Dict
import numpy as np

def load_json_dataset(file_path: str):
    """Load a JSON dataset from file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
        df = pd.DataFrame(data)
        return df
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        raise
    except json.JSONDecodeError:
        print(f"Invalid JSON in file: {file_path}")
        raise

def save_json_dataset(data: List[Dict], file_path: str):
    """Save data to JSON file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f)
            f.write('\n')

def random_sample(df: pd.DataFrame, top_k: int, seed: int = 42) -> pd.DataFrame:
    """Randomly sample top_k samples from dataframe."""
    random.seed(seed)
    np.random.seed(seed)
    return df.sample(n=min(top_k, len(df)), random_state=seed)

def main():
    parser = argparse.ArgumentParser(description='Randomly sample clean data from poison dataset')
    parser.add_argument('poison_file', type=str, help='Path to the poison dataset file')
    parser.add_argument('--top_k', type=int, default=200, help='Number of samples to select')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    # Load dataset
    df = load_json_dataset(args.poison_file)
    
    # Random sampling
    sampled_df = random_sample(df, args.top_k, args.seed)
    
    # Get backdoor type from filename
    backdoor_type = os.path.basename(args.poison_file).split('_')[1].split('.')[0]
    
    # Create output path
    output_dir = os.path.join(os.path.dirname(args.poison_file), f'pseudo_{args.seed}')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'pseudo_clean_{backdoor_type}.json')
    
    # Save sampled data
    save_json_dataset(sampled_df.to_dict('records'), output_file)
    print(f"Saved {len(sampled_df)} samples to {output_file}")

if __name__ == '__main__':
    main()
"""
Dataset loading utilities for decoder GMS purification
"""

import json
import pandas as pd
from datasets import Dataset
from typing import Any, List, Dict


def load_json_dataset(file_path: str) -> List[Dict[str, Any]]:
    """
    Load a JSON dataset from file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        List of dictionaries containing the dataset entries
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    try:
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    data.append(json.loads(line))
        
        print(f"Loaded {len(data)} samples from {file_path}")
        
        # Convert to the expected format for compatibility
        if data and isinstance(data[0], dict):
            # Return as list of dicts for decoder model compatibility
            return data
        
        return data
        
    except FileNotFoundError:
        print(f"ERROR: File not found: {file_path}")
        raise
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in file: {file_path} - {e}")
        raise
    except Exception as e:
        print(f"ERROR: Failed to load dataset from {file_path} - {e}")
        raise


def validate_dataset_format(dataset: List[Dict[str, Any]], dataset_name: str = "dataset") -> bool:
    """
    Validate that the dataset has the expected format.
    
    Args:
        dataset: The dataset to validate
        dataset_name: Name of the dataset for error messages
        
    Returns:
        True if valid, raises exception otherwise
    """
    if not isinstance(dataset, list):
        raise ValueError(f"{dataset_name} must be a list, got {type(dataset)}")
    
    if len(dataset) == 0:
        print(f"WARNING: {dataset_name} is empty")
        return True
    
    # Check first sample
    sample = dataset[0]
    if not isinstance(sample, dict):
        raise ValueError(f"{dataset_name} entries must be dictionaries, got {type(sample)}")
    
    # Check required fields
    required_fields = ['sentence', 'label']
    missing_fields = [field for field in required_fields if field not in sample]
    
    if missing_fields:
        raise ValueError(f"{dataset_name} entries missing required fields: {missing_fields}")
    
    print(f"{dataset_name} format validation passed")
    return True
"""
Dataset utilities for GMS encoder purification
"""

import json
import pandas as pd
from datasets import Dataset
from typing import Any


def load_json_dataset(file_path: str) -> Dataset:
    """
    Load a JSON dataset from file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
        df = pd.DataFrame(data)
        return Dataset.from_pandas(df)
    except FileNotFoundError:
        print(f"ERROR: Dataset file not found: {file_path}")
        raise
    except json.JSONDecodeError:
        print(f"ERROR: Invalid JSON in file: {file_path}")
        raise
    except Exception as e:
        print(f"ERROR: Failed to load dataset from {file_path}: {e}")
        raise
"""
Model utilities for GMS encoder purification
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from typing import Tuple, Any


def load_models(source_path: str, target_path: str) -> Tuple[Any, Any, Any]:
    """
    Load proxy and suspect victim models along with tokenizer.
    
    Args:
        source_path: Path to proxy model
        target_path: Path to suspect victim model
        
    Returns:
        Tuple of (tokenizer, proxy_model, suspect_victim_model)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading tokenizer from: {target_path}")
    tokenizer = AutoTokenizer.from_pretrained(target_path)
    
    print(f"Loading proxy model from: {source_path}")
    source_model = AutoModelForSequenceClassification.from_pretrained(source_path).to(device)
    
    print(f"Loading suspect victim model from: {target_path}")
    target_model = AutoModelForSequenceClassification.from_pretrained(target_path).to(device)

    return tokenizer, source_model, target_model


def create_classifier(model: Any, tokenizer: Any, batch_size: int = 128) -> Any:
    """
    Create a text classification pipeline for evaluation.
    
    Args:
        model: The model to use for classification
        tokenizer: Tokenizer for the model
        batch_size: Batch size for inference
        
    Returns:
        Classification pipeline
    """
    return pipeline(
        "text-classification",  
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
        batch_size=batch_size,
        max_length=128,
        truncation=True,
        top_k=None
    )

def extract_predictions_from_pipeline(results):
    """
    Extract predictions from pipeline results.
    
    Args:
        results: Output from text-classification pipeline
        
    Returns:
        List of prediction labels (integers)
    """
    predictions = []
    for result in results:
        if isinstance(result, list) and len(result) > 0:
            # Convert to score dictionary
            score_dict = {item['label']: item['score'] for item in result}
            num_labels = len(score_dict)
            # Get ordered scores: [LABEL_0_score, LABEL_1_score, ...]
            ordered_scores = [score_dict[f'LABEL_{i}'] for i in range(num_labels)]
            # Get prediction as argmax
            pred_label = int(np.argmax(ordered_scores))
            predictions.append(pred_label)
        else:
            # Handle single result format (top_k=1)
            pred_label = int(result['label'].split('_')[1])
            predictions.append(pred_label)
    return predictions

def get_model_type_and_layers(model: Any) -> Tuple[str, int]:
    """
    Determine model type and number of transformer layers.
    
    Args:
        model: Transformer model
        
    Returns:
        Tuple of (model_type, num_layers)
        
    Raises:
        ValueError: If model type is not supported
    """
    if hasattr(model, 'bert'):
        return 'bert', len(model.bert.encoder.layer)
    elif hasattr(model, 'roberta'):
        return 'roberta', len(model.roberta.encoder.layer)
    elif hasattr(model, 'distilbert'):
        return 'distilbert', len(model.distilbert.transformer.layer)
    else:
        # Try to infer from config
        config = model.config
        if hasattr(config, 'num_hidden_layers'):
            model_type = config.model_type if hasattr(config, 'model_type') else 'unknown'
            return model_type, config.num_hidden_layers
        else:
            raise ValueError(f"Unsupported model type. Model attributes: {list(model.__dict__.keys())}")


def get_module_path(model_type: str, layer_idx: int, module_name: str) -> str:
    """
    Get the full module path for a specific layer and module type.
    
    Args:
        model_type: Type of model (bert, roberta, distilbert)
        layer_idx: Layer index
        module_name: Module name from MODULE_MAPPING
        
    Returns:
        Full path to the module
    """
    from .config import MODULE_MAPPING
    
    base_module = MODULE_MAPPING[module_name]
    
    if model_type == 'bert':
        return f"bert.encoder.layer.{layer_idx}.{base_module}"
    elif model_type == 'roberta':
        return f"roberta.encoder.layer.{layer_idx}.{base_module}"
    elif model_type == 'distilbert':
        return f"distilbert.transformer.layer.{layer_idx}.{base_module}"
    else:
        # Generic fallback
        return f"encoder.layer.{layer_idx}.{base_module}"
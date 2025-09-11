"""
GMS (Guided Module Search) for Encoder Model Purification

This package implements the GMS method for purifying backdoored transformer models
by selectively replacing modules and layers from a proxy model.
"""

from .config import MODULE_MAPPING, DEFAULT_SEED, DEFAULT_ALPHA
from .dataset import load_json_dataset
from .model_utils import load_models, create_classifier, get_model_type_and_layers
from .purifier import ModelPurifier
from .strategy import find_best_strategy, guided_substitution_search

__version__ = "1.0.0"
__all__ = [
    "MODULE_MAPPING",
    "DEFAULT_SEED", 
    "DEFAULT_ALPHA",
    "load_json_dataset",
    "load_models",
    "create_classifier", 
    "get_model_type_and_layers",
    "ModelPurifier",
    "find_best_strategy",
    "guided_substitution_search",
]
"""
Model utilities for decoder GMS purification
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig
from typing import Tuple, Any, Dict


def get_num_labels(dataset_name: str) -> int:
    """
    Get number of labels for dataset.
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        Number of labels for the dataset
    """
    dataset_labels = {
        'sst2': 2, 'olid': 2, 'agnews': 4, 'qnli': 2, 'mnli': 3
    }
    return dataset_labels.get(dataset_name, 2)


def extract_dataset_from_path(path: str) -> str:
    """
    Extract dataset name from a model or data path.
    
    Args:
        path: Path containing dataset information
        
    Returns:
        Dataset name
    """
    # Try to extract dataset name from common path patterns
    # e.g., "./ckpts/Llama-2-7b-hf/sst2_42/train_badnet" -> "sst2"
    # e.g., "./datasets/sst2/test_clean.json" -> "sst2"
    
    path_lower = path.lower()
    for dataset in ['sst2', 'olid', 'agnews', 'qnli', 'mnli']:
        if dataset in path_lower:
            return dataset
    
    # Fallback: assume sst2
    return 'sst2'


def load_models(base_model_path: str, source_lora: str, target_lora: str) -> Tuple[Any, Any, Any]:
    """
    Load base model with source and target LoRA adaptors with extensive error checking.
    
    Args:
        base_model_path: Path to the base model
        source_lora: Path to the source LoRA weights
        target_lora: Path to the target LoRA weights
        
    Returns:
        Tuple of (tokenizer, source_model, target_model)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n=== Loading Decoder Models ===")
    print(f"Base Model: {base_model_path}")
    print(f"Source LoRA: {source_lora}")
    print(f"Target LoRA: {target_lora}")
    print(f"Device: {device}")
    
    # Load source and target LoRA configs
    print("\nLoading LoRA configurations...")
    source_config = PeftConfig.from_pretrained(source_lora)
    target_config = PeftConfig.from_pretrained(target_lora)
    
    print(f"Source LoRA config: r={source_config.r}, alpha={source_config.lora_alpha}")
    print(f"Target modules in source: {source_config.target_modules}")
    
    print(f"Target LoRA config: r={target_config.r}, alpha={target_config.lora_alpha}")
    print(f"Target modules in target: {target_config.target_modules}")
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    # Handle padding token for decoder models
    if getattr(tokenizer, "pad_token_id") is None:
        print("Setting pad token to eos token")
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
    
    # Determine number of labels dynamically from the LoRA paths
    dataset_name = extract_dataset_from_path(source_lora)
    num_labels = get_num_labels(dataset_name)
    print(f"Detected dataset: {dataset_name}")
    print(f"Setting up models with {num_labels} output classes")
    
    # Load base models with correct number of labels
    print("\nLoading base models...")
    source_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_path, 
        torch_dtype=torch.bfloat16,
        num_labels=num_labels,
        device_map="auto"
    )
    
    target_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_path, 
        torch_dtype=torch.bfloat16,
        num_labels=num_labels,
        device_map="auto"
    )
    
    # Ensure model configs also have the padding token
    source_model.config.pad_token_id = tokenizer.pad_token_id
    target_model.config.pad_token_id = tokenizer.pad_token_id
    
    print(f"Base models loaded successfully")
    
    # Load LoRA adaptations with error handling
    print("\nLoading LoRA adaptations...")
    try:
        print("Loading source LoRA adapter...")
        source_model = PeftModel.from_pretrained(source_model, source_lora)
        print("Source LoRA adapter loaded successfully")
    except Exception as e:
        print(f"Error loading source LoRA adapter: {e}")
        raise
        
    try:
        print("Loading target LoRA adapter...")
        target_model = PeftModel.from_pretrained(target_model, target_lora)
        print("Target LoRA adapter loaded successfully")
    except Exception as e:
        print(f"Error loading target LoRA adapter: {e}")
        raise
    
    # Print detailed LoRA information for debugging
    print("\n=== LoRA Details ===")
    
    # Check LoRA modules in target model
    print("\nChecking LoRA modules in target model:")
    target_lora_count = 0
    target_lora_modules = set()
    
    for name, module in target_model.named_modules():
        if hasattr(module, 'lora_A'):
            target_lora_count += 1
            # Extract module type from name
            parts = name.split('.')
            if len(parts) > 0:
                module_type = parts[-1]
                target_lora_modules.add(module_type)
            
            if target_lora_count <= 5:  # Just show first 5 modules
                print(f"  Found LoRA in {name}: A={module.lora_A['default'].weight.shape}, B={module.lora_B['default'].weight.shape}")
    
    print(f"Total LoRA modules in target model: {target_lora_count}")
    print(f"LoRA module types in target model: {target_lora_modules}")
    
    # Check LoRA modules in source model
    print("\nChecking LoRA modules in source model:")
    source_lora_count = 0
    source_lora_modules = set()
    
    for name, module in source_model.named_modules():
        if hasattr(module, 'lora_A'):
            source_lora_count += 1
            # Extract module type from name
            parts = name.split('.')
            if len(parts) > 0:
                module_type = parts[-1]
                source_lora_modules.add(module_type)
                
            if source_lora_count <= 5:  # Just show first 5 modules
                print(f"  Found LoRA in {name}: A={module.lora_A['default'].weight.shape}, B={module.lora_B['default'].weight.shape}")
    
    print(f"Total LoRA modules in source model: {source_lora_count}")
    print(f"LoRA module types in source model: {source_lora_modules}")
    
    # Find common LoRA modules between the two models
    common_modules = source_lora_modules.intersection(target_lora_modules)
    print(f"\nCommon LoRA module types between models: {common_modules}")
    
    source_model.eval()
    target_model.eval()
    
    print("\nMoving models to device...")
    source_model = source_model.to(device)
    target_model = target_model.to(device)
    
    print("Models loaded successfully.\n")
    
    return tokenizer, source_model, target_model


def get_model_type_and_layers(model):
    print("Inspecting model structure...")
    print(f"Model type: {type(model)}")
    
    model_layers = 0
    
    try:
        if hasattr(model, 'base_model'):
            if hasattr(model.base_model, 'model'):
                if hasattr(model.base_model.model, 'model'):
                    if hasattr(model.base_model.model.model, 'layers'):
                        model_layers = len(model.base_model.model.model.layers)
                        print(f"Found layers at model.base_model.model.model.layers: {model_layers} layers")
                        return 'llama', model_layers
                elif hasattr(model.base_model.model, 'layers'):
                    model_layers = len(model.base_model.model.layers)
                    print(f"Found layers at model.base_model.model.layers: {model_layers} layers")
                    return 'llama', model_layers
        
        print("Could not find model layers through standard attributes, exploring model structure...")
        
        # Try to find layers by exploring the model structure
        def explore_modules(module, prefix=""):
            nonlocal model_layers
            if hasattr(module, 'layers') and isinstance(module.layers, torch.nn.ModuleList):
                print(f"Found layers at {prefix}.layers: {len(module.layers)} layers")
                model_layers = len(module.layers)
                return True
                
            found = False
            for name, child in module.named_children():
                if explore_modules(child, f"{prefix}.{name}" if prefix else name):
                    found = True
                    break
            return found
        
        explore_modules(model)
        
        if model_layers > 0:
            print(f"Found {model_layers} layers through exploration")
            return 'llama', model_layers
    except Exception as e:
        print(f"Error exploring model structure: {e}")
    
    raise AttributeError(f"Cannot determine the number of layers in model of type {type(model)}")


def get_lora_parameters(model):
    """Extract LoRA parameters and configuration from a model."""
    lora_config = {
        "r": model.peft_config['default'].r,
        "alpha": model.peft_config['default'].lora_alpha,
        "dropout": model.peft_config['default'].lora_dropout,
        "target_modules": list(model.peft_config['default'].target_modules),
        "task_type": model.peft_config['default'].task_type,
        "bias": model.peft_config['default'].bias,
        "modules_to_save": list(model.peft_config['default'].modules_to_save) if isinstance(model.peft_config['default'].modules_to_save, set) else model.peft_config['default'].modules_to_save,
        "inference_mode": model.peft_config['default'].inference_mode
    }
    return lora_config


def create_model_copy(model: Any) -> Any:
    """
    Create a deep copy of a model for manipulation.
    
    Args:
        model: Model to copy
        
    Returns:
        Deep copy of the model
    """
    import copy
    return copy.deepcopy(model)
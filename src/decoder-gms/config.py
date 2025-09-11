"""
Configuration file for GMS (Guided Module Search) decoder purification
"""

from typing import Dict

# Module mapping for decoder transformer architectures (LLaMA, Qwen, etc.)
MODULE_MAPPING: Dict[str, str] = {
    'Q': 'self_attn.q_proj',      # Query projection in self-attention
    'K': 'self_attn.k_proj',      # Key projection in self-attention  
    'V': 'self_attn.v_proj',      # Value projection in self-attention
    'O': 'self_attn.o_proj',      # Output projection in self-attention
    'G': 'mlp.gate_proj',         # Gate projection in MLP
    'U': 'mlp.up_proj',           # Up projection in MLP
    'D': 'mlp.down_proj'          # Down projection in MLP
}

# Default configuration
DEFAULT_SEED = 42
DEFAULT_ALPHA = 0.4  # Different default for decoder models

# Early stopping configuration
EARLY_STOP_PATIENCE = 5
MAX_NO_IMPROVEMENT = 5

# Reverse mapping for display purposes (long name -> short name)
REVERSE_MODULE_MAPPING = {v.split('.')[-1]: k for k, v in MODULE_MAPPING.items()}

def get_short_module_names(module_set):
    """Convert full module names to short notation for display"""
    short_names = []
    for module in sorted(module_set):
        short_name = REVERSE_MODULE_MAPPING.get(module, module)
        short_names.append(short_name)
    return sorted(short_names)
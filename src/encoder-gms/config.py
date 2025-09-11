"""
Configuration file for GMS (Guided Module Search) encoder purification
"""

from typing import Dict

# Module mapping for transformer architectures
MODULE_MAPPING: Dict[str, str] = {
    'Q': 'attention.self.query',      # Query projection in self-attention
    'K': 'attention.self.key',        # Key projection in self-attention  
    'V': 'attention.self.value',      # Value projection in self-attention
    'O': 'attention.output.dense',    # Output projection in self-attention
    'F': 'intermediate.dense',        # First linear layer in feedforward
    'P': 'output.dense'               # Second linear layer in feedforward
}

# Default configuration
DEFAULT_SEED = 42
DEFAULT_ALPHA = 0.4

# Early stopping configuration
EARLY_STOP_PATIENCE = 5
MAX_NO_IMPROVEMENT = 5
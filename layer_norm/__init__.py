"""layer_norm package

Provides factory helpers and implementations for layer normalisation modules.
"""

from layer_norm.layer_norm import LayerNorm
from layer_norm.layer_norm_factory import get_layer_norm

__all__ = ["LayerNorm", "get_layer_norm"]

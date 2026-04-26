"""activation package

Provides factory helpers and implementations for activation modules.
"""

from activation.gelu import GELU
from activation.activation_factory import get_activation

__all__ = ["GELU", "get_activation"]

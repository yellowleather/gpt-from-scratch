"""activation/activation_factory.py

Factory function for creating activation modules.
"""

from typing import Literal

from torch import nn

from activation.gelu import GELU


def get_activation(
    activation_type: Literal["gelu"] = "gelu",
) -> nn.Module:
    """Return an activation module by name.

    Args:
        activation_type: Implementation to instantiate. Currently only "gelu" is supported.

    Returns:
        An activation module.

    Raises:
        ValueError: If activation_type is not recognised.
    """
    if activation_type == "gelu":
        return GELU()

    raise ValueError(
        f"Unknown activation type '{activation_type}'. Supported types: gelu"
    )


__all__ = ["get_activation"]

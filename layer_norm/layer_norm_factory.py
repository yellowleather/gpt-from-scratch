"""layer_norm/layer_norm_factory.py

Factory function for creating layer normalisation modules.
"""

from typing import Literal

from torch import nn

from layer_norm.layer_norm import LayerNorm


def get_layer_norm(
    norm_type: Literal["layer_norm"] = "layer_norm",
    *,
    emb_dim: int,
) -> nn.Module:
    """Return a layer normalisation module by name.

    Args:
        norm_type: Implementation to instantiate. Currently only "layer_norm" is supported.
        emb_dim: Size of the last dimension to normalise over.

    Returns:
        A layer normalisation module.

    Raises:
        ValueError: If norm_type is not recognised.
    """
    if norm_type == "layer_norm":
        return LayerNorm(emb_dim=emb_dim)

    raise ValueError(
        f"Unknown norm type '{norm_type}'. Supported types: layer_norm"
    )


__all__ = ["get_layer_norm"]

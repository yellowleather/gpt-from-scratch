"""transformer_block/transformer_block_factory.py

Factory function for creating transformer block modules.
"""

from typing import Literal

from torch import nn

from transformer_block.transformer_block import TransformerBlock


def get_transformer_block(
    block_type: Literal["transformer_block"] = "transformer_block",
    *,
    emb_dim: int,
    context_length: int,
    n_heads: int,
    drop_rate: float,
    qkv_bias: bool = False,
) -> nn.Module:
    """Return a transformer block module by name.

    Args:
        block_type: Implementation to instantiate. Currently only "transformer_block"
                    is supported.
        emb_dim: Embedding dimension (input, output, and residual stream width).
        context_length: Maximum sequence length; determines the causal mask size.
        n_heads: Number of attention heads; emb_dim must be divisible by n_heads.
        drop_rate: Dropout probability applied after attention and feed-forward.
        qkv_bias: Whether to include bias in the Q/K/V linear projections.

    Returns:
        A transformer block module.

    Raises:
        ValueError: If block_type is not recognised.
    """
    if block_type == "transformer_block":
        return TransformerBlock(
            emb_dim=emb_dim,
            context_length=context_length,
            n_heads=n_heads,
            drop_rate=drop_rate,
            qkv_bias=qkv_bias,
        )

    raise ValueError(
        f"Unknown block type '{block_type}'. Supported types: transformer_block"
    )


__all__ = ["get_transformer_block"]

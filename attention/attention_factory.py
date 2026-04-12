"""attention/attention_factory.py

Factory functions for creating attention modules.
"""

from typing import Literal

from torch import nn

from attention.multi_head_attention import MultiHeadAttentionWrapper


def get_attention(
    attention_type: Literal["multi_head_wrapper"] = "multi_head_wrapper",
    *,
    d_in: int,
    d_out: int,
    context_length: int,
    dropout: float,
    num_heads: int,
    qkv_bias: bool = False,
) -> nn.Module:
    """Return an attention module by name."""

    if attention_type == "multi_head_wrapper":
        return MultiHeadAttentionWrapper(
            d_in=d_in,
            d_out=d_out,
            context_length=context_length,
            dropout=dropout,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
        )

    raise ValueError(
        f"Unknown attention type '{attention_type}'. Supported types: multi_head_wrapper"
    )


__all__ = ["get_attention"]

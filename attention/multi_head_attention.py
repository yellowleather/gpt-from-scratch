"""attention/multi_head_attention.py

Naive multi-head attention wrapper: runs num_heads independent CausalSelfAttention
modules and concatenates their outputs.
"""

import torch
from torch import nn

from attention.causal_self_attention import CausalSelfAttention


class MultiHeadAttentionWrapper(nn.Module):
    """Naive multi-head causal attention (Variant A).

    Instantiates num_heads separate CausalSelfAttention heads, each with
    dimension d_out // num_heads, then concatenates and projects.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float,
        num_heads: int,
        qkv_bias: bool = False,
    ):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        head_dim = d_out // num_heads
        self.heads = nn.ModuleList(
            [
                CausalSelfAttention(d_in, head_dim, context_length, dropout, qkv_bias)
                for _ in range(num_heads)
            ]
        )
        self.out_proj = nn.Linear(d_out, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        context_vec = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.out_proj(context_vec)


__all__ = ["MultiHeadAttentionWrapper"]

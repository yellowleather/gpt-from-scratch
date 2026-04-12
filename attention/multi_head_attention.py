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

        # Each head projects into a subspace of size head_dim = d_out // num_heads
        # rather than the full d_out, so the total parameter count across all heads
        # stays equal to a single d_out-wide projection.  d_out must divide evenly
        # so every head gets the same dimension.
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        head_dim = d_out // num_heads

        # Create num_heads independent attention heads, each operating on the
        # full input (d_in) but projecting Q/K/V into the smaller head_dim space.
        # Using nn.ModuleList ensures PyTorch registers all heads as submodules,
        # so their parameters appear in model.parameters() and move correctly
        # with .to(device) / .train() / .eval().
        self.heads = nn.ModuleList(
            [
                CausalSelfAttention(d_in, head_dim, context_length, dropout, qkv_bias)
                for _ in range(num_heads)
            ]
        )

        # After the heads are concatenated their combined width is
        # num_heads * head_dim = d_out.  This linear mixes information across
        # heads and projects back to d_out, giving the module a learned way to
        # weight each head's contribution in the final output.
        self.out_proj = nn.Linear(d_out, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Run every head independently on the full input x.
        # Each head returns (b, n_tokens, head_dim); collecting them in a list
        # keeps the per-head tensors separate until we are ready to merge.
        # head(x) internally computes its own Q/K/V projections, applies the
        # causal mask, and returns a weighted sum of its value vectors.
        head_outputs = [head(x) for head in self.heads]

        # Concatenate along the feature axis (dim=-1) to reassemble d_out.
        # dim=0 would merge batches, dim=1 would merge sequence positions —
        # both wrong.  dim=-1 places each head's representation side by side
        # for every token: (b, n_tokens, head_dim) * num_heads
        #                → (b, n_tokens, num_heads * head_dim)
        #                = (b, n_tokens, d_out)
        context_vec = torch.cat(head_outputs, dim=-1)

        # Mix information across heads with a learned linear projection.
        # Without this, each head's slice of the output would be independent
        # of every other head's slice.  out_proj allows the model to learn
        # which combination of head outputs is most useful for downstream layers.
        # Shape in and out: (b, n_tokens, d_out).
        return self.out_proj(context_vec)


__all__ = ["MultiHeadAttentionWrapper"]

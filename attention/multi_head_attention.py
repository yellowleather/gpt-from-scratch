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
        # Step 1 — run each attention head independently on the full input.
        #
        # Every head has its own W_query, W_key, W_value weights, so each one
        # learns to attend to different aspects of the sequence (e.g. one head
        # might focus on syntax, another on co-reference, etc.).
        #
        # x shape:           (b, n_tokens, d_in)
        # each head output:  (b, n_tokens, head_dim)   where head_dim = d_out // num_heads
        head_outputs = []
        for head in self.heads:
            head_out = head(x)   # CausalSelfAttention.forward — Q/K/V + masked softmax
            head_outputs.append(head_out)

        # Step 2 — concatenate all head outputs along the feature axis (dim=-1).
        #
        # We now have num_heads tensors of shape (b, n_tokens, head_dim).
        # Stacking on dim=-1 places them side by side in the feature dimension:
        #
        #   head 0: [..., 0:head_dim]
        #   head 1: [..., head_dim:2*head_dim]
        #   ...
        #
        # Result shape: (b, n_tokens, num_heads * head_dim) == (b, n_tokens, d_out)
        #
        # dim=0 would wrongly merge batch items; dim=1 would wrongly merge
        # sequence positions — dim=-1 is the only axis that makes sense here.
        context_vec = torch.cat(head_outputs, dim=-1)

        # Step 3 — mix information across heads with a learned linear projection.
        #
        # After concatenation each head's slice is still isolated — the model
        # cannot yet combine what head 0 learned with what head 1 learned.
        # out_proj (nn.Linear d_out → d_out) applies a full matrix multiply
        # across all features, letting the model learn which blend of head
        # outputs is most useful for downstream layers.
        #
        # Shape in and out: (b, n_tokens, d_out) — unchanged.
        return self.out_proj(context_vec)


__all__ = ["MultiHeadAttentionWrapper"]

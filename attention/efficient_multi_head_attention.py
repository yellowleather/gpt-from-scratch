"""attention/efficient_multi_head_attention.py

Efficient multi-head attention (Variant B): a single set of Q/K/V projections
split into heads via reshape and transpose rather than num_heads separate modules.
"""

import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    """Efficient multi-head causal attention (Variant B).

    Uses one W_query/W_key/W_value projection for all heads combined, then
    reshapes and transposes to compute all heads in parallel.  More memory-
    efficient than MultiHeadAttentionWrapper because it avoids redundant
    Python-level loops over separate Linear modules.
    """

    mask: torch.Tensor  # registered buffer — declared so type checkers resolve .bool() correctly

    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float,
        num_heads: int,
        qkv_bias: bool = False,
    ):
        """
        Args:
            d_in: Input embedding dimension.
            d_out: Total output dimension across all heads; must be divisible by num_heads.
            context_length: Maximum sequence length; determines the causal mask size.
            dropout: Dropout probability applied to attention weights after softmax.
            num_heads: Number of attention heads to split d_out across.
            qkv_bias: Whether to include a bias term in the Q/K/V linear projections.
        """
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads

        # Each head operates on a slice of size head_dim = d_out // num_heads.
        # The full d_out-wide projection is later split along this dimension.
        self.head_dim = d_out // num_heads

        # Single Q/K/V projection for all heads combined.
        # Shape of each weight matrix: (d_in, d_out).
        # Contrast with MultiHeadAttentionWrapper which has num_heads * 3
        # separate Linear modules; here we have just 3, regardless of num_heads.
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # Projects the concatenated head outputs back to d_out, letting the
        # model learn which combination of head outputs matters downstream.
        self.out_proj = nn.Linear(d_out, d_out)

        self.dropout = nn.Dropout(dropout)

        # Upper-triangular causal mask — same role as in CausalSelfAttention.
        # mask[i, j] = 1 when j > i (future); those scores are filled with -inf.
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, num_tokens, d_in = x.shape

        # Step 1 — project input into Q/K/V spaces (all heads at once).
        # Each output is (b, num_tokens, d_out); the head split happens next.
        keys    = self.W_key(x)
        queries = self.W_query(x)
        values  = self.W_value(x)

        # Step 2 — split d_out into (num_heads, head_dim) by reshaping.
        # view adds a new dimension without copying data:
        #   (b, num_tokens, d_out) → (b, num_tokens, num_heads, head_dim)
        keys    = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values  = values.view(b, num_tokens, self.num_heads, self.head_dim)

        # Step 3 — move the num_heads dimension before num_tokens so that
        # the batch matmul in step 4 operates independently per head.
        #   (b, num_tokens, num_heads, head_dim) → (b, num_heads, num_tokens, head_dim)
        keys    = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values  = values.transpose(1, 2)

        # Step 4 — scaled dot-product attention scores, per head.
        # transpose(2, 3) swaps num_tokens ↔ head_dim on keys so the inner
        # dimensions align for matmul:
        #   queries: (b, num_heads, num_tokens, head_dim)
        #   keys.T:  (b, num_heads, head_dim,   num_tokens)
        #   result:  (b, num_heads, num_tokens, num_tokens)
        attn_scores = queries @ keys.transpose(2, 3)

        # Step 5 — apply causal mask: positions where mask=1 (j > i) become -inf
        # so softmax drives their weight to zero.  Sliced to num_tokens to
        # support sequences shorter than context_length.
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # Step 6 — scale and normalise into attention weights.
        # Dividing by sqrt(head_dim) prevents large dot products from
        # saturating softmax and killing gradients.
        attn_weights = torch.softmax(attn_scores / self.head_dim ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Step 7 — weighted sum of value vectors.
        # (b, num_heads, num_tokens, num_tokens) @ (b, num_heads, num_tokens, head_dim)
        # → (b, num_heads, num_tokens, head_dim)
        # transpose(1, 2) moves num_heads back after num_tokens:
        # → (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Step 8 — merge the num_heads and head_dim axes back into d_out.
        # contiguous() ensures the tensor is stored in a contiguous memory
        # block after the transpose before view can reshape it.
        # (b, num_tokens, num_heads, head_dim) → (b, num_tokens, d_out)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)

        # Step 9 — mix information across heads with the output projection.
        return self.out_proj(context_vec)


__all__ = ["MultiHeadAttention"]

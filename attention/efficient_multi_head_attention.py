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

    Complexity notation used throughout this file:
        B   = batch size
        T   = num_tokens (seq_len, up to context_length)
        D   = d_out  (= d_in in transformer block usage, both equal emb_dim)
        H   = num_heads
        d_h = head_dim = D / H

    Parameters:
        W_query, W_key, W_value : (D, D) each  →  3 D²  params
        out_proj                 : (D, D)        →    D²  params
        Total trainable          : 4 D²  (+3D biases if qkv_bias=True, +D for out_proj always)
        Causal mask buffer       : context_length²  booleans  (not trainable)

    Activation memory during forward:
        Q, K, V projections : (B, T, D) each  →  3 B T D  floats
        reshape/transpose   : views of Q/K/V — no new memory allocated
        attn_scores         : (B, H, T, T)    →  B H T²   floats  ← O(T²) bottleneck
        attn_weights        : (B, H, T, T)    →  B H T²   floats  (softmax output)
        context_vec         : (B, T, D)        →  B T D    floats
        Peak                : 2 B H T²  +  3 B T D
                              (attn_scores and attn_weights coexist during softmax;
                               Q/K/V are still live at that point)

    FLOPs per forward pass (matmuls dominate; elementwise ops noted separately):
        Q/K/V projections : 3 × 2 B T D²   =  6 B T D²
        Q @ Kᵀ scores    : 2 B T² D        (H heads × T×T output × 2 d_h ops/element;
                                             H × d_h = D so cost is independent of H)
        softmax + scale  : ~4 B H T²       (scale: B H T², exp+sum+div: ~3 B H T²)
        dropout          :   B H T²        (elementwise mask — 0 multiply-adds)
        weights @ V      : 2 B T² D        (same shape argument as Q @ Kᵀ)
        out_proj         : 2 B T D²
        Total matmuls    : 8 B T D²  +  4 B T² D
        (softmax/dropout are B H T² ≈ B T² when H is small — negligible vs matmuls)
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
        # Weight shape: (d_in, d_out) = (D, D) → D² params each (+D bias if qkv_bias).
        # Contrast with MultiHeadAttentionWrapper which has num_heads * 3
        # separate Linear modules; here we have just 3, regardless of num_heads.
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # Projects the concatenated head outputs back to d_out, letting the
        # model learn which combination of head outputs matters downstream.
        # Weight shape: (D, D) → D² params (+D bias, always present).
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
        # Each output is (B, T, D); the head split happens next.
        # Memory: 3 × (B, T, D) — 3 B T D floats total.
        # FLOPs:  3 × 2 B T D²  =  6 B T D²  (one matmul per projection).
        keys    = self.W_key(x)
        queries = self.W_query(x)
        values  = self.W_value(x)

        # Step 2 — split d_out into (num_heads, head_dim) by reshaping.
        # view adds a new dimension without copying data:
        #   (B, T, D) → (B, T, H, d_h)
        # Memory: 0 extra — view is a metadata-only operation, same storage.
        # FLOPs:  0.
        keys    = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values  = values.view(b, num_tokens, self.num_heads, self.head_dim)

        # Step 3 — move the num_heads dimension before num_tokens so that
        # the batch matmul in step 4 operates independently per head.
        #   (B, T, H, d_h) → (B, H, T, d_h)
        # Memory: 0 extra — transpose returns a view (non-contiguous), no copy.
        # FLOPs:  0.
        keys    = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values  = values.transpose(1, 2)

        # Step 4 — scaled dot-product attention scores, per head.
        # transpose(2, 3) swaps T ↔ d_h on keys so inner dims align for matmul:
        #   queries: (B, H, T, d_h)
        #   keys.T:  (B, H, d_h, T)
        #   result:  (B, H, T, T)
        # Memory: (B, H, T, T) — B H T² floats.  The O(T²) memory bottleneck.
        # FLOPs:  each of the B H T² output elements needs 2 d_h ops →
        #         B H T² × 2 d_h  =  2 B T² (H d_h)  =  2 B T² D.
        attn_scores = queries @ keys.transpose(2, 3)

        # Step 5 — apply causal mask: positions where mask=1 (j > i) become -inf
        # so softmax drives their weight to zero.  Sliced to num_tokens to
        # support sequences shorter than context_length.
        # Memory: reuses attn_scores storage (masked_fill_ is in-place).
        # FLOPs:  0 multiply-adds (boolean mask write, not arithmetic).
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # Step 6 — scale and normalise into attention weights.
        # Dividing by sqrt(head_dim) prevents large dot products from
        # saturating softmax and killing gradients.
        # Memory: (B, H, T, T) — B H T² floats; attn_scores and attn_weights
        #         coexist briefly, making this the peak memory point: 2 B H T².
        # FLOPs:  scale:   B H T²  multiplications  (divide by scalar)
        #         softmax: ~3 B H T²  (exp per element + sum per row + divide)
        #         total:   ~4 B H T²
        attn_weights = torch.softmax(attn_scores / self.head_dim ** 0.5, dim=-1)

        # Memory: (B, H, T, T) — elementwise boolean mask, same shape.
        # FLOPs:  B H T²  (mask + rescale; 0 multiply-adds in eval mode).
        attn_weights = self.dropout(attn_weights)

        # Step 7 — weighted sum of value vectors.
        # (B, H, T, T) @ (B, H, T, d_h) → (B, H, T, d_h)
        # Memory: (B, H, T, d_h) = (B, T, D) floats for the result;
        #         transpose(1,2) moves H back: → (B, T, H, d_h) — still a view.
        # FLOPs:  each of the B H T d_h output elements needs 2T ops →
        #         B H T d_h × 2T  =  2 B T² (H d_h)  =  2 B T² D.
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Step 8 — merge the H and d_h axes back into D.
        # contiguous() materialises a new contiguous copy after the transpose
        # (transpose broke contiguity; view requires contiguous storage).
        # Memory: (B, T, D) — one extra copy triggered by contiguous().
        # FLOPs:  0.
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)

        # Step 9 — mix information across heads with the output projection.
        # Memory: (B, T, D) output.
        # FLOPs:  2 B T D²  (matmul).
        return self.out_proj(context_vec)


__all__ = ["MultiHeadAttention"]

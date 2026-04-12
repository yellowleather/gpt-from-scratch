"""attention/causal_self_attention.py

Single-head causal (masked) self-attention module.
"""

import torch
from torch import nn


class CausalSelfAttention(nn.Module):
    """Single-head causal self-attention with an upper-triangular mask."""

    mask: torch.Tensor  # registered buffer — declared so type checkers resolve .bool() correctly

    def __init__(self, d_in: int, d_out: int, context_length: int, dropout: float, qkv_bias: bool = False):
        """
        Args:
            d_in: Input embedding dimension.
            d_out: Output dimension (size of each Q/K/V projection).
            context_length: Maximum sequence length; determines the size of the causal mask buffer.
            dropout: Dropout probability applied to attention weights after softmax.
            qkv_bias: Whether to include a bias term in the Q/K/V linear projections.
        """
        super().__init__()

        # Store d_out so other modules (e.g. MultiHeadAttentionWrapper) can
        # read the head dimension without inspecting the linear layer directly.
        self.d_out = d_out

        # Learned linear projections that map each token's d_in-dimensional
        # embedding into query, key, and value spaces of size d_out.
        # All three share the same shape; qkv_bias controls whether an additive
        # bias is learned alongside the weight matrix.
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # Dropout applied to attention weights after softmax.  During training
        # this randomly zeros out some attention connections, acting as a
        # regulariser that prevents the model from over-relying on specific
        # token pairs.
        self.dropout = nn.Dropout(dropout)

        # Upper-triangular boolean mask of shape (context_length, context_length).
        # torch.triu(..., diagonal=1) produces an upper-triangular matrix:
        # mask[i, j] = 1 when j > i (future position), 0 otherwise.
        # In forward(), attn_scores.masked_fill_(mask, -inf) uses this to write
        # -inf into every future-position score, so softmax drives those
        # attention weights to zero — token i can only attend to positions 0..i
        # (past and present), never to j > i (future).  register_buffer ensures the mask moves with the
        # model to the correct device (CPU/GPU) without being treated as a
        # learnable parameter.
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n_tokens, d_in = x.shape

        # Project input into query, key, and value spaces.
        # Each linear maps (b, n_tokens, d_in) → (b, n_tokens, d_out).
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # Scaled dot-product attention scores: Q @ Kᵀ → (b, n_tokens, n_tokens).
        # Entry [b, i, j] measures how much token i should attend to token j.
        # transpose(1, 2) swaps dims 1 (n_tokens) and 2 (d_out), turning keys
        # from (b, n_tokens, d_out) into (b, d_out, n_tokens) so the inner
        # dimensions align for matmul — dim 0 (batch) is left untouched.
        # Dividing by sqrt(d_out) keeps the dot products from growing too large
        # in magnitude as d_out increases, which would push softmax into regions
        # of near-zero gradients.
        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores = attn_scores / keys.shape[-1] ** 0.5

        # Apply the causal mask: set future positions to -inf so that after
        # softmax they contribute zero weight. The mask is upper-triangular
        # (diagonal=1), so position i can only attend to positions 0..i.
        # We slice to [:n_tokens, :n_tokens] to support sequences shorter than
        # context_length without shape mismatches.
        attn_scores.masked_fill_(self.mask.bool()[:n_tokens, :n_tokens], -torch.inf)

        # Normalise scores into a probability distribution over the sequence.
        # Each row sums to 1; future positions are effectively zero due to -inf.
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of value vectors: (b, n_tokens, n_tokens) @ (b, n_tokens, d_out)
        # → (b, n_tokens, d_out).  Each output token is a blend of all value
        # vectors, weighted by how much attention it pays to each position.
        return attn_weights @ values


__all__ = ["CausalSelfAttention"]

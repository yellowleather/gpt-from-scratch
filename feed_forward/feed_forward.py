"""feed_forward/feed_forward.py

Position-wise feed-forward network used in each transformer block.
"""

import torch
from torch import nn

from activation.gelu import GELU


class FeedForward(nn.Module):
    """Two-layer position-wise feed-forward network.

    Projects the input up to ff_expansion * emb_dim, applies GELU activation,
    then projects back down to emb_dim.  Applied independently to each
    position (token) in the sequence — hence "position-wise".

    The expansion ratio (conventionally 4×) gives the model extra capacity to
    learn non-linear feature combinations before projecting back, analogous to
    the widening in the original "Attention Is All You Need" FFN.

    Complexity notation used throughout this file:
        B      = batch size
        T      = seq_len
        D      = emb_dim
        ff_exp = ff_expansion (default 4)

    Parameters:
        Linear(D → ff_exp·D) weight : (D, ff_exp·D)       →  ff_exp D²  params
        Linear(D → ff_exp·D) bias   : (ff_exp·D,)         →  ff_exp D   params
        Linear(ff_exp·D → D) weight : (ff_exp·D, D)       →  ff_exp D²  params
        Linear(ff_exp·D → D) bias   : (D,)                →  D          params
        Total weights               : 2 ff_exp D²
        Total (weights + biases)    : 2 ff_exp D²  +  (ff_exp + 1) D  ≈  2 ff_exp D²

    Activation memory during forward:
        hidden  : (B, T, ff_exp·D) — ff_exp B T D floats  ← peak (wider than input/output)
        output  : (B, T, D)        — B T D floats
        Peak    : ff_exp B T D  (the hidden layer after the first Linear + GELU)
        Note: nn.Sequential reuses buffers where possible, so input and output
        of GELU may share storage (no extra allocation for the activation itself).

    FLOPs per forward pass:
        Linear D → ff_exp·D : output has ff_exp·D elements per token.
                              Each element = dot product of length D:
                                D multiplications  (x[0]*W[0,j], x[1]*W[1,j], ..., x[D-1]*W[D-1,j])
                                + D additions      (sum those products together)
                                = 2D ops per output element.
                              Per token: ff_exp·D elements × 2D ops = 2 ff_exp D²
                              Full batch: × B T tokens  →  2 ff_exp B T D²
        GELU                : elementwise over (B, T, ff_exp·D)  →  ~ff_exp B T D
                              (negligible vs the matmuls)
        Linear ff_exp·D → D : output has D elements per token.
                              Each element = dot product of length ff_exp·D  →  2 ff_exp·D ops.
                              Per token: D elements × 2 ff_exp·D ops = 2 ff_exp D²
                              Full batch: × B T tokens  →  2 ff_exp B T D²
                              (same cost as the first Linear — matmul a×b @ b×c always costs 2abc)
        Total               : 4 ff_exp B T D²  (= 16 B T D² when ff_exp=4)
    """

    def __init__(self, emb_dim: int, ff_expansion: int = 4):
        """
        Args:
            emb_dim: Input and output embedding dimension.
            ff_expansion: Hidden layer expansion ratio relative to emb_dim.
                          Defaults to 4, matching GPT-2 and the original
                          Transformer.  The hidden layer width is
                          ff_expansion * emb_dim.
        """
        super().__init__()
        hidden_dim = ff_expansion * emb_dim
        self.layers = nn.Sequential(
            # Expand from D to ff_exp·D to increase model capacity.
            # Params: ff_exp D² weights + ff_exp D biases.
            # FLOPs:  2 ff_exp B T D²  (matmul).
            # Memory: (B, T, ff_exp·D) — peak activation in this module.
            nn.Linear(emb_dim, hidden_dim),
            # Non-linear activation; GELU is smoother than ReLU and empirically
            # performs better for language model pre-training.
            # FLOPs:  ~ff_exp B T D  (elementwise — negligible vs matmuls).
            # Memory: (B, T, ff_exp·D) — same shape as input, can reuse buffer.
            GELU(),
            # Project back to D so the output matches the residual stream.
            # Params: ff_exp D² weights + D biases.
            # FLOPs:  2 ff_exp B T D²  (matmul).
            # Memory: (B, T, D) output.
            nn.Linear(hidden_dim, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the feed-forward network to each position independently.

        Args:
            x: Input tensor of shape (batch, seq_len, emb_dim).

        Returns:
            Tensor of shape (batch, seq_len, emb_dim).
        """
        # nn.Sequential applies Linear → GELU → Linear in order.
        # Because Linear operates on the last dimension, each token position
        # is processed independently — no information flows between positions here.
        # Memory: peak at the hidden layer (B, T, ff_exp·D) = ff_exp B T D floats.
        # FLOPs:  4 ff_exp B T D²  total (two matmuls; GELU is negligible).
        return self.layers(x)


__all__ = ["FeedForward"]

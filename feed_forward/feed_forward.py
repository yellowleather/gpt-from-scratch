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
            # Expand from emb_dim to hidden_dim to increase model capacity.
            nn.Linear(emb_dim, hidden_dim),
            # Non-linear activation; GELU is smoother than ReLU and empirically
            # performs better for language model pre-training.
            GELU(),
            # Project back to emb_dim so the output matches the residual stream.
            nn.Linear(hidden_dim, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the feed-forward network to each position independently.

        Args:
            x: Input tensor of shape (batch, seq_len, emb_dim).

        Returns:
            Tensor of shape (batch, seq_len, emb_dim).
        """
        # nn.Sequential applies the three layers in order.
        # Because Linear operates on the last dimension, each token position
        # is processed independently — no information flows between positions here.
        return self.layers(x)


__all__ = ["FeedForward"]

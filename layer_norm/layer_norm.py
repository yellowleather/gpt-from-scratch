"""layer_norm/layer_norm.py

Custom layer normalisation with learnable affine parameters.
"""

import torch
from torch import nn


class LayerNorm(nn.Module):
    """Layer normalisation with learnable scale and shift.

    Normalises the last dimension of the input to zero mean and unit variance,
    then applies a learnable affine transform (scale * x + shift).

    Uses population variance (unbiased=False) for stability, matching the
    GPT-2 reference implementation.
    """

    def __init__(self, emb_dim: int):
        """
        Args:
            emb_dim: Size of the last dimension to normalise over.
        """
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise x along its last dimension and apply affine transform.

        Args:
            x: Input tensor of any shape (..., emb_dim).

        Returns:
            Tensor of the same shape as x.
        """
        # Compute per-token mean and variance along the embedding dimension.
        # keepdim=True preserves the trailing dimension so broadcasting works
        # correctly when we subtract/divide below.
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        # Subtract mean and divide by std.  eps avoids division by zero for
        # constant or near-constant inputs (e.g. all-zero vectors).
        norm_x = (x - mean) / torch.sqrt(var + self.eps)

        # Apply learnable affine transform: scale shifts the variance and
        # shift offsets the mean, letting the model undo the normalisation
        # selectively if that turns out to be optimal for a given layer.
        return self.scale * norm_x + self.shift


__all__ = ["LayerNorm"]

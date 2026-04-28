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

    Complexity notation (when input is the standard transformer shape):
        B = batch size
        T = seq_len
        D = emb_dim  (the dimension normalised over)

    Parameters:
        scale : (D,) — D floats
        shift : (D,) — D floats
        Total : 2 D  — negligible compared to attention/FFN weights (~12 D²)

    Activation memory during forward:
        mean   : (B, T, 1) — one scalar per token; B T floats (tiny)
        var    : (B, T, 1) — one scalar per token; B T floats (tiny)
        norm_x : (B, T, D) — B T D floats  (held until the affine step)
        output : (B, T, D) — B T D floats
        Peak   : ~2 B T D  (norm_x and output coexist briefly)

    FLOPs per forward pass:
        mean computation      : B T D  additions + B T divisions  ≈  B T D
        var computation       : B T D  subtractions + B T D squares
                                + B T D additions  + B T divisions ≈  3 B T D
        normalise (x - μ)/σ  : B T D  subtractions + B T D divisions
                                + B T  sqrts (one per position, negligible) ≈  2 B T D
        scale * norm_x        : B T D  multiplications
        + shift               : B T D  additions
        Total                 : ~8 B T D

        All operations are element-wise or reductions — no matmuls.
        LayerNorm is cheap relative to attention (8 B T D vs ~24 B T D² + 4 B T² D).
    """

    def __init__(self, emb_dim: int):
        """
        Args:
            emb_dim: Size of the last dimension to normalise over.
        """
        super().__init__()
        self.eps = 1e-5
        # (D,) each — one learnable scalar per embedding dimension.
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise x along its last dimension and apply affine transform.

        Args:
            x: Input tensor of any shape (..., emb_dim).

        Returns:
            Tensor of the same shape as x.
        """
        # Sum D values and divide — one mean scalar per token position.
        # keepdim=True preserves the trailing dimension so broadcasting works
        # correctly when we subtract/divide below.
        # Memory: (B, T, 1) — B T floats (tiny relative to B T D).
        # FLOPs:  B T D additions + B T divisions  ≈  B T D.
        mean = x.mean(dim=-1, keepdim=True)

        # Compute squared deviation from the mean per token.
        # unbiased=False uses population variance (divide by D) rather than
        # sample variance (divide by D-1).  For normalisation we are not
        # estimating an unknown population statistic — we want the exact
        # variance of this specific vector — so dividing by D is correct.
        # Memory: (B, T, 1) — same shape as mean.
        # FLOPs per token: D subtractions (x[i] - mean)
        #                  + D squarings   ((x[i] - mean)²)
        #                  + D additions   (sum of squares)
        #                  + 1 division    (divide by D)
        #                  = 3D + 1 ops per token → B T (3D + 1) ≈ 3 B T D total.
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        # Subtract mean and divide by std.  eps avoids division by zero for
        # constant or near-constant inputs (e.g. all-zero vectors).
        # Memory: (B, T, D) — held until the affine step below.
        # FLOPs per token: D subtractions (x[i] - mean)
        #                  + 1 sqrt      (sqrt(var + eps), computed once, reused for all D divisions)
        #                  + D divisions ((x[i] - mean) / sqrt(var + eps))
        #                  = 2D + 1 ops per token → B T (2D + 1) ≈ 2 B T D total.
        norm_x = (x - mean) / torch.sqrt(var + self.eps)

        # Apply learnable affine transform: scale shifts the variance and
        # shift offsets the mean, letting the model undo the normalisation
        # selectively if that turns out to be optimal for a given layer.
        # Memory: (B, T, D) output; norm_x can be freed after this line.
        # FLOPs:  B T D multiplications (scale) + B T D additions (shift)  =  2 B T D.
        return self.scale * norm_x + self.shift


__all__ = ["LayerNorm"]

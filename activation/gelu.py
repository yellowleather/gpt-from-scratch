"""activation/gelu.py

GELU activation function (tanh approximation).
"""

import torch
from torch import nn


class GELU(nn.Module):
    """Gaussian Error Linear Unit activation (tanh approximation).

    Uses the tanh-based approximation from the original GELU paper:
        0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))

    This matches the GPT-2 reference implementation and is numerically
    equivalent to torch.nn.GELU(approximate='tanh').
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the GELU activation element-wise.

        Args:
            x: Input tensor of any shape.

        Returns:
            Tensor of the same shape as x.
        """
        # The tanh approximation has two parts:
        #   inner = sqrt(2/π) * (x + 0.044715 * x³)
        #   output = 0.5 * x * (1 + tanh(inner))
        # For large positive x: tanh(inner) → 1, so output → x  (identity).
        # For large negative x: tanh(inner) → -1, so output → 0  (gate closed).
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


__all__ = ["GELU"]

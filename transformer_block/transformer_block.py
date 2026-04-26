"""transformer_block/transformer_block.py

Pre-norm transformer block: LayerNorm → Attention → residual,
then LayerNorm → FeedForward → residual.
"""

import torch
from torch import nn

from attention.efficient_multi_head_attention import MultiHeadAttention
from feed_forward.feed_forward import FeedForward
from layer_norm.layer_norm import LayerNorm


class TransformerBlock(nn.Module):
    """Single transformer block with pre-layer-norm and residual connections.

    Each block applies two sublayers in sequence, each wrapped in the same
    pattern: normalise → sublayer → dropout → add residual.

    Pre-norm (normalise *before* the sublayer) stabilises gradient flow in
    deep stacks compared to the original post-norm formulation, and is the
    convention used in GPT-2 and later models.
    """

    def __init__(
        self,
        emb_dim: int,
        context_length: int,
        n_heads: int,
        drop_rate: float,
        qkv_bias: bool = False,
    ):
        """
        Args:
            emb_dim: Embedding dimension (input, output, and residual stream width).
                     Must be divisible by n_heads.
            context_length: Maximum sequence length; determines the causal mask size.
            n_heads: Number of attention heads.
            drop_rate: Dropout probability applied after attention and feed-forward.
            qkv_bias: Whether to include bias in the Q/K/V linear projections.
        """
        super().__init__()

        # Multi-head causal self-attention (efficient variant B).
        # d_in == d_out == emb_dim so the residual addition is shape-compatible.
        self.att = MultiHeadAttention(
            d_in=emb_dim,
            d_out=emb_dim,
            context_length=context_length,
            dropout=drop_rate,
            num_heads=n_heads,
            qkv_bias=qkv_bias,
        )

        # Position-wise feed-forward with 4× hidden expansion.
        self.ff = FeedForward(emb_dim=emb_dim)

        # Separate LayerNorm instances for each sublayer so their scale/shift
        # parameters can specialise independently.
        self.norm1 = LayerNorm(emb_dim=emb_dim)  # applied before attention
        self.norm2 = LayerNorm(emb_dim=emb_dim)  # applied before feed-forward

        # Shared dropout applied after each sublayer before the residual add.
        self.drop_shortcut = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply attention and feed-forward sublayers with pre-norm and residuals.

        Args:
            x: Input tensor of shape (batch, seq_len, emb_dim).

        Returns:
            Tensor of shape (batch, seq_len, emb_dim).
        """
        # --- Attention sublayer ---
        # Save input so we can add it back as a residual after the sublayer.
        shortcut = x
        # Pre-norm: normalise before attention so the sublayer sees a
        # well-conditioned input regardless of how x has been scaled so far.
        x = self.norm1(x)
        x = self.att(x)
        # Dropout regularises by randomly zeroing activations during training.
        x = self.drop_shortcut(x)
        # Residual connection: adding the original input back lets gradients
        # flow directly to earlier layers and prevents the attention sublayer
        # from having to learn an identity mapping.
        x = x + shortcut

        # --- Feed-forward sublayer (same pattern) ---
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x


__all__ = ["TransformerBlock"]

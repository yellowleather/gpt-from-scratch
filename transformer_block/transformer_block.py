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

    Complexity notation used throughout this file:
        B  = batch size
        T  = sequence length (num_tokens, up to context_length)
        D  = emb_dim
        H  = n_heads
        d_h = head_dim = D / H

    Parameters per block (weights only, ignoring biases):
        Attention  :  4 D²   (3 D² for Q/K/V projections + D² for out_proj)
        FeedForward: 8 D²   (D→4D expand: 4 D², 4D→D contract: 4 D²)
        LayerNorms :  2 × 2D  ≈  0  (negligible vs D²)
        Total      : ~12 D²  per block

    Activation memory during forward (dominant terms):
        Attention score matrix  : (B, H, T, T) = B H T²  elements  ← O(T²) bottleneck
        QKV + intermediate acts : O(B T D)  each
        FFN hidden layer        : (B, T, 4D) = 4 B T D  elements
        Peak is typically the attention matrix for long sequences.

    FLOPs per forward pass (multiply-adds counted as 2 ops each):
        LayerNorm × 2     : 2 × 8 B T D           =  16 B T D
        Q/K/V projections : 3 × 2 B T D²          =   6 B T D²
        Attention scores  : 2 B T² D               (Q @ K^T, all heads combined)
        Attention output  : 2 B T² D               (weights @ V)
        Output projection : 2 B T D²
        FFN expand D→4D  : 8 B T D²
        FFN contract 4D→D: 8 B T D²
        Residuals × 2     : 2 × B T D              =   2 B T D  (element-wise adds)
        Dropout × 2       : 2 × B T D              =   2 B T D  (element-wise mask)
        Total             : ~24 B T D²  +  4 B T² D  +  20 B T D
        The 20 B T D linear terms are negligible for large D (e.g. D=256 → 20×256
        vs 24×256² per token-step), so the dominant costs remain:
            24 B T D²  — matmuls, dominates for wide models
             4 B T² D  — attention scores/output, dominates for long sequences
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
        # Params: 3 × (D × D) for W_query/W_key/W_value + D² for out_proj = 4 D²
        #         (+4D biases when qkv_bias=True, +D² bias for out_proj always)
        # Causal mask buffer: context_length² booleans (not trainable params).
        self.att = MultiHeadAttention(
            d_in=emb_dim,
            d_out=emb_dim,
            context_length=context_length,
            dropout=drop_rate,
            num_heads=n_heads,
            qkv_bias=qkv_bias,
        )

        # Position-wise feed-forward with 4× hidden expansion.
        # Params: D × 4D (expand) + 4D × D (contract) = 8 D²  (+5D biases)
        self.ff = FeedForward(emb_dim=emb_dim)

        # Separate LayerNorm instances for each sublayer so their scale/shift
        # parameters can specialise independently.
        # Params: 2 × 2D (scale + shift each) = 4D total — negligible vs D².
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
        shortcut = x                 # (B, T, D) — alias, no extra memory allocated

        # Pre-norm: normalise before attention so the sublayer sees a
        # well-conditioned input regardless of how x has been scaled so far.
        # Memory: (B, T, D) activation — B T D elements
        # FLOPs:  ~8 B T D  (mean: BTD, var: 3BTD, normalise: 2BTD, scale+shift: 2BTD)
        x = self.norm1(x)

        # Attention: Q/K/V projections, scaled dot-product, output projection.
        # Memory: peak at score matrix (B, H, T, T) = B H T² elements;
        #         Q/K/V tensors each (B, T, D) = B T D elements.
        # FLOPs:  6 B T D²  (Q/K/V projections)
        #       + 2 B T² D  (Q @ K^T scores)
        #       + 2 B T² D  (weights @ V)
        #       + 2 B T D²  (out_proj)
        #       = 8 B T D²  + 4 B T² D
        x = self.att(x)

        # Dropout regularises by randomly zeroing activations during training.
        # Memory: (B, T, D) — element-wise mask, same shape as input.
        # FLOPs:  B T D  (mask + scale operations; zero-cost when eval mode)
        x = self.drop_shortcut(x)

        # Residual connection: adding the original input back lets gradients
        # flow directly to earlier layers and prevents the attention sublayer
        # from having to learn an identity mapping.
        # FLOPs: B T D  (element-wise addition)
        x = x + shortcut

        # --- Feed-forward sublayer (same pattern) ---
        shortcut = x                 # (B, T, D) — alias

        # Memory: (B, T, D) activation
        # FLOPs:  ~8 B T D  (same as norm1)
        x = self.norm2(x)

        # FFN: expand D → 4D, GELU, contract 4D → D.
        # Memory: peak at hidden layer (B, T, 4D) = 4 B T D elements.
        # FLOPs:  8 B T D²  (Linear D→4D)
        #       + ~B T 4D   (GELU, elementwise — negligible vs matmuls)
        #       + 8 B T D²  (Linear 4D→D)
        #       = 16 B T D²
        x = self.ff(x)

        # FLOPs: B T D  (mask + scale); same shape as input
        x = self.drop_shortcut(x)

        # FLOPs: B T D  (element-wise addition)
        x = x + shortcut

        return x


__all__ = ["TransformerBlock"]

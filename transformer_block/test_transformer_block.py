"""transformer_block/test_transformer_block.py"""

import torch
import pytest

from transformer_block.transformer_block import TransformerBlock
from transformer_block.transformer_block_factory import get_transformer_block


EMB_DIM = 8
CTX = 16
N_HEADS = 2


@pytest.fixture
def block():
    return TransformerBlock(
        emb_dim=EMB_DIM,
        context_length=CTX,
        n_heads=N_HEADS,
        drop_rate=0.0,
    )


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------


class TestTransformerBlockOutputShape:
    def test_shape_preserved(self, block):
        x = torch.randn(4, 10, EMB_DIM)
        assert block(x).shape == (4, 10, EMB_DIM)

    def test_single_token(self, block):
        x = torch.randn(1, 1, EMB_DIM)
        assert block(x).shape == (1, 1, EMB_DIM)

    def test_seq_len_equals_context_length(self, block):
        x = torch.randn(2, CTX, EMB_DIM)
        assert block(x).shape == (2, CTX, EMB_DIM)


# ---------------------------------------------------------------------------
# Residual connections
# ---------------------------------------------------------------------------


class TestTransformerBlockResiduals:
    def test_output_is_not_identical_to_input(self, block):
        """The block should transform the input, not pass it through unchanged."""
        block.eval()
        x = torch.randn(2, 6, EMB_DIM)
        assert not torch.allclose(block(x), x)

    def test_zero_dropout_deterministic(self):
        """With drop_rate=0 and eval mode, two forward passes must be identical."""
        block = TransformerBlock(
            emb_dim=EMB_DIM, context_length=CTX, n_heads=N_HEADS, drop_rate=0.0
        )
        block.eval()
        x = torch.randn(2, 6, EMB_DIM)
        assert torch.allclose(block(x), block(x))


# ---------------------------------------------------------------------------
# Causal masking (inherited from attention sublayer)
# ---------------------------------------------------------------------------


class TestTransformerBlockCausalMask:
    def test_future_tokens_do_not_affect_past(self):
        """Changing a future token must not alter earlier positions' outputs."""
        block = TransformerBlock(
            emb_dim=EMB_DIM, context_length=CTX, n_heads=N_HEADS, drop_rate=0.0
        )
        block.eval()
        torch.manual_seed(0)
        x = torch.randn(1, 6, EMB_DIM)
        out_original = block(x).detach().clone()

        # Corrupt only the last token
        x_modified = x.clone()
        x_modified[0, -1] = torch.randn(EMB_DIM)
        out_modified = block(x_modified).detach()

        # Position 0 must be unaffected by the change at the last position
        assert torch.allclose(out_original[0, 0], out_modified[0, 0])


# ---------------------------------------------------------------------------
# Internal structure
# ---------------------------------------------------------------------------


class TestTransformerBlockStructure:
    def test_has_two_layer_norms(self, block):
        from layer_norm.layer_norm import LayerNorm
        assert isinstance(block.norm1, LayerNorm)
        assert isinstance(block.norm2, LayerNorm)

    def test_has_feed_forward(self, block):
        from feed_forward.feed_forward import FeedForward
        assert isinstance(block.ff, FeedForward)

    def test_has_attention(self, block):
        from attention.efficient_multi_head_attention import MultiHeadAttention
        assert isinstance(block.att, MultiHeadAttention)

    def test_n_heads_not_dividing_emb_dim_raises(self):
        with pytest.raises(AssertionError):
            TransformerBlock(
                emb_dim=EMB_DIM, context_length=CTX, n_heads=3, drop_rate=0.0
            )


# ---------------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------------


class TestTransformerBlockGradients:
    def test_gradients_flow_through_all_sublayers(self, block):
        x = torch.randn(2, 5, EMB_DIM)
        block(x).sum().backward()
        # Attention weights
        assert block.att.W_query.weight.grad is not None
        assert block.att.W_key.weight.grad is not None
        assert block.att.W_value.weight.grad is not None
        # Feed-forward weights
        assert block.ff.layers[0].weight.grad is not None
        assert block.ff.layers[2].weight.grad is not None
        # LayerNorm parameters
        assert block.norm1.scale.grad is not None
        assert block.norm2.scale.grad is not None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestTransformerBlockFactory:
    def test_returns_transformer_block(self):
        module = get_transformer_block(
            emb_dim=EMB_DIM, context_length=CTX, n_heads=N_HEADS, drop_rate=0.0
        )
        assert isinstance(module, TransformerBlock)

    def test_explicit_type(self):
        module = get_transformer_block(
            "transformer_block",
            emb_dim=EMB_DIM,
            context_length=CTX,
            n_heads=N_HEADS,
            drop_rate=0.0,
        )
        assert isinstance(module, TransformerBlock)

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown block type"):
            get_transformer_block(
                "encoder_block",  # type: ignore[arg-type]
                emb_dim=EMB_DIM,
                context_length=CTX,
                n_heads=N_HEADS,
                drop_rate=0.0,
            )

    def test_factory_output_shape(self):
        block = get_transformer_block(
            emb_dim=EMB_DIM, context_length=CTX, n_heads=N_HEADS, drop_rate=0.0
        )
        x = torch.randn(3, 8, EMB_DIM)
        assert block(x).shape == (3, 8, EMB_DIM)

"""tests/test_attention.py"""

import torch
from torch import nn
import pytest

from attention.causal_self_attention import CausalSelfAttention
from attention.multi_head_attention import MultiHeadAttentionWrapper
from attention.efficient_multi_head_attention import MultiHeadAttention
from attention.attention_factory import get_attention


D_IN = 8
D_OUT = 8
CTX = 16
NUM_HEADS = 2


@pytest.fixture
def csa():
    return CausalSelfAttention(d_in=D_IN, d_out=D_OUT, context_length=CTX, dropout=0.0)


@pytest.fixture
def mha():
    return MultiHeadAttentionWrapper(
        d_in=D_IN, d_out=D_OUT, context_length=CTX, dropout=0.0, num_heads=NUM_HEADS
    )


# ---------------------------------------------------------------------------
# CausalSelfAttention
# ---------------------------------------------------------------------------


class TestCausalSelfAttentionOutputShape:
    def test_shape_batch_seq_dout(self, csa):
        x = torch.randn(4, 10, D_IN)
        assert csa(x).shape == (4, 10, D_OUT)

    def test_single_token(self, csa):
        x = torch.randn(1, 1, D_IN)
        assert csa(x).shape == (1, 1, D_OUT)

    def test_seq_len_equals_context_length(self, csa):
        x = torch.randn(2, CTX, D_IN)
        assert csa(x).shape == (2, CTX, D_OUT)


class TestCausalMask:
    def test_future_tokens_do_not_affect_past(self):
        """Changing a future token should not alter earlier positions' outputs."""
        csa = CausalSelfAttention(d_in=D_IN, d_out=D_OUT, context_length=CTX, dropout=0.0)
        csa.eval()
        torch.manual_seed(0)
        x = torch.randn(1, 6, D_IN)
        out_original = csa(x).detach().clone()

        # Corrupt the last token
        x_modified = x.clone()
        x_modified[0, -1] = torch.randn(D_IN)
        out_modified = csa(x_modified).detach()

        # Position 0 should be identical; the last position may differ
        assert torch.allclose(out_original[0, 0], out_modified[0, 0])

    def test_mask_is_upper_triangular(self, csa):
        mask = csa.mask
        # Everything on/below the diagonal must be zero
        lower = torch.tril(mask)
        assert lower.sum().item() == 0


class TestCausalSelfAttentionGradients:
    def test_gradients_flow_through_weights(self, csa):
        x = torch.randn(2, 5, D_IN)
        csa(x).sum().backward()
        assert csa.W_query.weight.grad is not None
        assert csa.W_key.weight.grad is not None
        assert csa.W_value.weight.grad is not None


# ---------------------------------------------------------------------------
# MultiHeadAttentionWrapper
# ---------------------------------------------------------------------------


class TestMultiHeadAttentionWrapperOutputShape:
    def test_shape_matches_input_seq_and_dout(self, mha):
        x = torch.randn(4, 10, D_IN)
        assert mha(x).shape == (4, 10, D_OUT)

    def test_single_token(self, mha):
        x = torch.randn(1, 1, D_IN)
        assert mha(x).shape == (1, 1, D_OUT)

    def test_seq_len_equals_context_length(self, mha):
        x = torch.randn(2, CTX, D_IN)
        assert mha(x).shape == (2, CTX, D_OUT)


class TestMultiHeadAttentionWrapperStructure:
    def test_num_heads_matches_module_list(self, mha):
        assert len(mha.heads) == NUM_HEADS

    def test_each_head_is_causal_self_attention(self, mha):
        for head in mha.heads:
            assert isinstance(head, CausalSelfAttention)

    def test_head_dim_is_d_out_divided_by_num_heads(self, mha):
        expected_head_dim = D_OUT // NUM_HEADS
        for head in mha.heads:
            assert head.d_out == expected_head_dim

    def test_out_proj_maps_dout_to_dout(self, mha):
        assert mha.out_proj.in_features == D_OUT
        assert mha.out_proj.out_features == D_OUT

    def test_d_out_not_divisible_by_num_heads_raises(self):
        with pytest.raises(AssertionError):
            MultiHeadAttentionWrapper(
                d_in=D_IN, d_out=7, context_length=CTX, dropout=0.0, num_heads=3
            )


class TestMultiHeadAttentionWrapperGradients:
    def test_gradients_flow_through_all_heads_and_proj(self, mha):
        x = torch.randn(2, 5, D_IN)
        mha(x).sum().backward()
        assert mha.out_proj.weight.grad is not None
        for head in mha.heads:
            assert head.W_query.weight.grad is not None


# ---------------------------------------------------------------------------
# MultiHeadAttention (efficient Variant B)
# ---------------------------------------------------------------------------


@pytest.fixture
def emha():
    return MultiHeadAttention(
        d_in=D_IN, d_out=D_OUT, context_length=CTX, dropout=0.0, num_heads=NUM_HEADS
    )


class TestMultiHeadAttentionOutputShape:
    def test_shape_matches_input_seq_and_dout(self, emha):
        x = torch.randn(4, 10, D_IN)
        assert emha(x).shape == (4, 10, D_OUT)

    def test_single_token(self, emha):
        x = torch.randn(1, 1, D_IN)
        assert emha(x).shape == (1, 1, D_OUT)

    def test_seq_len_equals_context_length(self, emha):
        x = torch.randn(2, CTX, D_IN)
        assert emha(x).shape == (2, CTX, D_OUT)


class TestMultiHeadAttentionStructure:
    def test_head_dim_is_d_out_divided_by_num_heads(self, emha):
        assert emha.head_dim == D_OUT // NUM_HEADS

    def test_single_qkv_projections(self, emha):
        # Variant B uses one W_query/W_key/W_value for all heads combined,
        # not a ModuleList — verify no .heads attribute exists.
        assert not hasattr(emha, "heads")
        assert isinstance(emha.W_query, nn.Linear)
        assert isinstance(emha.W_key, nn.Linear)
        assert isinstance(emha.W_value, nn.Linear)

    def test_qkv_projection_width_is_d_out(self, emha):
        # Each projection maps d_in → d_out (all heads share one matrix).
        assert emha.W_query.out_features == D_OUT
        assert emha.W_key.out_features == D_OUT
        assert emha.W_value.out_features == D_OUT

    def test_out_proj_maps_dout_to_dout(self, emha):
        assert emha.out_proj.in_features == D_OUT
        assert emha.out_proj.out_features == D_OUT

    def test_d_out_not_divisible_by_num_heads_raises(self):
        with pytest.raises(AssertionError):
            MultiHeadAttention(
                d_in=D_IN, d_out=7, context_length=CTX, dropout=0.0, num_heads=3
            )


class TestMultiHeadAttentionCausalMask:
    def test_future_tokens_do_not_affect_past(self):
        emha = MultiHeadAttention(
            d_in=D_IN, d_out=D_OUT, context_length=CTX, dropout=0.0, num_heads=NUM_HEADS
        )
        emha.eval()
        torch.manual_seed(0)
        x = torch.randn(1, 6, D_IN)
        out_original = emha(x).detach().clone()

        x_modified = x.clone()
        x_modified[0, -1] = torch.randn(D_IN)
        out_modified = emha(x_modified).detach()

        assert torch.allclose(out_original[0, 0], out_modified[0, 0])


class TestMultiHeadAttentionGradients:
    def test_gradients_flow_through_weights_and_proj(self, emha):
        x = torch.randn(2, 5, D_IN)
        emha(x).sum().backward()
        assert emha.W_query.weight.grad is not None
        assert emha.W_key.weight.grad is not None
        assert emha.W_value.weight.grad is not None
        assert emha.out_proj.weight.grad is not None


# ---------------------------------------------------------------------------
# attention_factory
# ---------------------------------------------------------------------------


class TestAttentionFactory:
    def test_returns_multi_head_wrapper(self):
        module = get_attention(
            "multi_head_wrapper",
            d_in=D_IN,
            d_out=D_OUT,
            context_length=CTX,
            dropout=0.0,
            num_heads=NUM_HEADS,
        )
        assert isinstance(module, MultiHeadAttentionWrapper)

    def test_default_type_is_multi_head_wrapper(self):
        module = get_attention(
            d_in=D_IN,
            d_out=D_OUT,
            context_length=CTX,
            dropout=0.0,
            num_heads=NUM_HEADS,
        )
        assert isinstance(module, MultiHeadAttentionWrapper)

    def test_returns_multi_head(self):
        module = get_attention(
            "multi_head",
            d_in=D_IN,
            d_out=D_OUT,
            context_length=CTX,
            dropout=0.0,
            num_heads=NUM_HEADS,
        )
        assert isinstance(module, MultiHeadAttention)

    def test_unknown_type_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown attention type"):
            get_attention(
                "unknown",
                d_in=D_IN,
                d_out=D_OUT,
                context_length=CTX,
                dropout=0.0,
                num_heads=NUM_HEADS,
            )

    def test_factory_output_shape(self):
        module = get_attention(
            d_in=D_IN,
            d_out=D_OUT,
            context_length=CTX,
            dropout=0.0,
            num_heads=NUM_HEADS,
        )
        x = torch.randn(3, 8, D_IN)
        assert module(x).shape == (3, 8, D_OUT)

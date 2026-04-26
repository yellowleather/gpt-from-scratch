"""feed_forward/test_feed_forward.py"""

import torch
import pytest

from feed_forward.feed_forward import FeedForward
from feed_forward.feed_forward_factory import get_feed_forward


EMB_DIM = 8


@pytest.fixture
def ff():
    return FeedForward(emb_dim=EMB_DIM)


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------


class TestFeedForwardOutputShape:
    def test_shape_preserved(self, ff):
        x = torch.randn(4, 10, EMB_DIM)
        assert ff(x).shape == (4, 10, EMB_DIM)

    def test_single_token(self, ff):
        x = torch.randn(1, 1, EMB_DIM)
        assert ff(x).shape == (1, 1, EMB_DIM)

    def test_batch_size_one(self, ff):
        x = torch.randn(1, 6, EMB_DIM)
        assert ff(x).shape == (1, 6, EMB_DIM)


# ---------------------------------------------------------------------------
# Internal structure
# ---------------------------------------------------------------------------


class TestFeedForwardStructure:
    def test_hidden_dim_is_4x_emb_dim_by_default(self, ff):
        # Default expansion: emb_dim → 4 * emb_dim
        assert ff.layers[0].out_features == 4 * EMB_DIM

    def test_custom_expansion_ratio(self):
        ff = FeedForward(emb_dim=EMB_DIM, ff_expansion=8)
        assert ff.layers[0].out_features == 8 * EMB_DIM

    def test_output_proj_maps_back_to_emb_dim(self, ff):
        # Last linear always maps back to emb_dim regardless of expansion
        assert ff.layers[2].out_features == EMB_DIM

    def test_positions_are_independent(self, ff):
        """Changing one token position should not alter any other position's output."""
        ff.eval()
        torch.manual_seed(0)
        x = torch.randn(1, 6, EMB_DIM)
        out_original = ff(x).detach().clone()

        # Overwrite the last token with random values
        x_modified = x.clone()
        x_modified[0, -1] = torch.randn(EMB_DIM)
        out_modified = ff(x_modified).detach()

        # All positions except the last must be unchanged
        assert torch.allclose(out_original[0, :-1], out_modified[0, :-1])


# ---------------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------------


class TestFeedForwardGradients:
    def test_gradients_flow_through_both_linears(self, ff):
        x = torch.randn(2, 5, EMB_DIM)
        ff(x).sum().backward()
        assert ff.layers[0].weight.grad is not None
        assert ff.layers[2].weight.grad is not None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestFeedForwardFactory:
    def test_returns_feed_forward(self):
        module = get_feed_forward(emb_dim=EMB_DIM)
        assert isinstance(module, FeedForward)

    def test_explicit_type(self):
        module = get_feed_forward("feed_forward", emb_dim=EMB_DIM)
        assert isinstance(module, FeedForward)

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown feed-forward type"):
            get_feed_forward("mlp", emb_dim=EMB_DIM)  # type: ignore[arg-type]

    def test_factory_output_shape(self):
        ff = get_feed_forward(emb_dim=EMB_DIM)
        x = torch.randn(3, 6, EMB_DIM)
        assert ff(x).shape == (3, 6, EMB_DIM)

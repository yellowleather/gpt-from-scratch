"""layer_norm/test_layer_norm.py"""

import torch
import pytest

from layer_norm.layer_norm import LayerNorm
from layer_norm.layer_norm_factory import get_layer_norm


EMB_DIM = 8


@pytest.fixture
def ln():
    return LayerNorm(emb_dim=EMB_DIM)


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------


class TestLayerNormOutputShape:
    def test_3d_input(self, ln):
        x = torch.randn(4, 10, EMB_DIM)
        assert ln(x).shape == (4, 10, EMB_DIM)

    def test_2d_input(self, ln):
        x = torch.randn(4, EMB_DIM)
        assert ln(x).shape == (4, EMB_DIM)

    def test_single_token(self, ln):
        x = torch.randn(1, 1, EMB_DIM)
        assert ln(x).shape == (1, 1, EMB_DIM)


# ---------------------------------------------------------------------------
# Normalisation behaviour
# ---------------------------------------------------------------------------


class TestLayerNormBehaviour:
    def test_output_mean_near_zero(self, ln):
        """Before the affine transform modifies it, the normalised mean should be 0."""
        # Force scale=1, shift=0 (default init) and verify the raw normalisation.
        x = torch.randn(16, EMB_DIM)
        out = ln(x)
        # Mean over the last dim should be close to zero given default shift=0
        assert torch.allclose(out.mean(dim=-1), torch.zeros(16), atol=1e-5)

    def test_output_std_near_one(self, ln):
        """With default scale=1, std should be near 1."""
        x = torch.randn(16, EMB_DIM) * 10  # large variance to exercise normalisation
        out = ln(x)
        std = out.std(dim=-1, unbiased=False)
        assert torch.allclose(std, torch.ones(16), atol=1e-4)

    def test_shift_offsets_output(self):
        """Setting shift to a constant should shift the output mean by that constant."""
        ln = LayerNorm(emb_dim=EMB_DIM)
        with torch.no_grad():
            ln.shift.fill_(2.0)
        x = torch.randn(8, EMB_DIM)
        out = ln(x)
        assert torch.allclose(out.mean(dim=-1), torch.full((8,), 2.0), atol=1e-5)

    def test_scale_zero_collapses_to_shift(self):
        """With scale=0 and shift=c, output should be identically c."""
        ln = LayerNorm(emb_dim=EMB_DIM)
        with torch.no_grad():
            ln.scale.fill_(0.0)
            ln.shift.fill_(3.0)
        x = torch.randn(4, EMB_DIM)
        out = ln(x)
        assert torch.allclose(out, torch.full_like(out, 3.0), atol=1e-6)


# ---------------------------------------------------------------------------
# Learnable parameters
# ---------------------------------------------------------------------------


class TestLayerNormParameters:
    def test_scale_initial_value(self, ln):
        assert torch.allclose(ln.scale, torch.ones(EMB_DIM))

    def test_shift_initial_value(self, ln):
        assert torch.allclose(ln.shift, torch.zeros(EMB_DIM))

    def test_scale_is_parameter(self, ln):
        assert isinstance(ln.scale, torch.nn.Parameter)

    def test_shift_is_parameter(self, ln):
        assert isinstance(ln.shift, torch.nn.Parameter)


# ---------------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------------


class TestLayerNormGradients:
    def test_gradients_flow_through_scale_and_shift(self, ln):
        x = torch.randn(2, 5, EMB_DIM)
        ln(x).sum().backward()
        assert ln.scale.grad is not None
        assert ln.shift.grad is not None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestLayerNormFactory:
    def test_returns_layer_norm(self):
        module = get_layer_norm(emb_dim=EMB_DIM)
        assert isinstance(module, LayerNorm)

    def test_explicit_type(self):
        module = get_layer_norm("layer_norm", emb_dim=EMB_DIM)
        assert isinstance(module, LayerNorm)

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown norm type"):
            get_layer_norm("unknown", emb_dim=EMB_DIM)  # type: ignore[arg-type]

    def test_factory_output_shape(self):
        ln = get_layer_norm(emb_dim=EMB_DIM)
        x = torch.randn(3, 6, EMB_DIM)
        assert ln(x).shape == (3, 6, EMB_DIM)

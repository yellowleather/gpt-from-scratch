"""activation/test_gelu.py"""

import torch
import pytest

from activation.gelu import GELU
from activation.activation_factory import get_activation


@pytest.fixture
def gelu():
    return GELU()


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------


class TestGELUOutputShape:
    def test_3d_input(self, gelu):
        x = torch.randn(4, 10, 8)
        assert gelu(x).shape == (4, 10, 8)

    def test_1d_input(self, gelu):
        x = torch.randn(16)
        assert gelu(x).shape == (16,)

    def test_scalar_zero(self, gelu):
        x = torch.tensor(0.0)
        assert gelu(x).shape == ()


# ---------------------------------------------------------------------------
# Numerical behaviour
# ---------------------------------------------------------------------------


class TestGELUBehaviour:
    def test_zero_maps_to_zero(self, gelu):
        assert torch.allclose(gelu(torch.tensor(0.0)), torch.tensor(0.0), atol=1e-6)

    def test_large_positive_approaches_identity(self, gelu):
        """For very large x, GELU(x) ≈ x."""
        x = torch.tensor(10.0)
        assert torch.allclose(gelu(x), x, atol=1e-3)

    def test_large_negative_approaches_zero(self, gelu):
        """For very negative x, GELU(x) ≈ 0."""
        x = torch.tensor(-10.0)
        assert torch.allclose(gelu(x), torch.tensor(0.0), atol=1e-4)

    def test_minimum_is_bounded(self, gelu):
        """GELU has a minimum around -0.17; output should never go below -0.2."""
        x = torch.linspace(-5, 5, 1000)
        assert gelu(x).min().item() > -0.2

    def test_matches_torch_gelu_approx(self, gelu):
        """Our implementation should match PyTorch's tanh approximation."""
        torch_gelu = torch.nn.GELU(approximate="tanh")
        x = torch.linspace(-3, 3, 100)
        assert torch.allclose(gelu(x), torch_gelu(x), atol=1e-5)


# ---------------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------------


class TestGELUGradients:
    def test_gradients_flow(self, gelu):
        x = torch.randn(4, 8, requires_grad=True)
        gelu(x).sum().backward()
        assert x.grad is not None

    def test_grad_at_zero_is_half(self, gelu):
        """d/dx GELU(x) at x=0 ≈ 0.5."""
        x = torch.tensor(0.0, requires_grad=True)
        gelu(x).backward()
        assert torch.allclose(x.grad, torch.tensor(0.5), atol=1e-4)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestActivationFactory:
    def test_returns_gelu(self):
        module = get_activation("gelu")
        assert isinstance(module, GELU)

    def test_default_type_is_gelu(self):
        module = get_activation()
        assert isinstance(module, GELU)

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown activation type"):
            get_activation("relu")  # type: ignore[arg-type]

    def test_factory_output_shape(self):
        act = get_activation()
        x = torch.randn(3, 6, 8)
        assert act(x).shape == (3, 6, 8)

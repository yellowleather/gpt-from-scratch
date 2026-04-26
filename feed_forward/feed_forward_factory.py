"""feed_forward/feed_forward_factory.py

Factory function for creating feed-forward modules.
"""

from typing import Literal

from torch import nn

from feed_forward.feed_forward import FeedForward


def get_feed_forward(
    ff_type: Literal["feed_forward"] = "feed_forward",
    *,
    emb_dim: int,
    ff_expansion: int = 4,
) -> nn.Module:
    """Return a feed-forward module by name.

    Args:
        ff_type: Implementation to instantiate. Currently only "feed_forward" is supported.
        emb_dim: Input and output embedding dimension.
        ff_expansion: Hidden layer expansion ratio. Defaults to 4 (GPT-2 / original Transformer).

    Returns:
        A feed-forward module.

    Raises:
        ValueError: If ff_type is not recognised.
    """
    if ff_type == "feed_forward":
        return FeedForward(emb_dim=emb_dim, ff_expansion=ff_expansion)

    raise ValueError(
        f"Unknown feed-forward type '{ff_type}'. Supported types: feed_forward"
    )


__all__ = ["get_feed_forward"]

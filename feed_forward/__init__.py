"""feed_forward package

Provides factory helpers and implementations for position-wise feed-forward modules.
"""

from feed_forward.feed_forward import FeedForward
from feed_forward.feed_forward_factory import get_feed_forward

__all__ = ["FeedForward", "get_feed_forward"]

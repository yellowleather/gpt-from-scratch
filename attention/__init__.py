"""attention package

Provides factory helpers and implementations for attention modules used in the language model pipeline.
"""

from attention.attention_factory import get_attention
from attention.causal_self_attention import CausalSelfAttention
from attention.multi_head_attention import MultiHeadAttentionWrapper

__all__ = ["get_attention", "CausalSelfAttention", "MultiHeadAttentionWrapper"]

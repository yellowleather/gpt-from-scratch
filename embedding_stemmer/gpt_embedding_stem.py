"""embedding_stemmer/gpt_embedding_stem.py

Implementation of the GPT-style token and positional embedding stem.
"""

import torch
from torch import nn


class GPTEmbeddingStem(nn.Module):
    """Token and positional embedding stem for GPT-style language models."""

    def __init__(self, vocab_size: int, embedding_dim: int, context_length: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(context_length, embedding_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape

        # Build a position index tensor [0, 1, ..., seq_len-1] and broadcast
        # it across the batch dimension so every sequence in the batch gets
        # the same positional indices.  Keeping it on the same device as
        # input_ids ensures compatibility with both CPU and GPU training.
        positions = torch.arange(seq_len, device=input_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size, seq_len)

        # Look up a learned vector for each token id.
        # Shape: (batch_size, seq_len, embedding_dim).
        token_embeddings = self.token_embedding(input_ids)

        # Look up a learned vector for each absolute position (0 … seq_len-1).
        # These capture order information that the token embeddings alone cannot,
        # since the same token at different positions should carry different context.
        # Shape: (batch_size, seq_len, embedding_dim).
        position_embeddings = self.position_embedding(positions)

        # Element-wise addition fuses token identity and position into a single
        # vector per token.  This combined representation is what gets passed
        # into the attention layers.
        return token_embeddings + position_embeddings


__all__ = ["GPTEmbeddingStem"]

"""embedding_stemmer/gpt_embedding_stem.py

Implementation of the GPT-style token and positional embedding stem.
"""

import torch
from torch import nn


class GPTEmbeddingStem(nn.Module):
    """Token and positional embedding stem for GPT-style language models.

    Complexity notation used throughout this file:
        B     = batch size
        T     = seq_len (up to context_length)
        D     = embedding_dim
        V     = vocab_size

    Parameters (weights):
        token_embedding weight    : (V, D) — V D floats  (e.g. 50257 × 256 ≈ 12.9 M)
        position_embedding weight : (T, D) — T D floats  (e.g.   128 × 256 ≈  32 K)
        Total                     : (V + T) D — dominated entirely by the token table

    Activation memory during forward:
        token_embeddings    : (B, T, D) — B T D floats
        position_embeddings : (B, T, D) — B T D floats  (held simultaneously before add)
        output              : (B, T, D) — B T D floats
        Peak                : 2 B T D  (the two lookup results before the addition)

    FLOPs per forward pass (arithmetic ops — the ML convention):
        Token lookup    : B T reads  (one per token ID in the (B, T) input)
        Position lookup : B T reads  (indices 0…T-1 repeated for each batch item;
                          only T unique rows touched in the weight table)
        Both lookups are pure indexed memory reads — 0 multiply-adds each.
        Element-wise addition: B T D additions  ← the only arithmetic in this module.
        Total multiply-adds  : B T D  (entirely from the addition; if there were no
                               positional embedding, this would be 0)

        Note — memory bandwidth cost (not captured by FLOPs):
        Unlike sorting or RAM-model analysis, ML FLOP counts exclude memory reads.
        The lookups transfer B T D floats from the weight table to registers, which
        is real work. For large vocabularies (e.g. V=50257, D=768 → ~150 MB token
        table) this can be the actual runtime bottleneck — a memory-bandwidth limit
        rather than a compute limit. FLOPs measure the compute bottleneck only;
        profiling tools (e.g. torch.profiler) are needed to surface bandwidth costs.

    Backward pass note:
        Gradients are sparse — only the B T rows that were actually read receive
        an update each step (out of V rows in the token table). PyTorch represents
        these as sparse tensors, keeping the per-step update cost proportional to
        B T D regardless of vocabulary size.
    """

    def __init__(self, vocab_size: int, embedding_dim: int, context_length: int):
        super().__init__()
        # Weight matrix: (V, D) — the dominant parameter cost of the stem.
        # Each of the V vocabulary tokens gets its own learned D-dimensional vector.
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)

        # Weight matrix: (context_length, D) — tiny compared to the token table.
        # One learned vector per absolute position 0 … context_length-1.
        self.position_embedding = nn.Embedding(context_length, embedding_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape

        # Pure table lookup: each of the B×T integer token IDs indexes one row
        # of the (V, D) weight matrix.  No arithmetic — just indexed memory reads.
        # Memory: (B, T, D) — B T D floats materialised from the weight table.
        # FLOPs:  0 multiply-adds.
        token_embeddings = self.token_embedding(input_ids)

        # Build position indices [0, 1, …, seq_len-1] and broadcast across batch.
        # Keeping on the same device as input_ids supports both CPU and GPU.
        positions = torch.arange(seq_len, device=input_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size, seq_len)

        # Same indexed-read pattern as above but into the (context_length, D) table.
        # Captures order: the same token ID at position 3 vs. position 7 will sum
        # with different position vectors, giving the model a sense of sequence order
        # that attention alone (being a set operation) cannot provide.
        # Memory: (B, T, D) — held simultaneously with token_embeddings until the add.
        # FLOPs:  0 multiply-adds.
        position_embeddings = self.position_embedding(positions)

        # Fuse token identity and position into one vector per token.
        # Memory: (B, T, D) output — token_embeddings and position_embeddings can be
        #         freed immediately after this line.
        # FLOPs:  B T D additions — the only arithmetic in the entire forward pass.
        return token_embeddings + position_embeddings


__all__ = ["GPTEmbeddingStem"]

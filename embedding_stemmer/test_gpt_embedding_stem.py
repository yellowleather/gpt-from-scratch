"""tests/test_gpt_embedding_stem.py"""

import torch
from torch import nn
import pytest
from embedding_stemmer.gpt_embedding_stem import GPTEmbeddingStem


VOCAB = 50
DIM = 8
CTX = 16


@pytest.fixture
def stem():
    return GPTEmbeddingStem(vocab_size=VOCAB, embedding_dim=DIM, context_length=CTX)


class TestOutputShape:
    def test_shape_batch_seq_dim(self, stem):
        inp = torch.randint(0, VOCAB, (4, 10))
        out = stem(inp)
        assert out.shape == (4, 10, DIM)

    def test_single_sample(self, stem):
        inp = torch.randint(0, VOCAB, (1, 6))
        out = stem(inp)
        assert out.shape == (1, 6, DIM)

    def test_seq_len_equals_context_length(self, stem):
        inp = torch.randint(0, VOCAB, (2, CTX))
        out = stem(inp)
        assert out.shape == (2, CTX, DIM)


class TestEmbeddingArithmetic:
    def test_output_is_sum_of_token_and_position_embeddings(self):
        stem = GPTEmbeddingStem(vocab_size=VOCAB, embedding_dim=DIM, context_length=CTX)
        nn.init.constant_(stem.token_embedding.weight, 1.0)
        nn.init.constant_(stem.position_embedding.weight, 2.0)
        inp = torch.zeros(1, 4, dtype=torch.long)
        out = stem(inp)
        # Every element should be 1.0 + 2.0 = 3.0
        assert torch.allclose(out, torch.full((1, 4, DIM), 3.0))

    def test_different_token_ids_produce_different_outputs(self, stem):
        inp_a = torch.zeros(1, 4, dtype=torch.long)
        inp_b = torch.ones(1, 4, dtype=torch.long)
        out_a = stem(inp_a)
        out_b = stem(inp_b)
        assert not torch.allclose(out_a, out_b)

    def test_same_token_id_produces_same_token_embedding_regardless_of_position(self):
        stem = GPTEmbeddingStem(vocab_size=VOCAB, embedding_dim=DIM, context_length=CTX)
        # Zero out position embeddings so only token embeddings contribute
        nn.init.constant_(stem.position_embedding.weight, 0.0)
        # Same token id at two different positions should yield the same vector
        inp = torch.tensor([[3, 3]])  # token 3 at positions 0 and 1
        out = stem(inp)
        assert torch.allclose(out[0, 0], out[0, 1])


class TestPositions:
    def test_positions_run_from_zero_to_seq_len(self):
        stem = GPTEmbeddingStem(vocab_size=VOCAB, embedding_dim=DIM, context_length=CTX)
        # Zero out token embeddings; position embeddings carry the signal
        nn.init.constant_(stem.token_embedding.weight, 0.0)
        # Each position row = its index as a constant vector for easy verification
        with torch.no_grad():
            for i in range(CTX):
                stem.position_embedding.weight[i] = float(i)

        inp = torch.zeros(1, 5, dtype=torch.long)
        out = stem(inp)
        # Position 0 → 0.0, position 1 → 1.0, ...
        for pos in range(5):
            assert torch.allclose(out[0, pos], torch.full((DIM,), float(pos)))

    def test_positions_are_same_across_batch(self):
        stem = GPTEmbeddingStem(vocab_size=VOCAB, embedding_dim=DIM, context_length=CTX)
        nn.init.constant_(stem.token_embedding.weight, 0.0)
        inp = torch.zeros(3, 6, dtype=torch.long)
        out = stem(inp)
        # All three batch items have the same token ids and so the same output
        assert torch.allclose(out[0], out[1])
        assert torch.allclose(out[1], out[2])


class TestGradientFlow:
    def test_output_is_differentiable(self, stem):
        inp = torch.randint(0, VOCAB, (2, 5))
        out = stem(inp)
        loss = out.sum()
        loss.backward()
        assert stem.token_embedding.weight.grad is not None
        assert stem.position_embedding.weight.grad is not None

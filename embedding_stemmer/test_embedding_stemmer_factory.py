"""tests/test_embedding_stemmer_factory.py"""

import pytest
from embedding_stemmer.embedding_stemmer_factory import get_embedding_stem
from embedding_stemmer.gpt_embedding_stem import GPTEmbeddingStem


def test_gpt_stem_returns_gpt_embedding_stem():
    stem = get_embedding_stem(stem_type="gpt", vocab_size=100, embedding_dim=16, context_length=32)
    assert isinstance(stem, GPTEmbeddingStem)


def test_default_stem_type_is_gpt():
    stem = get_embedding_stem(vocab_size=100, embedding_dim=16, context_length=32)
    assert isinstance(stem, GPTEmbeddingStem)


def test_unknown_stem_type_raises_value_error():
    with pytest.raises(ValueError, match="Unknown embedding stem"):
        get_embedding_stem(stem_type="nonexistent", vocab_size=100, embedding_dim=16, context_length=32)


def test_passes_vocab_size_and_embedding_dim():
    stem = get_embedding_stem(vocab_size=200, embedding_dim=32, context_length=64)
    assert stem.token_embedding.num_embeddings == 200
    assert stem.token_embedding.embedding_dim == 32


def test_passes_context_length():
    stem = get_embedding_stem(vocab_size=100, embedding_dim=16, context_length=48)
    assert stem.position_embedding.num_embeddings == 48

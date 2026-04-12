"""tests/test_tokenizer_factory.py"""

import pytest
from tokenizer.tokenizer_factory import get_tokenizer


def test_returns_object_with_encode_and_decode():
    enc = get_tokenizer()
    assert callable(enc.encode)
    assert callable(enc.decode)


def test_returned_encoding_has_name():
    enc = get_tokenizer()
    assert hasattr(enc, "name")
    assert enc.name == "gpt2"


def test_encode_returns_list_of_ints():
    enc = get_tokenizer()
    tokens = enc.encode("hello world")
    assert isinstance(tokens, list)
    assert all(isinstance(t, int) for t in tokens)


def test_round_trip():
    enc = get_tokenizer()
    text = "The quick brown fox"
    assert enc.decode(enc.encode(text)) == text


def test_invalid_encoding_raises_runtime_error():
    with pytest.raises(RuntimeError, match="failed to obtain tokenizer encoding"):
        get_tokenizer("this_encoding_does_not_exist_xyz")

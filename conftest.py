"""tests/conftest.py

Shared fixtures for all test modules.
"""

import pytest


class MockTokenizer:
    """Minimal tokenizer: maps each character to its index in the string.

    Produces one token per character, making window/stride math trivial to verify.
    """

    def encode(self, text: str) -> list[int]:
        return list(range(len(text)))

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t + 32) for t in tokens)


@pytest.fixture
def mock_tokenizer() -> MockTokenizer:
    return MockTokenizer()


@pytest.fixture
def sample_text() -> str:
    """A text long enough for dataset tests (max_length=4, stride=2 etc.)."""
    return "abcdefghijklmnopqrstuvwxyz"  # 26 chars → 26 tokens

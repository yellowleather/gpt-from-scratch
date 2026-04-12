"""tests/test_provider_factory.py"""

import pytest
from training_data_provider.training_data_provider_factory import get_provider
from training_data_provider.simple_text_provider import SimpleTextProvider


def test_simple_text_returns_simple_text_provider():
    provider = get_provider(provider_type="simple_text", url="http://example.com/text.txt")
    assert isinstance(provider, SimpleTextProvider)


def test_simple_text_passes_params():
    provider = get_provider(
        provider_type="simple_text",
        url="http://example.com/text.txt",
        cache_path="/tmp/cache.txt",
        timeout=10,
        use_cache=False,
    )
    assert provider.url == "http://example.com/text.txt"
    assert provider.cache_path == "/tmp/cache.txt"
    assert provider.timeout == 10
    assert provider.use_cache is False


def test_missing_url_raises_value_error():
    with pytest.raises(ValueError, match="url parameter is required"):
        get_provider(provider_type="simple_text")


def test_unknown_type_raises_value_error():
    with pytest.raises(ValueError, match="Unknown provider_type"):
        get_provider(provider_type="nonexistent", url="http://example.com")

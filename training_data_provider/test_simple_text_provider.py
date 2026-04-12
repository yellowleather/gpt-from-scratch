"""tests/test_simple_text_provider.py"""

import os
from io import BytesIO
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError

import pytest

from training_data_provider.simple_text_provider import SimpleTextProvider


def _make_mock_response(content: bytes, charset: str | None = "utf-8") -> MagicMock:
    resp = MagicMock()
    resp.read.return_value = content
    resp.headers.get_content_charset.return_value = charset
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


class TestCacheBehaviour:
    def test_reads_from_cache_when_exists(self, tmp_path):
        cache = tmp_path / "cache.txt"
        cache.write_text("cached content", encoding="utf-8")
        provider = SimpleTextProvider(url="http://unused", cache_path=str(cache), use_cache=True)
        with patch("training_data_provider.simple_text_provider.urlopen") as mock_urlopen:
            result = provider.get_text()
        mock_urlopen.assert_not_called()
        assert result == "cached content"

    def test_downloads_when_cache_missing(self, tmp_path):
        cache = tmp_path / "cache.txt"
        provider = SimpleTextProvider(url="http://example.com", cache_path=str(cache), use_cache=True)
        mock_resp = _make_mock_response(b"downloaded content")
        with patch("training_data_provider.simple_text_provider.urlopen", return_value=mock_resp):
            result = provider.get_text()
        assert result == "downloaded content"

    def test_use_cache_false_always_downloads(self, tmp_path):
        cache = tmp_path / "cache.txt"
        cache.write_text("stale cache", encoding="utf-8")
        provider = SimpleTextProvider(url="http://example.com", cache_path=str(cache), use_cache=False)
        mock_resp = _make_mock_response(b"fresh content")
        with patch("training_data_provider.simple_text_provider.urlopen", return_value=mock_resp):
            result = provider.get_text()
        assert result == "fresh content"

    def test_saves_to_cache_after_download(self, tmp_path):
        cache = tmp_path / "cache.txt"
        provider = SimpleTextProvider(url="http://example.com", cache_path=str(cache))
        mock_resp = _make_mock_response(b"some text")
        with patch("training_data_provider.simple_text_provider.urlopen", return_value=mock_resp):
            provider.get_text()
        assert cache.read_text(encoding="utf-8") == "some text"

    def test_no_cache_path_does_not_save(self, tmp_path):
        provider = SimpleTextProvider(url="http://example.com", cache_path=None)
        mock_resp = _make_mock_response(b"some text")
        with patch("training_data_provider.simple_text_provider.urlopen", return_value=mock_resp):
            result = provider.get_text()
        assert result == "some text"
        # No files created in cwd
        assert not any(tmp_path.iterdir())

    def test_creates_parent_directories_for_cache(self, tmp_path):
        cache = tmp_path / "nested" / "dir" / "cache.txt"
        provider = SimpleTextProvider(url="http://example.com", cache_path=str(cache))
        mock_resp = _make_mock_response(b"text")
        with patch("training_data_provider.simple_text_provider.urlopen", return_value=mock_resp):
            provider.get_text()
        assert cache.exists()


class TestErrorHandling:
    def test_http_error_propagates(self):
        provider = SimpleTextProvider(url="http://example.com")
        with patch(
            "training_data_provider.simple_text_provider.urlopen",
            side_effect=HTTPError(url=None, code=404, msg="Not Found", hdrs=None, fp=None),
        ):
            with pytest.raises(HTTPError):
                provider.get_text()

    def test_url_error_propagates(self):
        provider = SimpleTextProvider(url="http://bad-host.invalid")
        with patch(
            "training_data_provider.simple_text_provider.urlopen",
            side_effect=URLError("Name or service not known"),
        ):
            with pytest.raises(URLError):
                provider.get_text()


class TestEncodingHandling:
    def test_uses_charset_from_headers(self):
        provider = SimpleTextProvider(url="http://example.com")
        content = "héllo".encode("latin-1")
        mock_resp = _make_mock_response(content, charset="latin-1")
        with patch("training_data_provider.simple_text_provider.urlopen", return_value=mock_resp):
            result = provider.get_text()
        assert result == "héllo"

    def test_falls_back_to_utf8_when_no_charset(self):
        provider = SimpleTextProvider(url="http://example.com")
        mock_resp = _make_mock_response("hello world".encode("utf-8"), charset=None)
        with patch("training_data_provider.simple_text_provider.urlopen", return_value=mock_resp):
            result = provider.get_text()
        assert result == "hello world"

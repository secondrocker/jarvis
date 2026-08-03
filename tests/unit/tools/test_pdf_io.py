"""load_pdf_bytes 的单元测试（HTTP 下载用 monkeypatch，不联网）。"""

import httpx
import pytest

from agent_app.errors import AppError, ErrorCode
from agent_app.tools.pdf import io as pdf_io
from agent_app.tools.pdf.io import load_pdf_bytes


class _FakeResponse:
    def __init__(self, content: bytes) -> None:
        self.content = content

    def raise_for_status(self) -> None:
        return None


def test_load_pdf_bytes_downloads_url(pdf_bytes: bytes, monkeypatch) -> None:
    monkeypatch.setattr(pdf_io.httpx, "get", lambda url, timeout=None: _FakeResponse(pdf_bytes))

    assert load_pdf_bytes("https://example.com/sample.pdf") == pdf_bytes


def test_load_pdf_bytes_rejects_http_error(monkeypatch) -> None:
    def _raise(*_args, **_kwargs):
        raise httpx.ConnectError("boom")

    monkeypatch.setattr(pdf_io.httpx, "get", _raise)

    with pytest.raises(AppError) as error:
        load_pdf_bytes("https://example.com/sample.pdf")

    assert error.value.code is ErrorCode.INVALID_PARAMETERS
    assert error.value.public_message == "PDF source could not be loaded"

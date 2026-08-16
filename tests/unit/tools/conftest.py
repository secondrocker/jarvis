"""tools/pdf 测试共享 fixture（PDF 由 PyMuPDF 现场生成，不联网）。"""

import pymupdf
import pytest
from pydantic import SecretStr

from agent_app.config import Settings
from agent_app.tools.pdf import io as pdf_io


class _FakeResponse:
    """httpx.get 的最小替身，返回预设字节。"""

    def __init__(self, content: bytes) -> None:
        self.content = content

    def raise_for_status(self) -> None:
        return None


@pytest.fixture
def pdf_bytes() -> bytes:
    """生成 3 页 PDF 的原始字节。"""
    document = pymupdf.open()
    for index in range(3):
        document.new_page().insert_text((72, 72), f"Page {index + 1}")
    data = document.tobytes()
    document.close()
    return data


@pytest.fixture
def pdf_source_url(pdf_bytes: bytes, monkeypatch) -> str:
    """patch httpx.get 返回预构造 PDF 字节，返回可用 source URL。"""
    monkeypatch.setattr(
        pdf_io.httpx,
        "get",
        lambda url, timeout=None: _FakeResponse(pdf_bytes),
    )
    return "https://example.com/sample.pdf"


@pytest.fixture
def test_settings():
    """创建 tools 测试使用的最小应用配置。

    返回值:
        使用虚假凭据且不会主动联网的配置。
    """
    return Settings.model_validate(
        {"openai": {"api_key": SecretStr("test-key"), "model": "gpt-4o-mini"}}
    )

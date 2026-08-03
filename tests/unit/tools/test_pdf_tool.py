"""render_pdf_to_image 进程内核心的单元测试（不联网）。

验证 workflow 路径的错误语义：失败时直接抛 ``AppError``（不经 ToolError），
以便 ``WorkflowExecutor`` 保留稳定错误码与 HTTP 映射。
"""

import base64
from pathlib import Path

import pytest

from agent_app.errors import AppError, ErrorCode
from agent_app.tools.pdf import io as pdf_io
from agent_app.tools.pdf.persist import STATIC_URL_PREFIX
from agent_app.tools.pdf.tool import render_pdf_to_image


class _FakeResponse:
    def __init__(self, content: bytes) -> None:
        self.content = content

    def raise_for_status(self) -> None:
        return None


def test_url_mode_writes_files_and_returns_urls(pdf_source_url: str, tmp_path: Path) -> None:
    result = render_pdf_to_image(output_dir=tmp_path, source=pdf_source_url, return_mode="url")

    assert result["page_count"] == 3
    assert [img["page"] for img in result["images"]] == [1]
    url = result["images"][0]["url"]
    assert url.startswith(f"{STATIC_URL_PREFIX}/")
    relative = url.split(f"{STATIC_URL_PREFIX}/", 1)[1]
    assert (tmp_path / relative).exists()


def test_base64_mode_returns_png_magic(pdf_source_url: str, tmp_path: Path) -> None:
    result = render_pdf_to_image(output_dir=tmp_path, source=pdf_source_url, return_mode="base64")

    img = result["images"][0]
    assert set(img.keys()) == {"page", "data", "width", "height"}
    assert base64.b64decode(img["data"])[:8] == b"\x89PNG\r\n\x1a\n"


def test_unions_pages_and_ranges(pdf_source_url: str, tmp_path: Path) -> None:
    result = render_pdf_to_image(
        output_dir=tmp_path,
        source=pdf_source_url,
        pages=[1],
        page_ranges=[[2, 3]],
        return_mode="base64",
    )

    assert [img["page"] for img in result["images"]] == [1, 2, 3]


def test_rejects_out_of_range(pdf_source_url: str, tmp_path: Path) -> None:
    with pytest.raises(AppError) as error:
        render_pdf_to_image(output_dir=tmp_path, source=pdf_source_url, pages=[99])

    assert error.value.code is ErrorCode.INVALID_PARAMETERS
    assert error.value.public_message == "Requested pages out of range"


def test_rejects_invalid_pdf(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(pdf_io.httpx, "get", lambda url, timeout=None: _FakeResponse(b"not a pdf"))

    with pytest.raises(AppError) as error:
        render_pdf_to_image(output_dir=tmp_path, source="https://example.com/bad.pdf")

    assert error.value.code is ErrorCode.INVALID_PARAMETERS
    assert error.value.public_message == "PDF source could not be loaded"

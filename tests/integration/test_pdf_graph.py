"""PDF 转图片子图的集成测试（不访问网络，PDF 由 PyMuPDF 现场生成）。"""

import pymupdf
import pytest
from fakes import FakeObjectStorage

from agent_app.errors import AppError, ErrorCode
from agent_app.tools.pdf import io as pdf_io
from agent_app.workflows.pdf_to_image import build_pdf_to_image_graph


class _FakeResponse:
    """httpx.get 的最小替身，返回预设字节。"""

    def __init__(self, content: bytes) -> None:
        self.content = content

    def raise_for_status(self) -> None:
        return None


@pytest.fixture
def pdf_source_url(monkeypatch) -> str:
    """生成 3 页 PDF，patch httpx.get 返回其字节，返回 source URL。"""
    document = pymupdf.open()
    for index in range(3):
        document.new_page().insert_text((72, 72), f"Page {index + 1}")
    content = document.tobytes()
    document.close()
    monkeypatch.setattr(pdf_io.httpx, "get", lambda url, timeout=None: _FakeResponse(content))
    return "https://example.com/sample.pdf"


@pytest.mark.asyncio
async def test_render_defaults_to_first_page(pdf_source_url) -> None:
    graph = build_pdf_to_image_graph(storage=FakeObjectStorage())

    output = await graph.ainvoke({"url": pdf_source_url})

    result = output["result"]
    assert [image["page"] for image in result["images"]] == [1]
    assert result["page_count"] == 3
    assert result["rendered_pages"] == [1]
    url = result["images"][0]["url"]
    assert url.startswith("https://fake-s3.test/pdf_images/")
    assert url.endswith("/page_1.png")


@pytest.mark.asyncio
async def test_render_specific_discrete_pages(pdf_source_url) -> None:
    graph = build_pdf_to_image_graph(storage=FakeObjectStorage())

    output = await graph.ainvoke({"url": pdf_source_url, "pages": [1, 3]})

    assert [image["page"] for image in output["result"]["images"]] == [1, 3]
    assert all(image["url"].endswith(".png") for image in output["result"]["images"])


@pytest.mark.asyncio
async def test_render_unions_pages_and_ranges(pdf_source_url) -> None:
    graph = build_pdf_to_image_graph(storage=FakeObjectStorage())

    output = await graph.ainvoke({"url": pdf_source_url, "pages": [1], "page_ranges": [[2, 3]]})

    assert [image["page"] for image in output["result"]["images"]] == [1, 2, 3]


@pytest.mark.asyncio
async def test_render_supports_jpeg_format(pdf_source_url) -> None:
    graph = build_pdf_to_image_graph(storage=FakeObjectStorage())

    output = await graph.ainvoke({"url": pdf_source_url, "image_format": "jpeg"})

    assert output["result"]["images"][0]["url"].endswith("/page_1.jpeg")


@pytest.mark.asyncio
async def test_render_respects_custom_dpi(pdf_source_url) -> None:
    graph = build_pdf_to_image_graph(storage=FakeObjectStorage())

    output = await graph.ainvoke({"url": pdf_source_url, "dpi": 300})

    image = output["result"]["images"][0]
    # 300 DPI 下标准 A4 页宽约为 2480 像素，至少应大于默认 150 DPI 的宽度。
    assert image["width"] > 1500


@pytest.mark.asyncio
async def test_render_rejects_out_of_range_pages(pdf_source_url) -> None:
    graph = build_pdf_to_image_graph(storage=FakeObjectStorage())

    with pytest.raises(AppError) as error:
        await graph.ainvoke({"url": pdf_source_url, "pages": [99]})

    assert error.value.code is ErrorCode.INVALID_PARAMETERS
    assert error.value.public_message == "Requested pages out of range"


@pytest.mark.asyncio
async def test_render_rejects_non_pdf_payload(monkeypatch) -> None:
    monkeypatch.setattr(pdf_io.httpx, "get", lambda url, timeout=None: _FakeResponse(b"not a pdf"))
    graph = build_pdf_to_image_graph(storage=FakeObjectStorage())

    with pytest.raises(AppError) as error:
        await graph.ainvoke({"url": "https://example.com/bad.pdf"})

    assert error.value.code is ErrorCode.INVALID_PARAMETERS
    assert error.value.public_message == "PDF could not be loaded"

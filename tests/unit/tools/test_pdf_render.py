"""render_pdf_pages 纯核心的单元测试（不联网）。"""

import pytest

from agent_app.errors import AppError, ErrorCode
from agent_app.tools.pdf.render import render_pdf_pages


def test_render_pdf_pages_defaults_to_first_page(pdf_bytes: bytes) -> None:
    rendered, page_count = render_pdf_pages(pdf_bytes)

    assert [item.page for item in rendered] == [1]
    assert page_count == 3
    assert rendered[0].width > 0
    assert rendered[0].height > 0


def test_render_pdf_pages_unions_pages_and_ranges(pdf_bytes: bytes) -> None:
    rendered, _ = render_pdf_pages(pdf_bytes, pages=[1], page_ranges=[[2, 3]])

    assert [item.page for item in rendered] == [1, 2, 3]


def test_render_pdf_pages_respects_dpi(pdf_bytes: bytes) -> None:
    low, _ = render_pdf_pages(pdf_bytes, dpi=72)
    high, _ = render_pdf_pages(pdf_bytes, dpi=300)

    assert high[0].width > low[0].width


def test_render_pdf_pages_supports_jpeg(pdf_bytes: bytes) -> None:
    rendered, _ = render_pdf_pages(pdf_bytes, image_format="jpeg")

    # JPEG 文件头 magic bytes。
    assert rendered[0].image_bytes[:3] == b"\xff\xd8\xff"


def test_render_pdf_pages_rejects_invalid_pdf() -> None:
    with pytest.raises(AppError) as error:
        render_pdf_pages(b"not a pdf")

    assert error.value.code is ErrorCode.INVALID_PARAMETERS
    assert error.value.public_message == "PDF source could not be loaded"


def test_render_pdf_pages_rejects_out_of_range(pdf_bytes: bytes) -> None:
    with pytest.raises(AppError) as error:
        render_pdf_pages(pdf_bytes, pages=[99])

    assert error.value.code is ErrorCode.INVALID_PARAMETERS
    assert error.value.public_message == "Requested pages out of range"

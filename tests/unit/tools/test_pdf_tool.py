"""render_pdf_to_image 进程内核心的单元测试（不联网）。

验证 workflow 路径的错误语义：失败时直接抛 ``AppError``（不经 ToolError），
以便 ``WorkflowExecutor`` 保留稳定错误码与 HTTP 映射。
"""

import pytest
from fakes import FakeObjectStorage

from agent_app.errors import AppError, ErrorCode
from agent_app.tools.pdf.tool import render_pdf_to_image


def test_returns_uploader_urls_for_default_page(pdf_bytes: bytes) -> None:
    storage = FakeObjectStorage()
    result = render_pdf_to_image(storage=storage, pdf_bytes=pdf_bytes)

    assert result["page_count"] == 3
    assert [img["page"] for img in result["images"]] == [1]
    url = result["images"][0]["url"]
    assert url.startswith("https://fake-s3.test/pdf_images/")
    assert url.endswith("/page_1.png")
    assert storage.uploads[0][1] == "image/png"


def test_unions_pages_and_ranges(pdf_bytes: bytes) -> None:
    result = render_pdf_to_image(
        storage=FakeObjectStorage(),
        pdf_bytes=pdf_bytes,
        pages=[1],
        page_ranges=[[2, 3]],
    )

    assert [img["page"] for img in result["images"]] == [1, 2, 3]


def test_rejects_out_of_range(pdf_bytes: bytes) -> None:
    with pytest.raises(AppError) as error:
        render_pdf_to_image(storage=FakeObjectStorage(), pdf_bytes=pdf_bytes, pages=[99])

    assert error.value.code is ErrorCode.INVALID_PARAMETERS
    assert error.value.public_message == "Requested pages out of range"


def test_rejects_invalid_pdf() -> None:
    with pytest.raises(AppError) as error:
        render_pdf_to_image(storage=FakeObjectStorage(), pdf_bytes=b"not a pdf")

    assert error.value.code is ErrorCode.INVALID_PARAMETERS
    assert error.value.public_message == "PDF could not be loaded"

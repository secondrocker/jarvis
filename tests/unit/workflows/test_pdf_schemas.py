"""PDF 转图片工作流输入模型的校验测试。"""

import pytest
from pydantic import ValidationError

from agent_app.workflows.pdf_to_image.schemas import PdfInput


def test_pdf_input_accepts_url_source() -> None:
    value = PdfInput(source="https://example.com/sample.pdf")

    assert value.source == "https://example.com/sample.pdf"
    assert value.pages is None
    assert value.page_ranges is None
    assert value.dpi == 150
    assert value.image_format == "png"


def test_pdf_input_strips_source() -> None:
    value = PdfInput(source="  https://example.com/sample.pdf  ")

    assert value.source == "https://example.com/sample.pdf"


def test_pdf_input_rejects_missing_source() -> None:
    with pytest.raises(ValidationError):
        PdfInput()


def test_pdf_input_rejects_blank_source() -> None:
    with pytest.raises(ValidationError):
        PdfInput(source="   ")


def test_pdf_input_accepts_pages_and_ranges_together() -> None:
    value = PdfInput(source="https://example.com/sample.pdf", pages=[1, 3], page_ranges=[[5, 7]])

    assert value.pages == [1, 3]
    assert value.page_ranges == [[5, 7]]


def test_pdf_input_rejects_non_positive_pages() -> None:
    with pytest.raises(ValidationError):
        PdfInput(source="https://example.com/sample.pdf", pages=[0, 2])


@pytest.mark.parametrize("invalid_range", [[], [1], [1, 2, 3], [3, 1], [0, 2]])
def test_pdf_input_rejects_malformed_ranges(invalid_range: list[int]) -> None:
    with pytest.raises(ValidationError):
        PdfInput(source="https://example.com/sample.pdf", page_ranges=[invalid_range])


def test_pdf_input_enforces_dpi_bounds() -> None:
    assert PdfInput(source="https://example.com/sample.pdf", dpi=72).dpi == 72
    assert PdfInput(source="https://example.com/sample.pdf", dpi=600).dpi == 600

    with pytest.raises(ValidationError):
        PdfInput(source="https://example.com/sample.pdf", dpi=71)
    with pytest.raises(ValidationError):
        PdfInput(source="https://example.com/sample.pdf", dpi=601)


def test_pdf_input_rejects_unsupported_image_format() -> None:
    with pytest.raises(ValidationError):
        PdfInput(source="https://example.com/sample.pdf", image_format="gif")

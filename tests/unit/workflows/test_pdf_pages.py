"""页码解析纯函数的单元测试。"""

import pytest

from agent_app.errors import AppError, ErrorCode
from agent_app.workflows.pdf_to_image.pages import resolve_pages


def test_resolve_pages_defaults_to_first_page() -> None:
    assert resolve_pages(None, None, 3) == [0]


def test_resolve_pages_handles_discrete_pages() -> None:
    assert resolve_pages([1, 3], None, 3) == [0, 2]


def test_resolve_pages_deduplicates_and_sorts() -> None:
    assert resolve_pages([3, 3, 1], None, 3) == [0, 2]


def test_resolve_pages_handles_single_range() -> None:
    assert resolve_pages(None, [[1, 3]], 3) == [0, 1, 2]


def test_resolve_pages_handles_multiple_ranges() -> None:
    assert resolve_pages(None, [[1, 2], [5, 6]], 6) == [0, 1, 4, 5]


def test_resolve_pages_unions_pages_and_ranges() -> None:
    assert resolve_pages([1, 5], [[2, 3]], 5) == [0, 1, 2, 4]


def test_resolve_pages_rejects_out_of_range_discrete_page() -> None:
    with pytest.raises(AppError) as error:
        resolve_pages([4], None, 3)

    assert error.value.code is ErrorCode.INVALID_PARAMETERS
    assert error.value.public_message == "Requested pages out of range"


def test_resolve_pages_rejects_out_of_range_range() -> None:
    with pytest.raises(AppError) as error:
        resolve_pages(None, [[1, 5]], 3)

    assert error.value.code is ErrorCode.INVALID_PARAMETERS


def test_resolve_pages_rejects_empty_pdf() -> None:
    with pytest.raises(AppError) as error:
        resolve_pages(None, None, 0)

    assert error.value.code is ErrorCode.INVALID_PARAMETERS
    assert error.value.public_message == "PDF has no pages"

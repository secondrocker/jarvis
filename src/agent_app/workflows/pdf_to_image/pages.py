"""把结构化页码请求解析为去重升序的 0-based 页索引。"""

from agent_app.errors import AppError, ErrorCode


def resolve_pages(
    pages: list[int] | None,
    page_ranges: list[list[int]] | None,
    page_count: int,
) -> list[int]:
    """合并离散页码与多个闭区间，返回去重升序的 0-based 页索引。

    参数:
        pages: 1-based 离散固定页码列表，可为 None。
        page_ranges: 多个 1-based 闭区间 ``[start, end]``，可为 None。
        page_count: PDF 实际页数，用于越界校验。

    返回值:
        去重并升序排列的 0-based 页索引列表。

    异常:
        AppError: 页码越界或 PDF 没有页面时抛出 INVALID_PARAMETERS。
    """
    if page_count <= 0:
        raise AppError(ErrorCode.INVALID_PARAMETERS, "PDF has no pages")

    targets: set[int] = set()
    if pages:
        targets.update(pages)
    for start, end in page_ranges or []:
        targets.update(range(start, end + 1))

    # 两者皆空时默认渲染第一页。
    if not targets:
        targets.add(1)

    invalid = [page for page in targets if page < 1 or page > page_count]
    if invalid:
        raise AppError(ErrorCode.INVALID_PARAMETERS, "Requested pages out of range")

    return sorted(page - 1 for page in targets)

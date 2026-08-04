"""PDF 来源加载：从 URL 下载。"""

import httpx

from agent_app.errors import AppError, ErrorCode

# 下载远程 PDF 的最大等待秒数。
PDF_DOWNLOAD_TIMEOUT = 30.0


def load_pdf_bytes(url: str) -> bytes:
    """从 URL 下载 PDF 原始字节；失败时抛出安全的 INVALID_PARAMETERS。

    参数:
        url: 可下载的 PDF URL。

    返回值:
        可交给 PyMuPDF 打开的 PDF 原始字节。

    异常:
        AppError: 下载失败时抛出 INVALID_PARAMETERS。
    """
    try:
        response = httpx.get(url.strip(), timeout=PDF_DOWNLOAD_TIMEOUT)
        response.raise_for_status()
    except httpx.HTTPError as error:
        raise AppError(
            ErrorCode.INVALID_PARAMETERS,
            "PDF could not be loaded",
        ) from error
    return response.content

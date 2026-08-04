"""PDF 转图片能力的已校验输入契约（跨 workflow、MCP tool 与上传端点共用）。"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class PdfRenderOptions(BaseModel):
    """PDF 转图片的渲染选项（不含来源，供 multipart 文件上传端点复用）。

    页码可通过 ``pages``（离散固定页）与 ``page_ranges``（多个闭区间）同时指定，
    解析时取并集；两者皆空时默认渲染第一页。
    """

    pages: list[int] | None = None
    page_ranges: list[list[int]] | None = None
    dpi: int = Field(default=150, ge=72, le=600)
    image_format: Literal["png", "jpeg"] = "png"

    @field_validator("pages")
    @classmethod
    def require_positive_pages(cls, value: list[int] | None) -> list[int] | None:
        """确保每个离散页码都是不小于 1 的正整数。

        参数:
            value: 调用方指定的离散页码列表。

        返回值:
            校验通过的原始页码列表。
        """
        if value is None:
            return None
        if any(page < 1 for page in value):
            raise ValueError("page numbers must be >= 1")
        return value

    @field_validator("page_ranges")
    @classmethod
    def require_valid_ranges(cls, value: list[list[int]] | None) -> list[list[int]] | None:
        """确保每个区间长度为 2、元素不小于 1 且起始不大于结束。

        参数:
            value: 调用方指定的闭区间列表。

        返回值:
            校验通过的原始区间列表。
        """
        if value is None:
            return None
        for rng in value:
            if len(rng) != 2 or rng[0] < 1 or rng[1] < rng[0]:
                raise ValueError("each page range must be [start, end] with start <= end")
        return value


class PdfInput(PdfRenderOptions):
    """PDF 转图片接收的已校验输入（带可下载 URL）。

    ``url`` 为可下载的 PDF URL（唯一来源）。页码语义同 ``PdfRenderOptions``。
    """

    url: str = Field(min_length=1)

    @field_validator("url")
    @classmethod
    def strip_url(cls, value: str) -> str:
        """去除 URL 首尾空白；空白字符串视为非法。

        参数:
            value: 接口提交的 PDF 来源 URL。

        返回值:
            规范化后的来源字符串。

        异常:
            ValueError: 去除空白后为空时抛出。
        """
        stripped = value.strip()
        if not stripped:
            raise ValueError("url is required")
        return stripped

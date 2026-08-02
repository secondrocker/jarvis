"""PDF 转图片工作流的数据模型与状态。"""

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field, field_validator, model_validator


class PdfInput(BaseModel):
    """PDF 转图片工作流接收的已校验选项。

    来源二选一：``pdf_base64``（内联字节）优先，否则 ``source`` 视为可下载的 URL。
    页码可通过 ``pages``（离散固定页）与 ``page_ranges``（多个闭区间）同时指定，
    解析时取并集；两者皆空时默认渲染第一页。
    """

    source: str | None = None
    pdf_base64: str | None = None
    pages: list[int] | None = None
    page_ranges: list[list[int]] | None = None
    dpi: int = Field(default=150, ge=72, le=600)
    image_format: Literal["png", "jpeg"] = "png"

    @field_validator("source")
    @classmethod
    def strip_source(cls, value: str | None) -> str | None:
        """去除 URL 首尾空白；空字符串统一为 None。

        参数:
            value: 接口提交的 PDF 来源 URL。

        返回值:
            规范化后的来源字符串或 None。
        """
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

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

    @model_validator(mode="after")
    def require_source_or_base64(self) -> "PdfInput":
        """确保至少提供一种 PDF 来源。

        返回值:
            校验通过的当前输入。
        """
        if not self.source and not self.pdf_base64:
            raise ValueError("either source or pdf_base64 is required")
        return self


class PdfState(TypedDict, total=False):
    """在 PDF 转图片图中流转的状态。"""

    source: str | None
    pdf_base64: str | None
    pages: list[int] | None
    page_ranges: list[list[int]] | None
    dpi: int
    image_format: str
    result: dict[str, Any]

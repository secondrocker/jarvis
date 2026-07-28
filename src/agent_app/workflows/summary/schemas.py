"""结构化摘要工作流的数据模型与状态。"""

from typing import Any, TypedDict

from pydantic import BaseModel, Field, field_validator


class SummaryInput(BaseModel):
    """摘要工作流接收的已校验选项。"""

    text: str = Field(min_length=1)
    language: str | None = Field(default=None, max_length=32)
    max_words: int = Field(default=200, ge=50, le=1000)

    @field_validator("text")
    @classmethod
    def strip_and_require_text(cls, value: str) -> str:
        """去除首尾空白后，拒绝不包含实际内容的文本。

        参数:
            value: 待摘要的原始文本。

        返回值:
            去除首尾空白后的文本。
        """
        text = value.strip()
        if not text:
            raise ValueError("text must not be blank")
        return text


class SummaryResult(BaseModel):
    """摘要模型生成的结构化响应。"""

    summary: str = Field(min_length=1)
    key_points: list[str] = Field(min_length=1, max_length=10)


class SummaryState(TypedDict, total=False):
    """在固定摘要图中流转的状态。"""

    text: str
    language: str | None
    max_words: int
    normalized_text: str
    result: dict[str, Any]

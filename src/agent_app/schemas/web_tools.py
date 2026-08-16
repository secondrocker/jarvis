"""Web 网关 MCP 工具与 Agent 工具共用的已校验输入契约。"""

from pydantic import BaseModel, Field, field_validator


class WebSearchInput(BaseModel):
    """Web 搜索接收的已校验输入。"""

    query: str = Field(min_length=1)
    limit: int = Field(default=5, ge=1, le=10)

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        """去除查询首尾空白；空白字符串视为非法。"""
        stripped = value.strip()
        if not stripped:
            raise ValueError("query is required")
        return stripped


class WebFetchInput(BaseModel):
    """网页抓取接收的已校验输入。"""

    url: str = Field(min_length=1)
    max_chars: int = Field(default=20000, ge=1, le=50000)

    @field_validator("url")
    @classmethod
    def normalize_url(cls, value: str) -> str:
        """去除 URL 首尾空白；空白字符串视为非法。"""
        stripped = value.strip()
        if not stripped:
            raise ValueError("url is required")
        return stripped

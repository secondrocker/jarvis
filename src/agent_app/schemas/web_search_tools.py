"""网络搜索 / 网页阅读 MCP 工具的已校验输入契约。"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

# 搜索引擎编码（BigModel Web Search API 支持）。
SearchEngine = Literal["search_std", "search_pro", "search_pro_sogou", "search_pro_quark"]

# 搜索时间范围。
RecencyFilter = Literal["oneDay", "oneWeek", "oneMonth", "oneYear", "noLimit"]

# 网页阅读返回格式。
ReturnFormat = Literal["markdown", "text"]


class WebSearchInput(BaseModel):
    """web_search 接收的已校验输入。"""

    query: str = Field(min_length=1, max_length=70)
    count: int = Field(default=10, ge=1, le=50)
    search_engine: SearchEngine | None = None
    recency: RecencyFilter = "noLimit"
    domain: str | None = None

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        """去除首尾空白；空白或超长（API 上限 70 字符）视为非法。"""
        stripped = value.strip()
        if not stripped:
            raise ValueError("query is required")
        if len(stripped) > 70:
            raise ValueError("query must be at most 70 characters")
        return stripped

    @field_validator("domain")
    @classmethod
    def normalize_domain(cls, value: str | None) -> str | None:
        """去除空白；空白视为未提供。"""
        return value.strip() if value else None


class WebFetchInput(BaseModel):
    """web_fetch（网页阅读）接收的已校验输入。"""

    url: str = Field(min_length=1)
    return_format: ReturnFormat | None = None

    @field_validator("url")
    @classmethod
    def normalize_url(cls, value: str) -> str:
        """去除空白；必须是 http(s) 链接。"""
        stripped = value.strip()
        if not stripped.startswith(("http://", "https://")):
            raise ValueError("url must be an http(s) link")
        return stripped

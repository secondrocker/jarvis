"""网络搜索 / 网页阅读 MCP 工具的已校验输入契约。"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

# 搜索时间范围（web_search_prime 的 search_recency_filter）。
RecencyFilter = Literal["oneDay", "oneWeek", "oneMonth", "oneYear", "noLimit"]

# 搜索内容量（web_search_prime 的 content_size）。
ContentSize = Literal["medium", "high"]

# 搜索地域（web_search_prime 的 location）。
SearchLocation = Literal["cn", "us"]

# 网页阅读返回格式。
ReturnFormat = Literal["markdown", "text"]


class WebSearchInput(BaseModel):
    """web_search 接收的已校验输入。"""

    query: str = Field(min_length=1, max_length=70)
    count: int = Field(default=10, ge=1, le=50)
    recency: RecencyFilter = "noLimit"
    domain: str | None = None
    content_size: ContentSize | None = None
    location: SearchLocation | None = None

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        """去除首尾空白；空白或超长（API 上限 70 字符）视为非法。"""
        stripped = value.strip()
        if not stripped:
            raise ValueError("query 不能为空：请提供搜索关键词")
        if len(stripped) > 70:
            raise ValueError("query 过长：最多 70 字符，请精简为关键词或拆成多次调用")
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
            raise ValueError("url 必须是 http(s) 链接（以 http:// 或 https:// 开头）")
        return stripped

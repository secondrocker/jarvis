"""对象存储 MCP 工具的已校验输入契约。"""

from pydantic import BaseModel, Field, field_validator


class StorageUploadOptions(BaseModel):
    """对象存储上传的通用选项（content_type 与 key_prefix 规范化）。"""

    content_type: str = Field(min_length=1)
    key_prefix: str | None = None

    @field_validator("content_type")
    @classmethod
    def normalize_content_type(cls, value: str) -> str:
        """去除 Content-Type 首尾空白；空白字符串视为非法。"""
        stripped = value.strip()
        if not stripped:
            raise ValueError("content_type is required")
        return stripped

    @field_validator("key_prefix")
    @classmethod
    def normalize_key_prefix(cls, value: str | None) -> str:
        """规范化前缀为非空、无首尾斜杠的字符串；缺省返回 uploads。"""
        if value is None:
            return "uploads"
        stripped = value.strip().strip("/")
        if not stripped:
            return "uploads"
        return stripped


class StorageUploadFromUrlInput(StorageUploadOptions):
    """URL 中转上传接收的已校验输入。"""

    source_url: str = Field(min_length=1)

    @field_validator("source_url")
    @classmethod
    def normalize_source_url(cls, value: str) -> str:
        """去除 URL 首尾空白；空白字符串视为非法。"""
        stripped = value.strip()
        if not stripped:
            raise ValueError("source_url is required")
        return stripped


class StorageUploadUrlInput(StorageUploadOptions):
    """获取预签名上传 URL 接收的已校验输入。"""


class StorageDownloadUrlInput(BaseModel):
    """刷新预签名 URL 接收的已校验输入。"""

    key: str = Field(min_length=1)

    @field_validator("key")
    @classmethod
    def normalize_key(cls, value: str) -> str:
        """去除 key 首尾空白；空白字符串视为非法。"""
        stripped = value.strip()
        if not stripped:
            raise ValueError("key is required")
        return stripped

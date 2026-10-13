"""对象存储 MCP 工具的已校验输入契约。"""

from pydantic import BaseModel, Field, field_validator, model_validator


class StorageUploadOptions(BaseModel):
    """对象存储上传的通用选项（content_type 与 key 规范化）。"""

    content_type: str = Field(min_length=1)
    key: str | None = None

    @field_validator("content_type")
    @classmethod
    def normalize_content_type(cls, value: str) -> str:
        """去除 Content-Type 首尾空白；空白字符串视为非法。"""
        stripped = value.strip()
        if not stripped:
            raise ValueError("content_type is required")
        return stripped

    @field_validator("key")
    @classmethod
    def normalize_key(cls, value: str | None) -> str | None:
        """规范化对象 key 为非空、无首部斜杠的字符串；缺省返回 None（随机路径）。"""
        if value is None:
            return None
        stripped = value.strip().lstrip("/")
        if not stripped:
            raise ValueError("key must not be empty")
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


class StorageDeleteInput(StorageDownloadUrlInput):
    """删除对象接收的已校验输入。"""

    @field_validator("key")
    @classmethod
    def normalize_key(cls, value: str) -> str:
        """规范化删除 key 为非空、无首部斜杠的字符串。"""
        stripped = value.strip().lstrip("/")
        if not stripped:
            raise ValueError("key is required")
        return stripped


class StorageListInput(BaseModel):
    """列举指定路径下对象接收的已校验输入。"""

    path: str = ""

    @field_validator("path")
    @classmethod
    def normalize_path(cls, value: str) -> str:
        """规范化路径为无首部斜杠的字符串；空字符串表示列举全部对象。"""
        stripped = value.strip().lstrip("/")
        if not stripped:
            return ""
        # 目录语义的路径自动补尾部斜杠，避免误命中同名前缀的对象。
        return stripped if stripped.endswith("/") else f"{stripped}/"


class StorageRenameInput(BaseModel):
    """重命名/移动对象接收的已校验输入。"""

    source_key: str = Field(min_length=1)
    target_key: str = Field(min_length=1)

    @field_validator("source_key")
    @classmethod
    def normalize_source_key(cls, value: str) -> str:
        """规范化源 key 为非空、无首部斜杠的字符串。"""
        stripped = value.strip().lstrip("/")
        if not stripped:
            raise ValueError("source_key is required")
        return stripped

    @field_validator("target_key")
    @classmethod
    def normalize_target_key(cls, value: str) -> str:
        """规范化目标 key 为非空、无首部斜杠的字符串。"""
        stripped = value.strip().lstrip("/")
        if not stripped:
            raise ValueError("target_key is required")
        return stripped


class StorageMovePair(BaseModel):
    """批量移动中单个对象的源/目标映射。"""

    source_key: str = Field(min_length=1)
    target_key: str = Field(min_length=1)

    @field_validator("source_key")
    @classmethod
    def normalize_source_key(cls, value: str) -> str:
        """规范化源 key 为非空、无首部斜杠的字符串。"""
        stripped = value.strip().lstrip("/")
        if not stripped:
            raise ValueError("source_key is required")
        return stripped

    @field_validator("target_key")
    @classmethod
    def normalize_target_key(cls, value: str) -> str:
        """规范化目标 key 为非空、无首部斜杠的字符串。"""
        stripped = value.strip().lstrip("/")
        if not stripped:
            raise ValueError("target_key is required")
        return stripped


class StorageBatchMoveInput(BaseModel):
    """批量移动接收的已校验输入：前缀（目录级）模式或显式映射模式二选一。"""

    source_path: str | None = None
    target_path: str | None = None
    moves: list[StorageMovePair] | None = None

    @field_validator("source_path")
    @classmethod
    def normalize_source_path(cls, value: str | None) -> str | None:
        """规范化源路径为无首部斜杠、带尾部斜杠的目录前缀；空字符串表示根。"""
        if value is None:
            return None
        return StorageListInput.normalize_path(value)

    @field_validator("target_path")
    @classmethod
    def normalize_target_path(cls, value: str | None) -> str | None:
        """规范化目标路径；空字符串表示移动到 bucket 根目录。"""
        if value is None:
            return None
        return StorageListInput.normalize_path(value)

    @model_validator(mode="after")
    def require_exactly_one_mode(self) -> "StorageBatchMoveInput":
        """校验前缀模式与映射模式互斥且必须提供其一。"""
        if self.moves is not None and self.source_path is not None:
            raise ValueError("moves and source_path are mutually exclusive")
        if self.moves is not None:
            if not self.moves:
                raise ValueError("moves must not be empty")
            return self
        if self.source_path is None:
            raise ValueError("either moves or source_path is required")
        if self.target_path is None:
            raise ValueError("target_path is required with source_path")
        return self


class StorageBatchDeleteInput(BaseModel):
    """批量删除接收的已校验输入：路径（前缀）模式或显式 key 列表模式二选一。"""

    path: str | None = None
    keys: list[str] | None = None
    force: bool = False

    @field_validator("path")
    @classmethod
    def normalize_path(cls, value: str | None) -> str | None:
        """规范化路径为无首部斜杠、带尾部斜杠的目录前缀；空字符串表示根。"""
        if value is None:
            return None
        return StorageListInput.normalize_path(value)

    @field_validator("keys")
    @classmethod
    def normalize_keys(cls, value: list[str] | None) -> list[str] | None:
        """规范化每个 key 为非空、无首部斜杠的字符串并去重（保持顺序）。"""
        if value is None:
            return None
        normalized: list[str] = []
        for raw in value:
            stripped = raw.strip().lstrip("/")
            if not stripped:
                raise ValueError("keys must not contain empty entries")
            if stripped not in normalized:
                normalized.append(stripped)
        return normalized

    @model_validator(mode="after")
    def require_exactly_one_mode(self) -> "StorageBatchDeleteInput":
        """校验路径模式与列表模式互斥且必须提供其一，列表不允许为空。"""
        if self.keys is not None and self.path is not None:
            raise ValueError("keys and path are mutually exclusive")
        if self.keys is not None:
            if not self.keys:
                raise ValueError("keys must not be empty")
            return self
        if self.path is None:
            raise ValueError("either keys or path is required")
        return self

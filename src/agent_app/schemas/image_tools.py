"""图像生成 MCP 工具的已校验输入契约。"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# 支持的画面比例（与 Gemini 图像生成常见比例一致）。
AspectRatio = Literal["1:1", "2:3", "3:2", "3:4", "4:3", "9:16", "16:9", "21:9"]

# 支持的生图模型：gemini 走 gemini-web2api 反代，gpt-6-luna 走 image_vision 网关。
ImageModel = Literal["gemini", "gpt-6-luna"]


class ImageGenerateInput(BaseModel):
    """generate_image 接收的已校验输入。"""

    prompt: str = Field(min_length=1)
    aspect_ratio: AspectRatio | None = None
    path: str | None = None
    model: ImageModel = "gemini"

    @field_validator("prompt")
    @classmethod
    def normalize_prompt(cls, value: str) -> str:
        """去除 prompt 首尾空白；空白字符串视为非法。"""
        stripped = value.strip()
        if not stripped:
            raise ValueError("prompt is required")
        return stripped

    @field_validator("path")
    @classmethod
    def normalize_path(cls, value: str | None) -> str | None:
        """规范化自定义存储路径：去空白与首部斜杠；空白视为未提供。"""
        if value is None:
            return None
        stripped = value.strip().lstrip("/")
        return stripped or None


class ImageRecognizeInput(BaseModel):
    """recognize_image 接收的已校验输入：url 与 key 二选一。"""

    url: str | None = None
    key: str | None = None

    @field_validator("url", "key")
    @classmethod
    def normalize_optional_str(cls, value: str | None) -> str | None:
        """去除首尾空白；空白字符串视为未提供。"""
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @model_validator(mode="after")
    def check_exactly_one_source(self) -> "ImageRecognizeInput":
        """url 与 key 必须恰好提供一个。"""
        if (self.url is None) == (self.key is None):
            raise ValueError("exactly one of url or key is required")
        return self


class ImageOrganizeInput(BaseModel):
    """organize_images 接收的已校验输入。"""

    path: str = Field(default="", description="待整理的目录前缀，空表示 bucket 根")
    target_prefix: str = Field(default="图片", min_length=1)
    limit: int = Field(default=20, ge=1, le=200)
    dry_run: bool = False

    @field_validator("path", "target_prefix")
    @classmethod
    def normalize_prefix(cls, value: str) -> str:
        """规范化目录前缀：去空白与首部斜杠；空白视为未提供（根目录）。"""
        stripped = value.strip().lstrip("/")
        return stripped

    @field_validator("path")
    @classmethod
    def ensure_dir_prefix(cls, value: str) -> str:
        """path 语义为目录，补齐尾部斜杠避免误命中同名前缀对象。"""
        return value if not value or value.endswith("/") else f"{value}/"

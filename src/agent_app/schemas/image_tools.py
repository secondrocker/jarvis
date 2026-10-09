"""图像生成 MCP 工具的已校验输入契约。"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

# 支持的画面比例（与 Gemini 图像生成常见比例一致）。
AspectRatio = Literal["1:1", "2:3", "3:2", "3:4", "4:3", "9:16", "16:9", "21:9"]


class ImageGenerateInput(BaseModel):
    """generate_image 接收的已校验输入。"""

    prompt: str = Field(min_length=1)
    aspect_ratio: AspectRatio | None = None

    @field_validator("prompt")
    @classmethod
    def normalize_prompt(cls, value: str) -> str:
        """去除 prompt 首尾空白；空白字符串视为非法。"""
        stripped = value.strip()
        if not stripped:
            raise ValueError("prompt is required")
        return stripped

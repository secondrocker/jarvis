"""应用配置：从 YAML 读取层级结构。"""

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, SecretStr

DEFAULT_CONFIG_PATH = Path("config.yaml")


class OpenAIConfig(BaseModel):
    """OpenAI 客户端与模型选择配置。"""

    api_key: SecretStr
    model: str = Field(min_length=1)
    base_url: str | None = None
    summary_model: str | None = Field(default=None, min_length=1)
    solution_planning_model: str | None = Field(default=None, min_length=1)
    timeout_seconds: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=2, ge=0, le=5)


class TaskConfig(BaseModel):
    """任务执行超时配置。"""

    timeout_seconds: float = Field(default=300.0, gt=0)


class LogConfig(BaseModel):
    """日志配置。"""

    level: str = "INFO"


class S3Config(BaseModel):
    """S3 兼容对象存储配置。"""

    endpoint_url: str | None = None
    access_key: SecretStr | None = None
    secret_key: SecretStr | None = None
    region: str = "us-east-1"
    bucket: str | None = None
    url_expires_seconds: int = Field(default=604800, gt=0)


class McpConfig(BaseModel):
    """MCP 工具服务暴露配置。"""

    enabled: bool = True
    mount_path: str = "/mcp"


class Settings(BaseModel):
    """从 config.yaml 读取的层级应用配置。"""

    openai: OpenAIConfig
    task: TaskConfig = Field(default_factory=TaskConfig)
    log: LogConfig = Field(default_factory=LogConfig)
    s3: S3Config = Field(default_factory=S3Config)
    mcp: McpConfig = Field(default_factory=McpConfig)


def load_settings(path: str | Path = DEFAULT_CONFIG_PATH) -> Settings:
    """从 YAML 读取层级配置。

    参数:
        path: 配置文件路径，默认项目根 config.yaml。

    返回值:
        解析后的应用配置。
    """
    data: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return Settings.model_validate(data)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """返回进程内复用的应用配置实例。

    返回值:
        从 config.yaml 解析并缓存的配置。
    """
    return load_settings()

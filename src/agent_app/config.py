"""应用配置：从 YAML 读取层级结构。"""

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator

DEFAULT_CONFIG_PATH = Path("config.yaml")


class OpenAIConfig(BaseModel):
    """OpenAI 客户端与模型选择配置。"""

    api_key: SecretStr
    model: str = Field(min_length=1)
    base_url: str | None = None
    summary_model: str | None = Field(default=None, min_length=1)
    solution_planning_model: str | None = Field(default=None, min_length=1)
    info_price_model: str | None = Field(default=None, min_length=1)
    timeout_seconds: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=2, ge=0, le=5)


class TaskConfig(BaseModel):
    """任务执行超时配置。"""

    timeout_seconds: float = Field(default=600.0, gt=0)


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


class WebGatewayConfig(BaseModel):
    """外部 Web 网关（搜索/抓取）配置。"""

    base_url: str | None = None
    api_token: SecretStr | None = None
    search_timeout_seconds: float = Field(default=15.0, gt=0)
    fetch_timeout_seconds: float = Field(default=40.0, gt=0)
    # web_search + web_fetch 在单次任务内的合计调用次数上限（全任务级，
    # 跨所有子代理共享，见 tools/web_budget.py）。建议 5-8。
    total_call_limit: int = Field(default=8, ge=1)

    @field_validator("base_url")
    @classmethod
    def normalize_base_url(cls, value: str | None) -> str | None:
        """去除首尾空白与尾斜杠；空白字符串视为未配置。"""
        if value is None:
            return None
        stripped = value.strip().rstrip("/")
        return stripped or None

    @model_validator(mode="after")
    def check_url_token_pair(self) -> "WebGatewayConfig":
        """base_url 与 api_token 必须同时配置或同时缺省。

        只配一半属于配置错误，应在校验阶段立即失败而非静默禁用，
        与“拼写错误不被静默路由”的整体哲学一致。
        """
        if (self.base_url is None) != (self.api_token is None):
            raise ValueError(
                "web_gateway.base_url and web_gateway.api_token must be configured together"
            )
        return self


class Settings(BaseModel):
    """从 config.yaml 读取的层级应用配置。"""

    openai: OpenAIConfig
    task: TaskConfig = Field(default_factory=TaskConfig)
    log: LogConfig = Field(default_factory=LogConfig)
    s3: S3Config = Field(default_factory=S3Config)
    mcp: McpConfig = Field(default_factory=McpConfig)
    web_gateway: WebGatewayConfig = Field(default_factory=WebGatewayConfig)


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

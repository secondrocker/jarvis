"""应用配置模型及其缓存入口。"""

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """集中定义从环境变量和 .env 文件读取的应用配置。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    openai_api_key: SecretStr
    openai_model: str = Field(min_length=1)
    summary_model: str | None = Field(default=None, min_length=1)
    solution_planning_model: str | None = Field(default=None, min_length=1)
    openai_base_url: str | None = None
    llm_timeout_seconds: float = Field(default=60.0, gt=0)
    llm_max_retries: int = Field(default=2, ge=0, le=5)
    task_timeout_seconds: float = Field(default=300.0, gt=0)
    log_level: str = "INFO"
    pdf_image_output_dir: str = ".data/pdf_images"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """返回进程内复用的应用配置实例。

    返回值:
        从环境变量与 .env 文件解析并缓存的配置。
    """
    return Settings()

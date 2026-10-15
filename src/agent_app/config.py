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
    grade_homework_model: str | None = Field(default=None, min_length=1)
    # 批改模型的独立端点覆盖(可选);未配置时沿用全局 api_key/base_url。
    grade_homework_base_url: str | None = Field(default=None, min_length=1)
    grade_homework_api_key: SecretStr | None = None
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
    public_domain: str | None = Field(
        default=None, description="公开访问端点，在此端点直接签名；不带协议时默认 HTTPS"
    )


class McpConfig(BaseModel):
    """MCP 工具服务暴露配置。"""

    enabled: bool = True
    mount_path: str = "/mcp"
    # Bearer token；配置后 MCP 端点要求 Authorization: Bearer <token> 请求头，
    # 未配置则不做认证（保持向后兼容）。
    token: SecretStr | None = None


class ZhipuWebConfig(BaseModel):
    """智谱 BigModel 工具 API（网络搜索 / 网页阅读）配置。

    api_key 未配置时不注册对应 MCP 工具、不向 Deep Agent 注入
    （与 image_gen 的装配哲学一致）。
    """

    # 开放平台服务地址（OpenAPI servers.url，一般无需修改）。
    base_url: str = Field(default="https://open.bigmodel.cn/api", min_length=1)
    api_key: SecretStr | None = None
    # 默认搜索引擎：search_std / search_pro / search_pro_sogou / search_pro_quark。
    search_engine: str = Field(default="search_std", min_length=1)
    # 默认返回结果条数（1-50）。
    search_count: int = Field(default=10, ge=1, le=50)
    # 搜索内容量：medium（摘要）/ high（详细）。
    content_size: str = Field(default="medium", min_length=1)
    # 网页阅读返回格式（markdown / text）。
    return_format: str = Field(default="markdown", min_length=1)
    search_timeout_seconds: float = Field(default=15.0, gt=0)
    fetch_timeout_seconds: float = Field(default=30.0, gt=0)

    @field_validator("base_url")
    @classmethod
    def normalize_base_url(cls, value: str) -> str:
        """去除首尾空白与尾斜杠。"""
        return value.strip().rstrip("/") or "https://open.bigmodel.cn/api"

    @property
    def enabled(self) -> bool:
        """api_key 已配置且非空白时视为启用。"""
        return self.api_key is not None and bool(self.api_key.get_secret_value().strip())


class ImageGenConfig(BaseModel):
    """Gemini 图像生成服务（gemini-web2api 反代）配置。

    base_url 与 api_key 必须同时配置或同时缺省；缺省时不注册图像生成
    MCP 工具（与 web_gateway 的装配哲学一致）。
    """

    base_url: str | None = None
    api_key: SecretStr | None = None
    model: str = Field(default="gemini-image", min_length=1)
    # gpt-6-luna 生图在 image_vision 网关 /v1/responses 请求的编排模型名
    # （由其调用 image_generation 工具，上游生图后端固定 gpt-image-2）。
    luna_model: str = Field(default="gpt-6-luna", min_length=1)
    # gemini 生图失败时是否自动降级到 gpt-6-luna（需 image_vision 已配置）。
    luna_fallback: bool = True
    timeout_seconds: float = Field(default=180.0, gt=0)

    @field_validator("base_url")
    @classmethod
    def normalize_base_url(cls, value: str | None) -> str | None:
        """去除首尾空白与尾斜杠；空白字符串视为未配置。"""
        if value is None:
            return None
        stripped = value.strip().rstrip("/")
        return stripped or None

    @model_validator(mode="after")
    def check_url_token_pair(self) -> "ImageGenConfig":
        """base_url 与 api_key 必须同时配置或同时缺省。"""
        if (self.base_url is None) != (self.api_key is None):
            raise ValueError("image_gen.base_url and image_gen.api_key must be configured together")
        return self


class ImageVisionConfig(BaseModel):
    """Gemini 图像识别（vision）服务（OpenAI 兼容反代）配置。

    base_url 与 api_key 必须同时配置或同时缺省；缺省时不注册图像识别
    MCP 工具（与 image_gen 的装配哲学一致）。
    """

    base_url: str | None = None
    api_key: SecretStr | None = None
    model: str = Field(default="gemini-3.8-flash", min_length=1)
    # 主模型连续失败时的降级备用模型（网关对部分模型需登录态账号且偶发 504）。
    fallback_model: str = Field(default="gemini-3.6-flash", min_length=1)
    timeout_seconds: float = Field(default=120.0, gt=0)
    # 送识别前兜底压缩的最大边长（像素）；过大图片压缩后可显著降低延迟。
    max_image_edge: int = Field(default=1280, ge=256)
    # 单次 organize_images 批量整理的默认对象数上限。
    batch_limit: int = Field(default=20, ge=1, le=200)

    @field_validator("base_url")
    @classmethod
    def normalize_base_url(cls, value: str | None) -> str | None:
        """去除首尾空白与尾斜杠；空白字符串视为未配置。"""
        if value is None:
            return None
        stripped = value.strip().rstrip("/")
        return stripped or None

    @model_validator(mode="after")
    def check_url_token_pair(self) -> "ImageVisionConfig":
        """base_url 与 api_key 必须同时配置或同时缺省。"""
        if (self.base_url is None) != (self.api_key is None):
            raise ValueError(
                "image_vision.base_url and image_vision.api_key must be configured together"
            )
        return self


class GradeHomeworkConfig(BaseModel):
    """批改作业工作流的图片限制配置。"""

    max_image_mb: int = Field(default=8, ge=1)
    max_image_edge: int = Field(default=2048, ge=256)


class AuthConfig(BaseModel):
    """批改小程序鉴权配置;access_password 为空表示开放访问。"""

    access_password: str | None = None
    secret: str = "dev-secret"
    token_ttl_hours: int = Field(default=168, ge=1)


class Settings(BaseModel):
    """从 config.yaml 读取的层级应用配置。"""

    openai: OpenAIConfig
    task: TaskConfig = Field(default_factory=TaskConfig)
    log: LogConfig = Field(default_factory=LogConfig)
    s3: S3Config = Field(default_factory=S3Config)
    mcp: McpConfig = McpConfig()
    zhipu_web: ZhipuWebConfig = Field(default_factory=ZhipuWebConfig)
    grade_homework: GradeHomeworkConfig = Field(default_factory=GradeHomeworkConfig)
    image_gen: ImageGenConfig = Field(default_factory=ImageGenConfig)
    image_vision: ImageVisionConfig = Field(default_factory=ImageVisionConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)


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

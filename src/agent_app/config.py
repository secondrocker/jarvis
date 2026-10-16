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
    # 桶对匿名读开放（public-read bucket policy）时置 true：download_url
    # 直接返回 {public_domain}/{bucket}/{key} 的无签名 URL——签名参数
    # （X-Amz-Signature 等）会被部分调用方（如 openclaw）当密钥脱敏导致
    # 链接失效，公开读下无签名链接无此问题。上传仍走预签名 PUT。
    public_read: bool = Field(default=False)


class McpConfig(BaseModel):
    """MCP 工具服务暴露配置。"""

    enabled: bool = True
    mount_path: str = "/mcp"
    # Bearer token；配置后 MCP 端点要求 Authorization: Bearer <token> 请求头，
    # 未配置则不做认证（保持向后兼容）。
    token: SecretStr | None = None


class ZhipuWebConfig(BaseModel):
    """智谱 GLM Coding Plan 专属 MCP（联网搜索 / 网页读取）配置。

    走 Coding Plan 的 Remote MCP Server（`/api/mcp/web_search_prime/mcp` 与
    `/api/mcp/web_reader/mcp`），额度记在编程套餐上；开放平台的
    `/paas/v4/web_search`、`/paas/v4/reader` 是另一套按量计费，未充值会报
    1113「余额不足或无可用资源包」。

    api_key 未配置时不注册对应 MCP 工具、不向 Deep Agent 注入
    （与 image_gen 的装配哲学一致）。
    """

    # 联网搜索 MCP Server（工具名 web_search_prime）。
    search_mcp_url: str = Field(
        default="https://open.bigmodel.cn/api/mcp/web_search_prime/mcp", min_length=1
    )
    # 网页读取 MCP Server（工具名 webReader）。
    reader_mcp_url: str = Field(
        default="https://open.bigmodel.cn/api/mcp/web_reader/mcp", min_length=1
    )
    # Coding Plan 套餐 Key（与开放平台普通 API Key 不通用）。
    api_key: SecretStr | None = None
    # 搜索内容量：medium（400-600 字摘要）/ high（更全面，单价更高）。
    content_size: str = Field(default="medium", min_length=1)
    # 搜索地域：cn（中文区域）/ us（非中文区域）。
    location: str = Field(default="cn", min_length=1)
    # 网页阅读返回格式（markdown / text）。
    return_format: str = Field(default="markdown", min_length=1)
    search_timeout_seconds: float = Field(default=20.0, gt=0)
    fetch_timeout_seconds: float = Field(default=40.0, gt=0)

    @field_validator("search_mcp_url", "reader_mcp_url")
    @classmethod
    def normalize_mcp_url(cls, value: str) -> str:
        """去除首尾空白；必须是 http(s) 端点。"""
        stripped = value.strip()
        if not stripped.startswith(("http://", "https://")):
            raise ValueError("mcp url must be an http(s) endpoint")
        return stripped

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
    # 账号池中 free/失效账号会被上游静默剥掉生图工具导致偶发无图，
    # 对"无图结果"的重试次数（0-2）。
    luna_retries: int = Field(default=1, ge=0, le=2)
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

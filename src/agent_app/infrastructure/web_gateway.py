"""外部 Web 网关客户端：互联网搜索与网页抓取。"""

from typing import Any

import httpx

from agent_app.config import WebGatewayConfig
from agent_app.errors import AppError, ErrorCode


class WebGatewayClient:
    """以 Bearer token 调用 Web 网关搜索/抓取接口的同步客户端。"""

    def __init__(
        self,
        *,
        base_url: str,
        token: str,
        search_timeout: float,
        fetch_timeout: float,
    ) -> None:
        """初始化网关客户端。

        参数:
            base_url: 已去除尾斜杠的网关根地址（如 https://surf.leegoo.ltd）。
            token: Bearer 鉴权 token，仅写入请求头，不输出到日志。
            search_timeout: 搜索请求的等待秒数。
            fetch_timeout: 抓取请求的等待秒数。
        """
        self._base_url = base_url
        self._token = token
        self._search_timeout = search_timeout
        self._fetch_timeout = fetch_timeout

    def search(self, *, query: str, limit: int = 5) -> dict[str, Any]:
        """搜索互联网信息并返回网关原始响应。

        参数:
            query: 已规范化的搜索关键词。
            limit: 返回结果条数上限。

        返回值:
            网关响应 JSON（结构由网关定义，此处透传）。

        异常:
            AppError: 网关不可用或响应非法时抛 UPSTREAM_UNAVAILABLE。
        """
        return self._post(
            "/search",
            {"query": query, "limit": limit},
            timeout=self._search_timeout,
        )

    def fetch(self, *, url: str, max_chars: int = 20000) -> dict[str, Any]:
        """抓取指定网页并返回网关原始响应。

        参数:
            url: 已规范化的目标网页地址。
            max_chars: 返回正文的字符数上限。

        返回值:
            网关响应 JSON（结构由网关定义，此处透传）。

        异常:
            AppError: 网关不可用或响应非法时抛 UPSTREAM_UNAVAILABLE。
        """
        return self._post(
            "/fetch",
            {"url": url, "max_chars": max_chars},
            timeout=self._fetch_timeout,
        )

    def _post(self, path: str, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
        """向网关发送 POST 请求并归一化失败为安全的 AppError。

        参数:
            path: 网关接口路径（如 /search）。
            payload: 请求 JSON 体。
            timeout: 请求等待秒数。

        返回值:
            网关响应 JSON 字典。

        异常:
            AppError: 网络故障、非 2xx 或响应非法时抛 UPSTREAM_UNAVAILABLE。
        """
        try:
            response = httpx.post(
                f"{self._base_url}{path}",
                json=payload,
                headers={"Authorization": f"Bearer {self._token}"},
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as error:
            raise AppError(
                ErrorCode.UPSTREAM_UNAVAILABLE,
                "Web gateway is temporarily unavailable",
            ) from error
        except ValueError as error:
            # JSONDecodeError 是 ValueError 的子类，直接按父类捕获。
            raise AppError(
                ErrorCode.UPSTREAM_UNAVAILABLE,
                "Web gateway returned an invalid response",
            ) from error
        if not isinstance(data, dict):
            raise AppError(
                ErrorCode.UPSTREAM_UNAVAILABLE,
                "Web gateway returned an invalid response",
            )
        return data


def create_web_gateway(config: WebGatewayConfig | None) -> WebGatewayClient | None:
    """按配置构造 Web 网关客户端；未配置时返回 None 表示能力禁用。

    参数:
        config: web_gateway 配置节；None 或 base_url/token 缺省视为未启用。

    返回值:
        可用的 WebGatewayClient，或表示未启用的 None。
    """
    if config is None or config.base_url is None or config.api_token is None:
        return None
    return WebGatewayClient(
        base_url=config.base_url,
        token=config.api_token.get_secret_value(),
        search_timeout=config.search_timeout_seconds,
        fetch_timeout=config.fetch_timeout_seconds,
    )

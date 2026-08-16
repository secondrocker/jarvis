"""Web 网关客户端的进程内测试（monkeypatch httpx.post，不联网）。"""

import httpx
import pytest

from agent_app.config import WebGatewayConfig
from agent_app.errors import AppError, ErrorCode
from agent_app.infrastructure import web_gateway
from agent_app.infrastructure.web_gateway import WebGatewayClient, create_web_gateway


class _FakeResponse:
    """httpx.post 的最小替身，返回预设 JSON。"""

    def __init__(self, payload: dict, *, status_error: Exception | None = None) -> None:
        self._payload = payload
        self._status_error = status_error

    def raise_for_status(self) -> None:
        if self._status_error is not None:
            raise self._status_error

    def json(self) -> dict:
        return self._payload


@pytest.fixture
def captured(monkeypatch) -> dict:
    """patch web_gateway.httpx.post 捕获请求参数，返回可控替身。"""
    state: dict = {}

    def fake_post(url, *, json=None, headers=None, timeout=None):
        state["url"] = url
        state["json"] = json
        state["headers"] = headers
        state["timeout"] = timeout
        return _FakeResponse({"ok": True})

    monkeypatch.setattr(web_gateway.httpx, "post", fake_post)
    return state


def _client() -> WebGatewayClient:
    """构造固定参数的测试客户端。"""
    return WebGatewayClient(
        base_url="https://surf.leegoo.ltd",
        token="secret-token",
        search_timeout=15.0,
        fetch_timeout=40.0,
    )


def test_search_posts_query_with_bearer_token_and_timeout(captured) -> None:
    result = _client().search(query="langgraph 教程", limit=7)

    assert result == {"ok": True}
    assert captured["url"] == "https://surf.leegoo.ltd/search"
    assert captured["json"] == {"query": "langgraph 教程", "limit": 7}
    assert captured["headers"] == {"Authorization": "Bearer secret-token"}
    assert captured["timeout"] == 15.0


def test_search_applies_default_limit(captured) -> None:
    _client().search(query="任意")

    assert captured["json"] == {"query": "任意", "limit": 5}


def test_fetch_posts_url_with_max_chars_and_timeout(captured) -> None:
    result = _client().fetch(url="https://example.com/page", max_chars=3000)

    assert result == {"ok": True}
    assert captured["url"] == "https://surf.leegoo.ltd/fetch"
    assert captured["json"] == {"url": "https://example.com/page", "max_chars": 3000}
    assert captured["timeout"] == 40.0


def test_fetch_applies_default_max_chars(captured) -> None:
    _client().fetch(url="https://example.com/page")

    assert captured["json"] == {"url": "https://example.com/page", "max_chars": 20000}


@pytest.mark.parametrize(
    "error",
    [
        httpx.TimeoutException("boom"),
        httpx.ConnectError("boom"),
        httpx.HTTPStatusError("500", request=None, response=None),
    ],
)
def test_gateway_errors_normalize_to_upstream_unavailable(monkeypatch, error) -> None:
    def fake_post(url, *, json=None, headers=None, timeout=None):
        raise error

    monkeypatch.setattr(web_gateway.httpx, "post", fake_post)
    with pytest.raises(AppError) as raised:
        _client().search(query="任意")

    assert raised.value.code is ErrorCode.UPSTREAM_UNAVAILABLE
    assert raised.value.public_message == "Web gateway is temporarily unavailable"


def test_invalid_json_normalizes_to_upstream_unavailable(monkeypatch) -> None:
    class _BadJsonResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            raise ValueError("not json")

    monkeypatch.setattr(
        web_gateway.httpx,
        "post",
        lambda *args, **kwargs: _BadJsonResponse(),
    )
    with pytest.raises(AppError) as raised:
        _client().fetch(url="https://example.com/page")

    assert raised.value.public_message == "Web gateway returned an invalid response"


def test_non_dict_json_normalizes_to_upstream_unavailable(monkeypatch) -> None:
    class _ListResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> list:
            return ["not", "a", "dict"]

    monkeypatch.setattr(
        web_gateway.httpx,
        "post",
        lambda *args, **kwargs: _ListResponse(),
    )
    with pytest.raises(AppError) as raised:
        _client().fetch(url="https://example.com/page")

    assert raised.value.public_message == "Web gateway returned an invalid response"


def test_create_web_gateway_returns_none_when_unconfigured() -> None:
    assert create_web_gateway(None) is None
    assert create_web_gateway(WebGatewayConfig()) is None


def test_create_web_gateway_builds_client_from_config() -> None:
    client = create_web_gateway(
        WebGatewayConfig(
            base_url="https://surf.leegoo.ltd/",
            api_token="gateway-token",
            search_timeout_seconds=15.0,
            fetch_timeout_seconds=40.0,
        )
    )

    assert isinstance(client, WebGatewayClient)
    assert client._base_url == "https://surf.leegoo.ltd"
    assert client._token == "gateway-token"
    assert client._search_timeout == 15.0
    assert client._fetch_timeout == 40.0

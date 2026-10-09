"""图像生成 MCP 工具的进程内测试（FastMCP Client，不联网）。"""

import pytest
from fakes import FakeObjectStorage
from fastmcp import Client
from fastmcp.exceptions import ToolError
from pydantic import SecretStr

from agent_app.config import Settings
from agent_app.tools import build_mcp_server, image_tools

# PNG 魔数开头的最小合法字节，用于验证嗅探出的扩展名。
_PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"0" * 16


class _FakeResponse:
    """httpx 响应替身：固定状态码、字节与 JSON 体。"""

    def __init__(self, *, content: bytes = b"", json_body: dict | None = None) -> None:
        self.content = content
        self._json = json_body

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._json  # type: ignore[return-value]


@pytest.fixture
def fake_storage() -> FakeObjectStorage:
    """记录上传调用的 fake 存储。"""
    return FakeObjectStorage()


@pytest.fixture
def image_settings() -> Settings:
    """带图像生成配置的应用配置。"""
    return Settings.model_validate(
        {
            "openai": {"api_key": SecretStr("test-key"), "model": "gpt-4o-mini"},
            "image_gen": {
                "base_url": "https://gemini.test",
                "api_key": SecretStr("sk-test"),
            },
        }
    )


@pytest.fixture
def mcp_server(image_settings, fake_storage):
    """注册了 generate_image 的聚合 MCP 服务。"""
    return build_mcp_server(settings=image_settings, storage=fake_storage)


@pytest.fixture
def patch_generation(monkeypatch):
    """patch httpx 请求：POST 返回带 markdown 图片的回复，GET 返回 PNG 字节。

    返回捕获字典，可改写 content 与下载字节以构造不同场景。
    """
    captured: dict = {
        "post_payload": None,
        "content": "好的，这是图片：![image](https://img.test/cat.png)\n![2](https://img.test/cat2.png)",
        "download_bytes": _PNG_BYTES,
    }

    def fake_post(url, *, json, headers, timeout):  # noqa: ANN001
        captured["post_payload"] = {"url": url, "json": json, "headers": headers}
        message = {"content": captured["content"]}
        if isinstance(captured["content"], list):
            message = {"content": captured["content"]}
        return _FakeResponse(json_body={"choices": [{"message": message}]})

    def fake_get(url, *, timeout=None, follow_redirects=False):  # noqa: ANN001
        captured.setdefault("downloaded", []).append(url)
        return _FakeResponse(content=captured["download_bytes"])

    monkeypatch.setattr(image_tools.httpx, "post", fake_post)
    monkeypatch.setattr(image_tools.httpx, "get", fake_get)
    return captured


@pytest.mark.asyncio
async def test_generate_image_returns_categorized_url(
    mcp_server, fake_storage, patch_generation
) -> None:
    async with Client(mcp_server) as client:
        result = await client.call_tool(
            "generate_image",
            {"prompt": "画一只戴宇航员头盔的橘猫", "aspect_ratio": "16:9"},
        )

    data = result.data
    # 猫 → animals 目录；日期 + uuid 命名。
    assert data["key"].startswith("images/animals/")
    assert data["key"].endswith(".png")
    assert data["url"].startswith("https://fake-s3.test/images/animals/")
    assert len(data["images"]) == 2
    # 上传字节与 content_type 正确。
    assert fake_storage.uploads[0][0] == data["key"]
    assert fake_storage.uploads[0][1] == "image/png"
    # 请求体包含比例要求与 Bearer token。
    body = patch_generation["post_payload"]
    assert "画面比例 16:9" in body["json"]["messages"][0]["content"]
    assert body["json"]["model"] == "gemini-image"
    assert body["headers"]["Authorization"] == "Bearer sk-test"


@pytest.mark.asyncio
async def test_generate_image_fallback_category(mcp_server, fake_storage, patch_generation) -> None:
    async with Client(mcp_server) as client:
        result = await client.call_tool(
            "generate_image",
            {"prompt": "an abstract geometric pattern"},
        )

    assert result.data["key"].startswith("images/generated/")
    # 未传比例时 prompt 原样发送。
    assert patch_generation["post_payload"]["json"]["messages"][0]["content"] == (
        "an abstract geometric pattern"
    )


@pytest.mark.asyncio
async def test_generate_image_inline_data_url(mcp_server, fake_storage, patch_generation) -> None:
    # 真实反代服务以 ![image](data:image/png;base64,...) 内嵌返回图片。
    import base64

    encoded = base64.b64encode(_PNG_BYTES).decode()
    patch_generation["content"] = f"![image](data:image/png;base64,{encoded})"

    async with Client(mcp_server) as client:
        result = await client.call_tool("generate_image", {"prompt": "画一只猫"})

    # 内嵌 data URL 无需 http 下载。
    assert "downloaded" not in patch_generation
    assert result.data["key"].endswith(".png")
    assert result.data["images"][0]["source"] == "inline"
    assert fake_storage.uploads[0][1] == "image/png"


@pytest.mark.asyncio
async def test_generate_image_skips_non_image_urls(
    mcp_server, fake_storage, patch_generation
) -> None:
    # 占位链接返回 HTML 页面时应被跳过；全部无效时报 ToolError。
    patch_generation["download_bytes"] = b"<!DOCTYPE html><html></html>"
    async with Client(mcp_server) as client:
        with pytest.raises(ToolError):
            await client.call_tool("generate_image", {"prompt": "画一只猫"})
    assert fake_storage.uploads == []


@pytest.mark.asyncio
async def test_generate_image_no_image_raises(mcp_server, patch_generation) -> None:
    patch_generation["content"] = "抱歉，我无法生成该图片。"
    async with Client(mcp_server) as client:
        with pytest.raises(ToolError):
            await client.call_tool("generate_image", {"prompt": "画一只猫"})


@pytest.mark.asyncio
async def test_generate_image_rejects_blank_prompt(mcp_server) -> None:
    async with Client(mcp_server) as client:
        with pytest.raises(ToolError):
            await client.call_tool("generate_image", {"prompt": "   "})


@pytest.mark.asyncio
async def test_mcp_server_omits_tool_without_config(fake_storage) -> None:
    settings = Settings.model_validate(
        {"openai": {"api_key": SecretStr("test-key"), "model": "gpt-4o-mini"}}
    )
    server = build_mcp_server(settings=settings, storage=fake_storage)
    async with Client(server) as client:
        tools = await client.list_tools()
    assert "generate_image" not in {tool.name for tool in tools}

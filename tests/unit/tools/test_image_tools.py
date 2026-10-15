"""图像生成 MCP 工具的进程内测试（FastMCP Client，不联网）。"""

# PNG 魔数开头的最小合法字节，用于验证嗅探出的扩展名。
from io import BytesIO

import pytest
from fakes import FakeObjectStorage
from fastmcp import Client
from fastmcp.exceptions import ToolError
from PIL import Image
from pydantic import SecretStr

from agent_app.config import Settings
from agent_app.tools import build_mcp_server, image_tools

_png_buffer = BytesIO()
Image.new("RGB", (2, 2), "white").save(_png_buffer, format="PNG")
_PNG_BYTES = _png_buffer.getvalue()


class _FakeResponse:
    """httpx 响应替身：固定状态码、字节与 JSON 体。"""

    def __init__(
        self,
        *,
        content: bytes = b"",
        json_body: dict | None = None,
        status_code: int = 200,
    ) -> None:
        self.content = content
        self._json = json_body
        self.status_code = status_code

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._json  # type: ignore[return-value]


def patch_luna_responses(
    monkeypatch,
    items: list[dict] | None = None,
    *,
    error: dict | None = None,
    stream_error: str | None = None,
    text: str | None = None,
    sequence: list[dict] | None = None,
) -> list[dict]:
    """patch httpx.stream：luna 网关返回 /v1/responses SSE 流。

    items 为 image_generation_call 列表（含 result/revised_prompt）；
    error 形如 {"status": 400, "message": "..."} 时返回 HTTP 错误体；
    stream_error 在流内发送 error 事件；text 为模型文字输出；
    sequence 提供逐次调用的差异响应（每项可含 items/text/stream_error，
    超出部分复用最后一项），用于验证重试行为。返回捕获的请求列表。
    """
    import json as jsonlib

    captured: list[dict] = []

    def sse(chunk: dict) -> str:
        return "data: " + jsonlib.dumps(chunk, ensure_ascii=False)

    def build_lines(spec: dict) -> list[str]:
        lines: list[str] = []
        for item in spec.get("items") or []:
            done = {
                "type": "response.output_item.done",
                "item": {"type": "image_generation_call", **item},
            }
            lines.append(sse(done))
        if spec.get("text"):
            lines.append(sse({"type": "response.output_text.delta", "delta": spec["text"]}))
        if spec.get("stream_error") is not None:
            lines.append(sse({"type": "error", "error": {"message": spec["stream_error"]}}))
        lines.append(sse({"type": "response.completed"}))
        lines.append("data: [DONE]")
        return lines

    default_spec = {"items": items or [], "text": text, "stream_error": stream_error}
    state = {"index": 0}

    class _FakeStreamResponse:
        status_code = error["status"] if error else 200

        def __init__(self, spec_index: int = 0) -> None:
            spec = default_spec
            if sequence:
                spec = sequence[min(spec_index, len(sequence) - 1)]
            self._lines = build_lines(spec)

        def iter_lines(self):  # noqa: ANN202
            return iter(self._lines)

        def read(self) -> bytes:
            if error:
                return jsonlib.dumps({"error": {"message": error["message"]}}).encode()
            return b"{}"

    class _FakeStreamContext:
        def __enter__(self):  # noqa: ANN204
            spec_index = state["index"]
            state["index"] += 1
            return _FakeStreamResponse(spec_index)

        def __exit__(self, *exc_info: object) -> bool:
            return False

    def fake_stream(method, url, *, json, headers, timeout):  # noqa: ANN001
        captured.append({"url": url, "json": json, "headers": headers})
        return _FakeStreamContext()

    monkeypatch.setattr(image_tools.httpx, "stream", fake_stream)
    return captured


@pytest.fixture
def fake_storage() -> FakeObjectStorage:
    """记录上传调用的 fake 存储。"""
    return FakeObjectStorage()


@pytest.fixture
def image_settings() -> Settings:
    """带图像生成配置的应用配置（含 gpt-6-luna 复用的 image_vision 网关）。"""
    return Settings.model_validate(
        {
            "openai": {"api_key": SecretStr("test-key"), "model": "gpt-4o-mini"},
            "image_gen": {
                "base_url": "https://gemini.test",
                "api_key": SecretStr("sk-test"),
            },
            "image_vision": {
                "base_url": "https://codex.test",
                "api_key": SecretStr("sk-vision"),
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
        if isinstance(captured["content"], dict):
            # dict 形态直接作为完整 message（含 tool_calls 等字段）。
            message = captured["content"]
        else:
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
            {"prompt": "画一只戴宇航员头盔的橘猫", "aspect_ratio": "16:9", "model": "gemini"},
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
            {"prompt": "an abstract geometric pattern", "model": "gemini"},
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
        result = await client.call_tool("generate_image", {"prompt": "画一只猫", "model": "gemini"})

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
            await client.call_tool("generate_image", {"prompt": "画一只猫", "model": "gemini"})
    assert fake_storage.uploads == []


@pytest.mark.asyncio
async def test_generate_image_no_image_raises(mcp_server, patch_generation) -> None:
    patch_generation["content"] = "抱歉，我无法生成该图片。"
    async with Client(mcp_server) as client:
        with pytest.raises(ToolError):
            await client.call_tool("generate_image", {"prompt": "画一只猫", "model": "gemini"})


@pytest.mark.asyncio
async def test_generate_image_gemini_text_refusal_no_fallback(
    mcp_server, fake_storage, monkeypatch
) -> None:
    """gemini 文字拒绝时明确报安全拒绝，且不再降级重试 luna。"""
    post_calls: list[dict] = []

    def fake_post(url, *, json, headers, timeout):  # noqa: ANN001
        post_calls.append({"url": url, "json": json})
        refusal_text = "我无法生成包含裸体或色情内容的图像。"
        return _FakeResponse(json_body={"choices": [{"message": {"content": refusal_text}}]})

    monkeypatch.setattr(image_tools.httpx, "post", fake_post)

    async with Client(mcp_server) as client:
        with pytest.raises(ToolError, match="upstream safety policy"):
            await client.call_tool(
                "generate_image", {"prompt": "explicit content", "model": "gemini"}
            )

    assert len(post_calls) == 1
    assert not fake_storage.uploads


@pytest.mark.asyncio
async def test_generate_image_rejects_blank_prompt(mcp_server) -> None:
    async with Client(mcp_server) as client:
        with pytest.raises(ToolError):
            await client.call_tool("generate_image", {"prompt": "   ", "model": "gemini"})


@pytest.mark.asyncio
async def test_generate_image_uses_custom_path(mcp_server, fake_storage, patch_generation) -> None:
    async with Client(mcp_server) as client:
        result = await client.call_tool(
            "generate_image",
            {"prompt": "画一只猫", "path": "/assets/poster.png", "model": "gemini"},
        )

    data = result.data
    # 自定义路径固定使用（含文件名），自动多图时追加 -2 序号。
    assert [image["key"] for image in data["images"]] == [
        "assets/poster.png",
        "assets/poster-2.png",
    ]
    assert data["key"] == "assets/poster.png"
    assert data["path"] == "assets/poster.png"
    assert data["url"] == "https://fake-s3.test/assets/poster.png"
    assert fake_storage.uploads[0][0] == "assets/poster.png"


@pytest.mark.asyncio
async def test_generate_image_custom_path_appends_extension(
    mcp_server, fake_storage, patch_generation
) -> None:
    patch_generation["content"] = "![image](https://img.test/cat.png)"

    async with Client(mcp_server) as client:
        result = await client.call_tool(
            "generate_image",
            {"prompt": "画一只猫", "path": "assets/poster", "model": "gemini"},
        )

    # 无扩展名的自定义路径按识别出的格式补齐。
    assert result.data["key"] == "assets/poster.png"


@pytest.mark.asyncio
async def test_mcp_server_omits_tool_without_config(fake_storage) -> None:
    settings = Settings.model_validate(
        {"openai": {"api_key": SecretStr("test-key"), "model": "gpt-4o-mini"}}
    )
    server = build_mcp_server(settings=settings, storage=fake_storage)
    async with Client(server) as client:
        tools = await client.list_tools()
    assert "generate_image" not in {tool.name for tool in tools}


@pytest.mark.asyncio
async def test_generate_image_preserves_signed_url_in_all_mcp_fields(
    image_settings, fake_storage, patch_generation, monkeypatch
) -> None:
    """生成图片的顶层及 images 列表在 MCP 两种响应中均保留签名编码。"""
    import json

    expected = "https://public.test/b/image.jpg?Signature=fake%2B%2F%3D&Expires=123"
    monkeypatch.setattr(fake_storage, "download_url", lambda key: expected)
    server = build_mcp_server(settings=image_settings, storage=fake_storage)
    async with Client(server) as client:
        result = await client.call_tool(
            "generate_image", {"prompt": "an abstract pattern", "model": "gemini"}
        )
    for data in (result.data, result.structured_content, json.loads(result.content[0].text)):
        assert data["url"] == expected
        assert all(image["url"] == expected for image in data["images"])


@pytest.mark.asyncio
async def test_generate_image_with_luna_model(mcp_server, fake_storage, monkeypatch) -> None:
    """model=gpt-6-luna 走 /v1/responses 流式并解析 image_generation_call。"""
    import base64

    encoded = base64.b64encode(_PNG_BYTES).decode()
    captured = patch_luna_responses(
        monkeypatch,
        [{"result": encoded, "revised_prompt": "优化后的提示词"}],
    )

    async with Client(mcp_server) as client:
        result = await client.call_tool(
            "generate_image",
            {"prompt": "画一只猫", "model": "gpt-6-luna"},
        )

    data = result.data
    assert data["model"] == "gpt-6-luna"
    assert data["fallback"] is False
    assert data["revised_prompt"] == "优化后的提示词"
    assert data["key"].endswith(".png")
    assert fake_storage.uploads[0][0] == data["key"]
    assert fake_storage.uploads[0][1] == "image/png"
    # 请求发往 image_vision 网关的 responses 流式端点，携带生图工具与独立鉴权。
    body = captured[0]
    assert body["url"] == "https://codex.test/v1/responses"
    assert body["json"]["model"] == "gpt-6-luna"
    assert body["json"]["stream"] is True
    assert body["json"]["tools"] == [{"type": "image_generation"}]
    assert body["json"]["input"][0]["content"][0]["text"] == "画一只猫"
    assert body["headers"]["Authorization"] == "Bearer sk-vision"


@pytest.mark.asyncio
async def test_generate_image_luna_without_vision_config_raises(fake_storage) -> None:
    """image_vision 网关未配置时，gpt-6-luna 调用返回明确错误。"""
    settings = Settings.model_validate(
        {
            "openai": {"api_key": SecretStr("test-key"), "model": "gpt-4o-mini"},
            "image_gen": {
                "base_url": "https://gemini.test",
                "api_key": SecretStr("sk-test"),
            },
        }
    )
    server = build_mcp_server(settings=settings, storage=fake_storage)
    async with Client(server) as client:
        with pytest.raises(ToolError, match="not configured"):
            await client.call_tool("generate_image", {"prompt": "画一只猫", "model": "gpt-6-luna"})


@pytest.mark.asyncio
async def test_generate_image_rejects_unknown_model(mcp_server) -> None:
    """未注册的模型名在校验阶段即被拒绝。"""
    async with Client(mcp_server) as client:
        with pytest.raises(ToolError):
            await client.call_tool("generate_image", {"prompt": "画一只猫", "model": "dall-e-3"})


@pytest.mark.asyncio
async def test_generate_image_falls_back_to_luna(mcp_server, fake_storage, monkeypatch) -> None:
    """gemini 生图无图时自动降级到 gpt-6-luna 并成功交付。"""
    import base64

    encoded = base64.b64encode(_PNG_BYTES).decode()
    stream_calls = patch_luna_responses(
        monkeypatch,
        [{"result": encoded, "revised_prompt": "优化后的提示词"}],
    )
    post_calls: list[dict] = []

    def fake_post(url, *, json, headers, timeout):  # noqa: ANN001
        post_calls.append({"url": url, "json": json})
        # gemini 反代返回无图正文（非拒绝话术），触发降级。
        return _FakeResponse(
            json_body={"choices": [{"message": {"content": "这是一段文字说明，未包含图片。"}}]}
        )

    monkeypatch.setattr(image_tools.httpx, "post", fake_post)

    async with Client(mcp_server) as client:
        result = await client.call_tool("generate_image", {"prompt": "画一只猫", "model": "gemini"})

    data = result.data
    assert data["model"] == "gpt-6-luna"
    assert data["fallback"] is True
    assert data["revised_prompt"] == "优化后的提示词"
    assert data["key"].endswith(".png")
    # 先打 gemini 反代（非流式），失败后再打 luna 网关的 responses 流式端点。
    assert "gemini.test" in post_calls[0]["url"]
    assert stream_calls[0]["url"] == "https://codex.test/v1/responses"
    assert stream_calls[0]["json"]["tools"] == [{"type": "image_generation"}]


@pytest.mark.asyncio
async def test_generate_image_fallback_disabled(fake_storage, monkeypatch) -> None:
    """luna_fallback=false 时不降级，gemini 失败直接报错。"""
    from fastmcp import FastMCP

    from agent_app.config import ImageGenConfig, ImageVisionConfig
    from agent_app.tools.image_tools import register_image_tools

    mcp = FastMCP("Test")
    register_image_tools(
        mcp,
        config=ImageGenConfig(
            base_url="https://gemini.test",
            api_key=SecretStr("sk-test"),
            luna_fallback=False,
        ),
        storage=fake_storage,
        vision_config=ImageVisionConfig(
            base_url="https://codex.test", api_key=SecretStr("sk-vision")
        ),
    )

    def fake_post(url, *, json, headers, timeout):  # noqa: ANN001
        return _FakeResponse(json_body={"choices": [{"message": {"content": "抱歉，无法生成"}}]})

    monkeypatch.setattr(image_tools.httpx, "post", fake_post)
    async with Client(mcp) as client:
        with pytest.raises(ToolError):
            await client.call_tool("generate_image", {"prompt": "画一只猫", "model": "gemini"})


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["", "bm90IGFuIGltYWdl", None])
async def test_luna_empty_or_invalid_result_never_uploads(
    mcp_server, fake_storage, monkeypatch, value
):
    patch_luna_responses(monkeypatch, [{"result": value}])
    async with Client(mcp_server) as client:
        with pytest.raises(ToolError, match="No image was generated"):
            await client.call_tool(
                "generate_image", {"prompt": "adult fashion", "model": "gpt-6-luna"}
            )
    assert not fake_storage.uploads


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        "抱歉，我不能生成露骨色情或全裸性内容。",
        "Sorry, I cannot generate explicit content as it violates the policy.",
    ],
)
async def test_luna_safety_refusal_stops_without_upload_or_fallback(
    mcp_server, fake_storage, monkeypatch, message
):
    """上游内容政策拒绝（HTTP 错误体含拒绝话术）时给出明确安全拒绝错误。"""
    patch_luna_responses(monkeypatch, error={"status": 400, "message": message})
    async with Client(mcp_server) as client:
        with pytest.raises(ToolError, match="upstream safety policy"):
            await client.call_tool(
                "generate_image", {"prompt": "adult fashion", "model": "gpt-6-luna"}
            )
    assert not fake_storage.uploads


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    ["抱歉，我不能生成露骨色情内容。", "Sorry, I cannot generate explicit content."],
)
async def test_luna_stream_error_refusal_stops_without_upload(
    mcp_server, fake_storage, monkeypatch, message
):
    """流内 error 事件含拒绝话术时同样归类为安全拒绝。"""
    patch_luna_responses(monkeypatch, stream_error=message)
    async with Client(mcp_server) as client:
        with pytest.raises(ToolError, match="upstream safety policy"):
            await client.call_tool(
                "generate_image", {"prompt": "adult fashion", "model": "gpt-6-luna"}
            )
    assert not fake_storage.uploads


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs",
    [
        {"error": {"status": 503, "message": "no available accounts"}},
        {"stream_error": "upstream overloaded"},
    ],
)
async def test_luna_upstream_error_without_refusal_is_unavailable(
    mcp_server, fake_storage, monkeypatch, kwargs
):
    """非拒绝类上游错误报服务不可用，而非安全拒绝。"""
    patch_luna_responses(monkeypatch, **kwargs)
    async with Client(mcp_server) as client:
        with pytest.raises(ToolError, match="temporarily unavailable"):
            await client.call_tool("generate_image", {"prompt": "画一只猫", "model": "gpt-6-luna"})
    assert not fake_storage.uploads


@pytest.mark.asyncio
async def test_luna_no_image_text_refusal_stops_without_retry(
    mcp_server, fake_storage, monkeypatch
):
    """模型以纯文字拒绝（无图）时归类为安全拒绝且不重试。"""
    captured = patch_luna_responses(
        monkeypatch, sequence=[{"text": "抱歉，我目前无法在此对话中直接生成或返回图片。"}]
    )
    async with Client(mcp_server) as client:
        with pytest.raises(ToolError, match="upstream safety policy"):
            await client.call_tool(
                "generate_image", {"prompt": "adult fashion", "model": "gpt-6-luna"}
            )
    assert len(captured) == 1  # 拒绝不重试
    assert not fake_storage.uploads


@pytest.mark.asyncio
async def test_luna_empty_result_retries_then_succeeds(mcp_server, fake_storage, monkeypatch):
    """首次无图（工具被上游剥除，模型回文字）时自动重试并成功出图。"""
    import base64

    encoded = base64.b64encode(_PNG_BYTES).decode()
    captured = patch_luna_responses(
        monkeypatch,
        sequence=[
            {"text": "我画了一只橘猫。"},  # 无 image_generation_call
            {"items": [{"result": encoded, "revised_prompt": "优化后的提示词"}]},
        ],
    )
    async with Client(mcp_server) as client:
        result = await client.call_tool(
            "generate_image", {"prompt": "画一只猫", "model": "gpt-6-luna"}
        )
    assert len(captured) == 2  # 重试了一次
    assert result.data["key"].endswith(".png")
    assert result.data["revised_prompt"] == "优化后的提示词"
    assert fake_storage.uploads


@pytest.mark.asyncio
async def test_luna_empty_result_error_includes_model_text(fake_storage, monkeypatch) -> None:
    """重试耗尽仍无图时，错误信息带上模型原话便于定位。"""
    from fastmcp import FastMCP

    from agent_app.config import ImageGenConfig, ImageVisionConfig
    from agent_app.tools.image_tools import register_image_tools

    mcp = FastMCP("Test")
    register_image_tools(
        mcp,
        config=ImageGenConfig(
            base_url="https://gemini.test",
            api_key=SecretStr("sk-test"),
            luna_retries=1,
        ),
        storage=fake_storage,
        vision_config=ImageVisionConfig(
            base_url="https://codex.test", api_key=SecretStr("sk-vision")
        ),
    )
    captured = patch_luna_responses(monkeypatch, text="我画了一只橘猫（SVG 描述）。")
    async with Client(mcp) as client:
        with pytest.raises(ToolError, match="model replied"):
            await client.call_tool("generate_image", {"prompt": "画一只猫", "model": "gpt-6-luna"})
    assert len(captured) == 2  # 初次 + 1 次重试
    assert not fake_storage.uploads


def test_validated_image_rejects_truncated_png():
    assert image_tools._validated_image(_PNG_BYTES) == ("png", "image/png")
    assert image_tools._validated_image(_PNG_BYTES[:20]) is None
    assert image_tools._validated_image(b"") is None

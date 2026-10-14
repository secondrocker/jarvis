"""图像内容识别 MCP 工具的进程内测试（FastMCP Client，不联网）。"""

import httpx
import pytest
from fakes import FakeObjectStorage
from fastmcp import Client
from fastmcp.exceptions import ToolError
from PIL import Image
from pydantic import SecretStr

from agent_app.config import Settings
from agent_app.tools import build_mcp_server, image_vision_tools

# 模型返回的识别结果样例（穿白大褂的女医生 → 人物/医护）。
_RECOGNITION_JSON = (
    '{"category": "人物", "subcategory": "医护", "legwear": "黑丝", "figure": "高挑", '
    '"subject": "穿白大褂的女医生", '
    '"clothing": "白色医生大褂、听诊器", "occupation": "医生", '
    '"gender": "女", "scene": "医院走廊", '
    '"description": "一位穿白大褂的女医生站在医院走廊。", '
    '"filename": "医院走廊 女医生！"}'
)


def _jpeg_bytes() -> bytes:
    """现场生成一张最小 JPEG，用于解码路径。"""
    import io

    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), "white").save(buffer, "JPEG")
    return buffer.getvalue()


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
    """预置两张待整理图片的 fake 存储。"""
    storage = FakeObjectStorage()
    storage.put(_jpeg_bytes(), key="uploads/photo-1.jpeg", content_type="image/jpeg")
    storage.put(_jpeg_bytes(), key="uploads/photo-2.png", content_type="image/png")
    return storage


@pytest.fixture
def vision_settings() -> Settings:
    """带图像识别配置的应用配置。"""
    return Settings.model_validate(
        {
            "openai": {"api_key": SecretStr("test-key"), "model": "gpt-4o-mini"},
            "image_vision": {
                "base_url": "https://gemini.test",
                "api_key": SecretStr("sk-test"),
            },
        }
    )


@pytest.fixture
def mcp_server(vision_settings, fake_storage):
    """注册了图像识别工具的聚合 MCP 服务。"""
    return build_mcp_server(settings=vision_settings, storage=fake_storage)


@pytest.fixture
def patch_llm(monkeypatch):
    """patch httpx：POST 返回识别 JSON，GET 返回 JPEG 字节。

    返回捕获字典，可改写识别回复与下载字节。
    """
    captured: dict = {"post_payload": None, "content": _RECOGNITION_JSON, "called_models": []}

    def fake_post(url, *, json, headers, timeout):  # noqa: ANN001
        captured["post_payload"] = {"url": url, "json": json, "headers": headers}
        captured["called_models"].append(json["model"])
        if json["model"] in captured.get("fail_models", set()):
            raise httpx.HTTPStatusError(
                "504 Gateway Time-out",
                request=httpx.Request("POST", url),
                response=httpx.Response(504),
            )
        return _FakeResponse(json_body={"choices": [{"message": {"content": captured["content"]}}]})

    def fake_get(url, *, timeout=None, follow_redirects=False):  # noqa: ANN001
        return _FakeResponse(content=_jpeg_bytes())

    monkeypatch.setattr(image_vision_tools.httpx, "post", fake_post)
    monkeypatch.setattr(image_vision_tools.httpx, "get", fake_get)
    # 重试退避会真实 sleep，测试中置空避免拖慢。
    monkeypatch.setattr(image_vision_tools.time, "sleep", lambda _: None)
    return captured


@pytest.mark.asyncio
async def test_recognize_image_by_key(mcp_server, fake_storage, patch_llm) -> None:
    async with Client(mcp_server) as client:
        result = await client.call_tool("recognize_image", {"key": "uploads/photo-1.jpeg"})

    data = result.data
    assert data["category"] == "人物"
    assert data["occupation"] == "医生"
    assert data["clothing"] == "白色医生大褂、听诊器"
    # 文件名被规范化为小写连字符 slug。
    assert data["filename"] == "医院走廊-女医生-黑丝-高挑"
    assert data["legwear"] == "黑丝"
    assert data["figure"] == "高挑"
    assert data["subcategory"] == "医护"
    assert data["suggested_key"] == "图片/人物/医护/黑丝/医院走廊-女医生-黑丝-高挑.jpg"
    # 请求体为多模态 content（text + image_url base64）。
    body = patch_llm["post_payload"]
    assert body["json"]["model"] == "gemini-3.8-flash"
    parts = body["json"]["messages"][0]["content"]
    assert parts[0]["type"] == "text"
    assert parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert body["headers"]["Authorization"] == "Bearer sk-test"


@pytest.mark.asyncio
async def test_recognize_image_falls_back_to_backup_model(
    mcp_server, fake_storage, patch_llm
) -> None:
    # 主模型 gemini-3.8-flash 持续 504 → 降级到 gemini-3.6-flash 后成功。
    patch_llm["fail_models"] = {"gemini-3.8-flash"}
    async with Client(mcp_server) as client:
        result = await client.call_tool("recognize_image", {"key": "uploads/photo-1.jpeg"})

    assert result.data["category"] == "人物"
    assert patch_llm["called_models"] == ["gemini-3.8-flash"] * 3 + ["gemini-3.6-flash"]
    assert patch_llm["post_payload"]["json"]["model"] == "gemini-3.6-flash"


@pytest.mark.asyncio
async def test_recognize_image_rejects_both_sources(mcp_server) -> None:
    async with Client(mcp_server) as client:
        with pytest.raises(ToolError):
            await client.call_tool(
                "recognize_image", {"url": "https://img.test/a.jpg", "key": "a.jpg"}
            )


@pytest.mark.asyncio
async def test_recognize_image_falls_back_to_other_category(
    mcp_server, fake_storage, patch_llm
) -> None:
    patch_llm["content"] = '{"category": "外星", "filename": "！！"}'
    async with Client(mcp_server) as client:
        result = await client.call_tool("recognize_image", {"key": "uploads/photo-2.png"})
    assert result.data["category"] == "其他"
    assert result.data["filename"] == "图片"


@pytest.mark.asyncio
async def test_organize_images_moves_and_renames(mcp_server, fake_storage, patch_llm) -> None:
    async with Client(mcp_server) as client:
        result = await client.call_tool("organize_images", {"path": "uploads"})

    data = result.data
    assert data["moved_count"] == 2
    # .jpeg 统一为 .jpg；不同扩展名同主干不冲突，各自成档。
    moved_keys = {item["target_key"] for item in data["moved"]}
    assert moved_keys == {
        "图片/人物/医护/黑丝/医院走廊-女医生-黑丝-高挑.jpg",
        "图片/人物/医护/黑丝/医院走廊-女医生-黑丝-高挑.png",
    }
    assert ("uploads/photo-1.jpeg", "图片/人物/医护/黑丝/医院走廊-女医生-黑丝-高挑.jpg") in (
        fake_storage.renames
    )
    assert "uploads/photo-1.jpeg" not in fake_storage.objects


@pytest.mark.asyncio
async def test_organize_images_dry_run_keeps_objects(mcp_server, fake_storage, patch_llm) -> None:
    async with Client(mcp_server) as client:
        result = await client.call_tool("organize_images", {"path": "uploads", "dry_run": True})

    data = result.data
    assert data["moved_count"] == 0
    assert len(data["plan"]) == 2
    assert fake_storage.renames == []
    assert "uploads/photo-1.jpeg" in fake_storage.objects


@pytest.mark.asyncio
async def test_organize_images_skips_non_image_keys(mcp_server, fake_storage, patch_llm) -> None:
    fake_storage.put(b"pdf-bytes", key="uploads/report.pdf", content_type="application/pdf")
    async with Client(mcp_server) as client:
        result = await client.call_tool("organize_images", {"path": "uploads"})
    assert result.data["moved_count"] == 2
    assert "uploads/report.pdf" in fake_storage.objects


@pytest.mark.asyncio
async def test_organize_images_collects_failures(mcp_server, fake_storage, patch_llm) -> None:
    # 识别服务返回不可解析文本 → 逐项失败汇总，不中断其他对象。
    patch_llm["content"] = "模型拒绝回答"
    async with Client(mcp_server) as client:
        result = await client.call_tool("organize_images", {"path": "uploads"})
    data = result.data
    assert data["moved_count"] == 0
    assert data["failed_count"] == 2
    assert fake_storage.renames == []


@pytest.mark.asyncio
async def test_organize_images_skips_already_correct_key(
    mcp_server, fake_storage, patch_llm
) -> None:
    fake_storage.put(
        _jpeg_bytes(),
        key="图片/人物/医护/黑丝/医院走廊-女医生-黑丝-高挑.jpg",
        content_type="image/jpeg",
    )
    async with Client(mcp_server) as client:
        result = await client.call_tool("organize_images", {"path": "图片/人物"})
    data = result.data
    assert data["moved_count"] == 0
    assert data["skipped"] == ["图片/人物/医护/黑丝/医院走廊-女医生-黑丝-高挑.jpg"]


@pytest.mark.asyncio
async def test_mcp_server_omits_tools_without_config(fake_storage) -> None:
    settings = Settings.model_validate(
        {"openai": {"api_key": SecretStr("test-key"), "model": "gpt-4o-mini"}}
    )
    server = build_mcp_server(settings=settings, storage=fake_storage)
    async with Client(server) as client:
        tools = await client.list_tools()
    names = {tool.name for tool in tools}
    assert "recognize_image" not in names
    assert "organize_images" not in names


@pytest.mark.asyncio
async def test_recognize_image_exposure_branch(mcp_server, fake_storage, patch_llm) -> None:
    # 裸露图片：子类强制私密写真，裸露程度写入文件名前缀。
    patch_llm["content"] = (
        '{"category": "人物", "subcategory": "通用", "exposure": "全裸", '
        '"legwear": "裸腿", "figure": "纤细", "filename": "湖边女性"}'
    )
    async with Client(mcp_server) as client:
        result = await client.call_tool("recognize_image", {"key": "uploads/photo-1.jpeg"})

    data = result.data
    assert data["exposure"] == "全裸"
    assert data["subcategory"] == "私密写真"
    assert data["filename"] == "全裸-湖边女性-裸腿-纤细"
    assert data["suggested_key"] == "图片/人物/私密写真/裸腿/全裸-湖边女性-裸腿-纤细.jpg"


@pytest.mark.asyncio
async def test_recognize_image_invalid_exposure_defaults_none(
    mcp_server, fake_storage, patch_llm
) -> None:
    # 非法裸露取值保守视为无裸露，走常规子类。
    patch_llm["content"] = (
        '{"category": "人物", "subcategory": "职场", "exposure": "非常裸", '
        '"legwear": "黑丝", "figure": null, "filename": "办公室女性"}'
    )
    async with Client(mcp_server) as client:
        result = await client.call_tool("recognize_image", {"key": "uploads/photo-1.jpeg"})

    data = result.data
    assert data["exposure"] == "无裸露"
    assert data["subcategory"] == "职场"
    assert data["filename"] == "办公室女性-黑丝"

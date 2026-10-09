"""基于 gemini-web2api 反代的图像生成 MCP 工具。

调用 OpenAI 兼容的 /v1/chat/completions（model=gemini-image），从回复正文中
解析图片 URL，下载字节后上传对象存储，返回可下载 URL。对象 key 的目录
根据生成内容（prompt 语义）自动归类到 images/<category>/ 下。

错误处理范式与 storage_tools.py 一致：AppError 转为 ToolError，不抛裸异常。
"""

import re
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import httpx
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import ValidationError

from agent_app.config import ImageGenConfig
from agent_app.errors import AppError, ErrorCode
from agent_app.infrastructure.storage import ObjectStorage
from agent_app.schemas.image_tools import ImageGenerateInput

# 下载生成图片的最大等待秒数。
_DOWNLOAD_TIMEOUT = 60.0

# 对象 key 的顶层目录前缀。
_KEY_PREFIX = "images"

# markdown 图片语法，如 ![alt](https://...png) 或 ![image](data:image/jpeg;base64,...)。
_MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)\s]+)\)")

# base64 data URL 的 mime → (扩展名, Content-Type)。
_DATA_URL_FULL_RE = re.compile(r"data:image/(png|jpeg|jpg|webp|gif);base64,([A-Za-z0-9+/=\s]+)")

_DATA_URL_TYPES: dict[str, tuple[str, str]] = {
    "png": ("png", "image/png"),
    "jpeg": ("jpg", "image/jpeg"),
    "jpg": ("jpg", "image/jpeg"),
    "webp": ("webp", "image/webp"),
    "gif": ("gif", "image/gif"),
}

# 裸 URL（兜底：markdown 解析不到时捕获正文里的 http(s) 链接）。
_BARE_URL_RE = re.compile(r"(https?://[^\s\"'<>\]\)]+)")

# prompt 关键词 → 对象存储子目录；按声明顺序首个命中生效。
# 中英双语关键词覆盖常见生成主题，未命中时回退 "generated"。
_CATEGORY_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("猫", "cat", "kitten"), "animals"),
    (("狗", "dog", "puppy"), "animals"),
    (("动物", "animal", "bird", "鱼", "fish", "熊猫", "panda"), "animals"),
    (("人", "肖像", "portrait", "character", "女孩", "男孩", "girl", "boy"), "portrait"),
    (("风景", "山水", "landscape", "mountain", "ocean", "森林", "forest"), "landscape"),
    (("建筑", "building", "城市", "city", "房子", "house"), "architecture"),
    (("logo", "标志", "图标", "icon"), "logo"),
    (("插画", "illustration", "卡通", "cartoon", "anime", "二次元"), "illustration"),
    (("食物", "美食", "food", "cake", "咖啡", "coffee"), "food"),
    (("产品", "product", "商品"), "product"),
    (("图表", "海报", "poster", "banner", "封面", "cover"), "design"),
)

# 图片字节魔数 → (扩展名, Content-Type)；无法从 URL 推断时用于嗅探。
_MAGIC_SIGNATURES: tuple[tuple[bytes, str, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "png", "image/png"),
    (b"\xff\xd8\xff", "jpg", "image/jpeg"),
    (b"GIF87a", "gif", "image/gif"),
    (b"GIF89a", "gif", "image/gif"),
    (b"RIFF", "webp", "image/webp"),
)


def _classify_category(prompt: str) -> str:
    """根据 prompt 关键词推断对象存储子目录名。

    参数:
        prompt: 已规范化的生成提示词。

    返回值:
        命中的英文目录名；未命中时返回 "generated"。
    """
    lowered = prompt.lower()
    for keywords, category in _CATEGORY_RULES:
        if any(keyword in lowered for keyword in keywords):
            return category
    return "generated"


def _compose_prompt(prompt: str, aspect_ratio: str | None) -> str:
    """把画面比例要求并入提示词（反代服务只接受自然语言指令）。"""
    if not aspect_ratio:
        return prompt
    return f"{prompt}（画面比例 {aspect_ratio}）"


def _extract_image_sources(content: str) -> list[dict[str, Any]]:
    """从回复正文解析图片来源：内嵌 base64 data URL > markdown http 图片 > 裸链接。

    真实反代服务把生成图以 ![image](data:image/jpeg;base64,...) 内嵌在正文里，
    同时可能伴随不可直接下载的占位 http 链接，故 data URL 优先。

    参数:
        content: chat completion 的 assistant 消息正文。

    返回值:
        图片来源列表：data URL 直接解码为 {"data", "extension", "content_type"}，
        http 链接保持为 {"url"}，由调用方下载并嗅探；保持出现顺序、去重。
    """
    sources: list[dict[str, Any]] = []
    seen_keys: set[str] = set()

    def _add_data(matched: re.Match[str]) -> None:
        mime, payload = matched.group(1).lower(), matched.group(2)
        if mime not in _DATA_URL_TYPES:
            return
        try:
            import base64

            data = base64.b64decode(re.sub(r"\s+", "", payload), validate=True)
        except ValueError:
            return
        key = f"{mime}:{len(data)}"
        if key not in seen_keys:
            seen_keys.add(key)
            extension, content_type = _DATA_URL_TYPES[mime]
            sources.append({"data": data, "extension": extension, "content_type": content_type})

    def _add_url(url: str) -> None:
        if url not in seen_keys:
            seen_keys.add(url)
            sources.append({"url": url})

    for matched in re.finditer(r"!\[[^\]]*\]\(data:image/[^)]+\)", content):
        inner = _DATA_URL_FULL_RE.search(matched.group(0))
        if inner:
            _add_data(inner)
    for url in _MARKDOWN_IMAGE_RE.findall(content):
        if not url.startswith("data:"):
            _add_url(url)
    if not sources:
        for url in _BARE_URL_RE.findall(content):
            _add_url(url)
    return sources


def _sniff_image(data: bytes) -> tuple[str, str] | None:
    """按魔数嗅探图片字节，返回 (扩展名, content_type)；非图片返回 None。"""
    for magic, extension, content_type in _MAGIC_SIGNATURES:
        if data.startswith(magic):
            if extension == "webp" and data[8:12] != b"WEBP":
                continue
            return extension, content_type
    return None


def _download(url: str) -> tuple[bytes, str, str] | None:
    """下载图片字节并识别扩展名与 Content-Type；非图片内容返回 None。

    反代正文可能包含返回 HTML 的占位链接，嗅探失败即跳过该来源。

    参数:
        url: 生成图片的可下载 URL。

    返回值:
        (字节, 扩展名, content_type) 三元组；下载失败或非图片时为 None。

    异常:
        AppError: 网络层失败时抛出 UPSTREAM_UNAVAILABLE。
    """
    try:
        response = httpx.get(url, timeout=_DOWNLOAD_TIMEOUT, follow_redirects=True)
        response.raise_for_status()
    except httpx.HTTPError as error:
        raise AppError(
            ErrorCode.UPSTREAM_UNAVAILABLE,
            "Generated image could not be downloaded",
        ) from error
    sniffed = _sniff_image(response.content)
    if sniffed is None:
        return None
    return response.content, *sniffed


def _request_generation(config: ImageGenConfig, prompt: str) -> str:
    """调用反代的 chat completions 生成图片，返回回复正文。

    参数:
        config: 图像生成服务配置。
        prompt: 已并入比例要求的完整提示词。

    返回值:
        assistant 消息正文（通常包含 markdown 图片链接）。

    异常:
        AppError: 请求失败、无 choices 或正文为空时抛出 UPSTREAM_UNAVAILABLE。
    """
    payload = {
        "model": config.model,
        "messages": [{"role": "user", "content": prompt}],
    }
    headers = {
        "Authorization": f"Bearer {config.api_key.get_secret_value()}",  # type: ignore[union-attr]
        "Content-Type": "application/json",
    }
    try:
        response = httpx.post(
            f"{config.base_url}/v1/chat/completions",  # type: ignore[operator]
            json=payload,
            headers=headers,
            timeout=config.timeout_seconds,
        )
        response.raise_for_status()
        body = response.json()
    except (httpx.HTTPError, ValueError) as error:
        raise AppError(
            ErrorCode.UPSTREAM_UNAVAILABLE,
            "Image generation service is temporarily unavailable",
        ) from error
    try:
        content = body["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as error:
        raise AppError(
            ErrorCode.UPSTREAM_UNAVAILABLE,
            "Image generation service returned an unexpected response",
        ) from error
    if not str(content).strip():
        raise AppError(
            ErrorCode.UPSTREAM_UNAVAILABLE,
            "Image generation service returned no content",
        )
    return str(content)


def register_image_tools(
    mcp: FastMCP,
    *,
    config: ImageGenConfig,
    storage: ObjectStorage,
) -> None:
    """把图像生成 tool 注册到给定 MCP 服务。

    参数:
        mcp: 待注册的 FastMCP 服务实例。
        config: 图像生成服务配置（base_url/api_key 必须已配置）。
        storage: 上传生成图片的对象存储。
    """

    @mcp.tool
    def generate_image(prompt: str, aspect_ratio: str | None = None) -> dict[str, Any]:
        """根据自然语言要求生成图片并上传对象存储，返回可下载 URL。

        prompt 为图片内容描述（支持中英文）；aspect_ratio 可选，取值
        1:1 / 2:3 / 3:2 / 3:4 / 4:3 / 9:16 / 16:9 / 21:9，缺省由模型自选。
        生成图片按内容自动归类到 images/<category>/<日期>/<id>.<ext>。

        Args:
            prompt: 图片内容描述。
            aspect_ratio: 画面比例（宽:高），可选。

        返回值:
            含 key、url（首张图）、images（全部图的 key/url 列表）的字典。
        """
        try:
            params = ImageGenerateInput(prompt=prompt, aspect_ratio=aspect_ratio)
        except ValidationError as error:
            detail = error.errors()[0]["msg"] if error.errors() else "Invalid image parameters"
            raise ToolError(str(detail)) from error
        full_prompt = _compose_prompt(params.prompt, params.aspect_ratio)
        try:
            content = _request_generation(config, full_prompt)
            sources = _extract_image_sources(content)
            category = _classify_category(params.prompt)
            date_dir = datetime.now(UTC).strftime("%Y%m%d")
            images: list[dict[str, Any]] = []
            for source in sources:
                if "data" in source:
                    data = source["data"]
                    extension = source["extension"]
                    content_type = source["content_type"]
                    origin = "inline"
                else:
                    downloaded = _download(source["url"])
                    if downloaded is None:
                        # 占位/失效链接（如返回 HTML 的页面），跳过。
                        continue
                    data, extension, content_type = downloaded
                    origin = source["url"]
                key = f"{_KEY_PREFIX}/{category}/{date_dir}/{uuid4().hex}.{extension}"
                storage.put(data, key=key, content_type=content_type)
                images.append({"key": key, "url": storage.download_url(key), "source": origin})
                if len(images) >= 4:
                    break
            if not images:
                raise AppError(
                    ErrorCode.UPSTREAM_UNAVAILABLE,
                    "No image was generated for this prompt",
                )
            return {
                "key": images[0]["key"],
                "url": images[0]["url"],
                "prompt": params.prompt,
                "aspect_ratio": params.aspect_ratio,
                "images": images,
            }
        except AppError as error:
            raise ToolError(error.public_message) from error

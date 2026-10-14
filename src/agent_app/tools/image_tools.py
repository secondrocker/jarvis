"""基于 gemini-web2api 反代的图像生成 MCP 工具。

调用 OpenAI 兼容的 /v1/chat/completions（model=gemini-image），从回复正文中
解析图片 URL，下载字节后上传对象存储，返回可下载 URL。对象 key 的目录
根据生成内容（prompt 语义）自动归类到 images/<category>/ 下。

错误处理范式与 storage_tools.py 一致：AppError 转为 ToolError，不抛裸异常。
"""

import json
import re
from datetime import UTC, datetime
from io import BytesIO
from typing import Any
from uuid import uuid4

import httpx
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from PIL import Image
from pydantic import ValidationError

from agent_app.config import ImageGenConfig, ImageVisionConfig
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


def _custom_key(path: str, extension: str, index: int) -> str:
    """根据自定义路径推导第 index 张图（0 起）的对象 key。

    path 可含文件名与扩展名：带已知图片扩展名时按原样使用；无扩展名时按
    识别出的格式补齐。多张图时在扩展名前追加 ``-2``/``-3`` 序号避免覆盖。

    参数:
        path: 调用方传入的自定义存储路径（已规范化、无首部斜杠）。
        extension: 本张图片识别出的扩展名。
        index: 图片序号（0 起，仅 >0 时追加序号）。

    返回值:
        最终对象 key。
    """
    known_exts = ("png", "jpg", "jpeg", "webp", "gif")
    lowered = path.lower()
    for ext in known_exts:
        suffix = f".{ext}"
        if lowered.endswith(suffix):
            stem, ext_with_dot = path[: -len(suffix)], path[-len(suffix) :]
            return path if index == 0 else f"{stem}-{index + 1}{ext_with_dot}"
    suffix = f".{extension}"
    return f"{path}{suffix}" if index == 0 else f"{path}-{index + 1}.{extension}"


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


# 上游安全拒绝的正文话术标记（中英双语，小写匹配）。
# gpt-6-luna 被内容政策拒绝时不一定带 refusal 字段或 content_filter 终止原因，
# 常以纯文字道歉回复（"抱歉，我不能生成……"），据此识别并给出明确错误。
_REFUSAL_MARKERS: tuple[str, ...] = (
    "抱歉",
    "对不起",
    "不能生成",
    "无法生成",
    "无法创建",
    "不能创建",
    "违反政策",
    "内容政策",
    "sorry",
    "cannot",
    "can't",
    "unable to",
    "not able to",
    "inappropriate",
    "policy",
)


def _looks_like_refusal(content: str) -> bool:
    """判断正文是否为上游内容政策的文字拒绝（而非正常图片说明）。"""
    lowered = content.lower()
    return any(marker in lowered for marker in _REFUSAL_MARKERS)


# 上游安全拒绝的统一公开消息：降级逻辑据此识别并跳过 fallback。
_SAFETY_REFUSAL_MESSAGE = "Image generation was refused by the upstream safety policy"


def _safety_refusal_error() -> AppError:
    """构造上游安全拒绝错误（gemini 与 luna 链路共用）。"""
    return AppError(ErrorCode.UPSTREAM_UNAVAILABLE, _SAFETY_REFUSAL_MESSAGE)


def _stream_chat_completion(
    vision: ImageVisionConfig,
    payload: dict[str, Any],
    headers: dict[str, str],
) -> tuple[dict[str, Any], str | None]:
    """以 SSE 流式调用 chat completions 并聚合为非流式等价的消息结构。

    流式模式下 tool_calls 的 function.arguments 按分片到达，需按 index
    累积后整体解析；content/refusal 同理累积拼接。

    参数:
        vision: image_vision 网关配置。
        payload: 已含 stream=True 的请求体。
        headers: 认证等请求头。

    返回值:
        (聚合后的 message 字典, finish_reason)；finish_reason 可能缺失为 None。
    """
    content_parts: list[str] = []
    refusal_parts: list[str] = []
    tools: dict[int, dict[str, Any]] = {}
    finish_reason: str | None = None
    with httpx.stream(
        "POST",
        f"{vision.base_url}/v1/chat/completions",  # type: ignore[operator]
        json=payload,
        headers=headers,
        timeout=vision.timeout_seconds,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            line = line.strip()
            if not line.startswith("data: "):
                continue
            data = line[len("data: ") :]
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except ValueError:
                continue
            choice = (chunk.get("choices") or [None])[0]
            if not choice:
                continue
            finish_reason = choice.get("finish_reason") or finish_reason
            delta = choice.get("delta") or {}
            if delta.get("content"):
                content_parts.append(str(delta["content"]))
            if delta.get("refusal"):
                refusal_parts.append(str(delta["refusal"]))
            for call in delta.get("tool_calls") or []:
                index = call.get("index") or 0
                slot = tools.setdefault(index, {"name": None, "arguments": ""})
                function = call.get("function") or {}
                if function.get("name"):
                    slot["name"] = function["name"]
                slot["arguments"] += function.get("arguments") or ""
    message: dict[str, Any] = {
        "content": "".join(content_parts),
        "refusal": "".join(refusal_parts) or None,
        "tool_calls": [{"function": slot} for _, slot in sorted(tools.items())],
    }
    return message, finish_reason


def _request_luna_generation(
    vision: ImageVisionConfig,
    model: str,
    prompt: str,
) -> list[dict[str, Any]]:
    """调用 image_vision 网关上的 gpt-6-luna 生成图片，返回图片来源列表。

    gpt-6-luna 通过 OpenAI 的 image_generation 工具生图：请求需携带
    tools=[{"type": "image_generation"}]，图片以 tool_calls 里
    image_generation(result=<base64>) 的形式返回，正文则是文字说明。

    参数:
        vision: image_vision 网关配置（base_url/api_key 必须已配置）。
        model: 网关上请求的模型名（默认 gpt-6-luna）。
        prompt: 已并入比例要求的完整提示词。

    返回值:
        图片来源列表（data/extension/content_type），与
        _extract_image_sources 的 data 项同构；无图片时为空列表。

    异常:
        AppError: 请求失败或响应结构异常时抛出 UPSTREAM_UNAVAILABLE。
    """
    import base64
    import binascii

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "tools": [{"type": "image_generation"}],
        # 必须流式：网关对非流式请求有约 60 秒整体超时，复杂 prompt 生图
        # 超过该阈值会稳定 504；流式分块保持连接活跃可绕过（实测 95s 成功）。
        "stream": True,
    }
    headers = {
        "Authorization": f"Bearer {vision.api_key.get_secret_value()}",  # type: ignore[union-attr]
        "Content-Type": "application/json",
    }
    try:
        message, finish_reason = _stream_chat_completion(vision, payload, headers)
    except (httpx.HTTPError, ValueError) as error:
        raise AppError(
            ErrorCode.UPSTREAM_UNAVAILABLE,
            "Image generation service is temporarily unavailable",
        ) from error
    if message.get("refusal") or finish_reason == "content_filter":
        raise _safety_refusal_error()
    sources: list[dict[str, Any]] = []
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        if function.get("name") != "image_generation":
            continue
        try:
            arguments = json.loads(function.get("arguments") or "{}")
            result = arguments.get("result")
            if not isinstance(result, str) or not result.strip():
                continue
            data = base64.b64decode(re.sub(r"\s+", "", result), validate=True)
        except (ValueError, TypeError, AttributeError, binascii.Error):
            continue
        sniffed = _validated_image(data)
        if sniffed:
            sources.append({"data": data, "extension": sniffed[0], "content_type": sniffed[1]})
    # 兜底：个别网关可能直接把图片内嵌在正文中返回。
    content = message.get("content")
    if not sources and isinstance(content, str):
        sources = _extract_image_sources(content)
    # 仍未取得图片且正文是拒绝性话术时，视为被上游内容政策拒绝，
    # 抛出明确的安全拒绝错误（区别于笼统的"未生成图片"）。
    if not sources and isinstance(content, str) and _looks_like_refusal(content):
        raise _safety_refusal_error()
    return sources


def _validated_image(data: bytes) -> tuple[str, str] | None:
    """只接受可完整解码的图片，拒绝空数据、截断及伪造文件头。"""
    sniffed = _sniff_image(data)
    if not sniffed:
        return None
    try:
        with Image.open(BytesIO(data)) as image:
            image.verify()
        with Image.open(BytesIO(data)) as image:
            image.load()
    except (OSError, ValueError, SyntaxError):
        return None
    return sniffed


def register_image_tools(
    mcp: FastMCP,
    *,
    config: ImageGenConfig,
    storage: ObjectStorage,
    vision_config: ImageVisionConfig | None = None,
) -> None:
    """把图像生成 tool 注册到给定 MCP 服务。

    参数:
        mcp: 待注册的 FastMCP 服务实例。
        config: 图像生成服务配置（base_url/api_key 必须已配置）。
        storage: 上传生成图片的对象存储。
        vision_config: gpt-6-luna 生图复用的 image_vision 网关配置；
            未提供时 model=gpt-6-luna 的调用会返回明确错误。
    """

    @mcp.tool
    def generate_image(
        prompt: str,
        aspect_ratio: str | None = None,
        path: str | None = None,
        model: str = "gpt-6-luna",
    ) -> dict[str, Any]:
        """根据自然语言要求生成图片并上传对象存储，返回可下载 URL。

        prompt 为图片内容描述（支持中英文）；aspect_ratio 可选，取值
        1:1 / 2:3 / 3:2 / 3:4 / 4:3 / 9:16 / 16:9 / 21:9，缺省由模型自选。
        model 可选，取值 gpt-6-luna（默认，经 image_vision 网关的
        image_generation 工具）或 gemini（gemini-web2api 反代）。
        显式选择 gemini 失败时自动降级重试 gpt-6-luna（返回值 fallback=true）。
        生成图片按内容自动归类到 images/<category>/<日期>/<id>.<ext>；
        path 可选，为自定义存储路径（含文件名，如 "assets/poster.png"），
        传入时固定使用该路径：无扩展名自动补齐，多张图在文件名后追加
        -2/-3 序号。

        Args:
            prompt: 图片内容描述。
            aspect_ratio: 画面比例（宽:高），可选。
            path: 自定义存储路径（含文件名），可选；缺省按内容自动归类。
            model: 生图模型，gpt-6-luna（默认）或 gemini。

        返回值:
            含 key、url（首张图）、prompt、aspect_ratio、path、model
            （实际生成模型）、fallback（是否降级到 luna）、images 的字典；
            images 为全部上传图（最多 4 张）的列表，每项含 key、url、
            source（内嵌 base64 为 "inline"，外链为原始 URL）。
        """
        try:
            params = ImageGenerateInput(
                prompt=prompt, aspect_ratio=aspect_ratio, path=path, model=model
            )
        except ValidationError as error:
            detail = error.errors()[0]["msg"] if error.errors() else "Invalid image parameters"
            raise ToolError(str(detail)) from error
        full_prompt = _compose_prompt(params.prompt, params.aspect_ratio)
        try:
            fallback_used = False
            if params.model == "gpt-6-luna":
                if vision_config is None or vision_config.base_url is None:
                    raise AppError(
                        ErrorCode.UPSTREAM_UNAVAILABLE,
                        "gpt-6-luna image generation is not configured",
                    )
                sources = _request_luna_generation(vision_config, config.luna_model, full_prompt)
                used_model = "gpt-6-luna"
            else:
                try:
                    content = _request_generation(config, full_prompt)
                    sources = _extract_image_sources(content)
                    if not sources:
                        if _looks_like_refusal(content):
                            # gemini 反代常以纯文字拒绝（"我无法生成……"），
                            # 明确归类为安全拒绝且不触发降级重试。
                            raise _safety_refusal_error()
                        raise AppError(
                            ErrorCode.UPSTREAM_UNAVAILABLE,
                            "No image was generated for this prompt",
                        )
                    used_model = "gemini"
                except AppError as error:
                    # 安全拒绝不降级：违规 prompt 不应再交给另一个模型重试。
                    if error.public_message == _SAFETY_REFUSAL_MESSAGE:
                        raise
                    # gemini 生图失败时自动降级到 gpt-6-luna（可配置关闭）。
                    if not config.luna_fallback or vision_config is None:
                        raise
                    if vision_config.base_url is None:
                        raise
                    sources = _request_luna_generation(
                        vision_config, config.luna_model, full_prompt
                    )
                    used_model = "gpt-6-luna"
                    fallback_used = True
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
                validated = _validated_image(data)
                if not validated:
                    continue
                extension, content_type = validated
                if params.path:
                    key = _custom_key(params.path, extension, len(images))
                else:
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
                "path": params.path,
                # model 返回实际生成模型（降级时与请求模型不同），fallback 标记是否降级。
                "model": used_model,
                "fallback": fallback_used,
                "images": images,
            }
        except AppError as error:
            raise ToolError(error.public_message) from error

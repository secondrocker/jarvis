"""基于 Gemini vision 反代的图像内容识别 MCP 工具。

调用 OpenAI 兼容的 /v1/chat/completions（多模态 content，内嵌 base64 图片），
识别图片内容（衣着、职业、人物、场景等），产出结构化 JSON：

- recognize_image：识别单张图片（url 或 S3 key），返回内容描述与建议的
  归档路径/文件名；
- organize_images：批量整理指定前缀下的 S3 图片——逐张识别后移动到
  ``<target_prefix>/<类目>/<子类>/<袜装>/<裸露>-<内容文件名>-<袜装>-<身材>.<ext>``，实现按类目归并 +
  内容化重命名。

错误处理范式与 image_tools.py 一致：AppError 转为 ToolError，不抛裸异常。
"""

import base64
import io
import json
import re
import time
from typing import Any

import httpx
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from PIL import Image
from pydantic import ValidationError

from agent_app.config import ImageVisionConfig
from agent_app.errors import AppError, ErrorCode
from agent_app.infrastructure.storage import ObjectStorage
from agent_app.schemas.image_tools import ImageOrganizeInput, ImageRecognizeInput

# 送识别图片的最大体积（与 grade_homework 一致）。
_MAX_IMAGE_MB = 8

# 识别结果 JSON 的兜底提取正则。
_JSON_RE = re.compile(r"\{[\s\S]*\}")

# 归档类目白名单（中文目录名）；模型输出不在其中时回退 "其他"。
_CATEGORIES: tuple[str, ...] = (
    "人物",
    "动物",
    "风景",
    "建筑",
    "食物",
    "商品",
    "插画动漫",
    "设计",
    "文档",
    "交通",
    "其他",
)

# 视为图片的 key 扩展名集合。
_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".gif")

# 文件名合法字符（小写字母/数字/连字符/中文）；其余字符替换为连字符。
_FILENAME_INVALID_RE = re.compile(r"[^a-z0-9\u4e00-\u9fff-]+")

# 下载对象/图片的等待秒数。
_DOWNLOAD_TIMEOUT = 60.0

# 识别请求对网关瞬时故障（504 等）的总尝试次数与重试退避基数（秒）。
_REQUEST_ATTEMPTS = 3
_RETRY_BACKOFF_SECONDS = 3.0

# 归档根目录前缀（中文）。
_KEY_PREFIX = "图片"

# 不建袜装目录层的取值（不明显/裤装/无人物等；含规范化兜底值"图片"）。
_LEGWEAR_SKIP = {"不明显", "裤装", "无人物", "无", "图片"}

# 不写入文件名前缀的身材取值。
_FIGURE_SKIP = {"不明显", "无人物", "无", "图片"}

# 裸露程度：非"无裸露"时子类强制为私密写真，并把程度写进文件名前缀。
_EXPOSURE_LEVELS = ("全裸", "上身裸露", "下身裸露", "局部裸露", "无裸露")
_EXPOSURE_NONE = "无裸露"
_EXPOSURE_SUBCATEGORY = "私密写真"

# 识别提示词：要求严格 JSON 输出，目录与文件名均为中文。
_RECOGNIZE_PROMPT = "\n".join(
    [
        "请仔细识别这张图片并只输出一个 JSON 对象"
        "（不要 markdown 代码块、不要多余文字），字段如下：",
        "{",
        '  "category": "一级类目（中文），必须从以下选择：人物, 动物, 风景, 建筑, '
        '食物, 商品, 插画动漫, 设计, 文档, 交通, 其他",',
        '  "subcategory": "二级子类（中文，2-6 个字，按风格/职业/着装等最显著特征判断，'
        "如：医护/护士/运动员/学生制服/职场/汉服古风/旗袍/婚纱/黑色礼服/时尚日常/"
        '宠物/野生动物/街景/室内等；无合适子类给 通用）",',
        '  "subject": "画面主体的一句话概括（中文，如：穿白大褂的女医生）",',
        '  "exposure": "私密部位裸露程度（从以下选最匹配：全裸, 上身裸露, 下身裸露, '
        '局部裸露, 无裸露；无人物为 无裸露）",',
        '  "legwear": "腿部着装（中文，从以下选最匹配：黑丝, 白丝, 肉色丝袜, 网格袜, '
        '彩色丝袜, 过膝袜, 短袜, 裸腿, 裤装, 不明显；无人物为 无人物）",',
        '  "figure": "身材类型（中文 2-4 字，如：高挑, 纤细, 匀称, 丰满, 微胖, 健美；'
        '不明显或无人物为 null）",',
        '  "clothing": "人物衣着描述（中文；无人物为 null）",',
        '  "occupation": "推测的人物职业（中文；无法推测为 null）",',
        '  "gender": "人物性别（男/女/多人/未知；无人物为 null）",',
        '  "scene": "拍摄场景/环境（中文，如：医院走廊、户外街道）",',
        '  "description": "完整的图片内容描述（中文，一两句话）",',
        '  "filename": "基于图片内容的中文文件名主干，4-12 个字、不含扩展名与路径分隔符，'
        '如 医院走廊女医生（不要包含袜装/身材/裸露词，系统会自动拼接）"',
        "}",
    ]
)


def _compress_image(data: bytes, max_edge: int) -> str:
    """把图片字节兜底压缩为不超过 max_edge 的 JPEG 并返回 base64。

    参数:
        data: 原始图片字节。
        max_edge: 压缩后最大边长（像素）。

    返回值:
        JPEG base64 字符串（无 data URL 前缀）。

    异常:
        AppError: 图片超限或无法解码时抛出 INVALID_PARAMETERS。
    """
    if len(data) > _MAX_IMAGE_MB * 1024 * 1024:
        raise AppError(ErrorCode.INVALID_PARAMETERS, "image exceeds size limit")
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
        if max(image.size) > max_edge:
            image.thumbnail((max_edge, max_edge))
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=85)
    except Exception as error:  # noqa: BLE001 - 统一转为参数错误
        raise AppError(ErrorCode.INVALID_PARAMETERS, "cannot decode image") from error
    return base64.b64encode(buffer.getvalue()).decode()


def _download(url: str, timeout: float = _DOWNLOAD_TIMEOUT) -> bytes:
    """下载字节；失败抛出安全的 INVALID_PARAMETERS。

    参数:
        url: 可下载 URL（预签名或公网）。
        timeout: 下载等待秒数。

    返回值:
        响应字节。

    异常:
        AppError: 下载失败时抛出 INVALID_PARAMETERS。
    """
    try:
        response = httpx.get(url, timeout=timeout, follow_redirects=True)
        response.raise_for_status()
    except httpx.HTTPError as error:
        raise AppError(ErrorCode.INVALID_PARAMETERS, "image could not be downloaded") from error
    return response.content


def _request_once(
    config: ImageVisionConfig,
    model: str,
    image_b64: str,
) -> dict[str, Any]:
    """用指定模型发一次识别请求，成功返回解析后的响应体。

    参数:
        config: 图像识别服务配置。
        model: 本次请求使用的模型名（主模型或降级模型）。
        image_b64: JPEG base64 字符串。

    返回值:
        chat completion 响应体字典。

    异常:
        httpx.HTTPError / ValueError: 网络失败或响应非 JSON（由调用方重试/降级）。
    """
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _RECOGNIZE_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                ],
            }
        ],
    }
    headers = {
        "Authorization": f"Bearer {config.api_key.get_secret_value()}",  # type: ignore[union-attr]
        "Content-Type": "application/json",
    }
    response = httpx.post(
        f"{config.base_url}/v1/chat/completions",  # type: ignore[operator]
        json=payload,
        headers=headers,
        timeout=config.timeout_seconds,
    )
    response.raise_for_status()
    return response.json()  # type: ignore[no-any-return]


def _request_recognition(config: ImageVisionConfig, image_b64: str) -> str:
    """调用 vision 模型识别图片，返回回复正文；主模型失败时降级到备用模型。

    策略：主模型重试 _REQUEST_ATTEMPTS 次（线性退避）仍失败时，切换
    fallback_model 再试 _REQUEST_ATTEMPTS 次；两者皆失败才报
    UPSTREAM_UNAVAILABLE。网关对部分模型需要登录态账号且偶发 504，
    主/备模型互为兜底。

    参数:
        config: 图像识别服务配置。
        image_b64: JPEG base64 字符串。

    返回值:
        assistant 消息正文（应为 JSON 文本）。

    异常:
        AppError: 全部尝试失败、无 choices 或正文为空时抛出 UPSTREAM_UNAVAILABLE。
    """
    # 主模型与降级模型相同则只尝试一组，避免无意义双倍重试。
    models = [config.model]
    if config.fallback_model and config.fallback_model != config.model:
        models.append(config.fallback_model)
    last_error: Exception | None = None
    body: dict[str, Any] | None = None
    for model in models:
        for attempt in range(_REQUEST_ATTEMPTS):
            try:
                body = _request_once(config, model, image_b64)
                break
            except (httpx.HTTPError, ValueError) as error:
                last_error = error
                if attempt < _REQUEST_ATTEMPTS - 1:
                    time.sleep(_RETRY_BACKOFF_SECONDS * (attempt + 1))
        if body is not None:
            break
    if body is None:
        raise AppError(
            ErrorCode.UPSTREAM_UNAVAILABLE,
            "Image recognition service is temporarily unavailable",
        ) from last_error
    try:
        content = body["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as error:
        raise AppError(
            ErrorCode.UPSTREAM_UNAVAILABLE,
            "Image recognition service returned an unexpected response",
        ) from error
    # 部分网关把多模态回复放进 content 列表，先归一为纯文本。
    if isinstance(content, list):
        text = "".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
    else:
        text = str(content)
    if not text.strip():
        raise AppError(
            ErrorCode.UPSTREAM_UNAVAILABLE,
            "Image recognition service returned no content",
        )
    return text


def _sanitize_filename(raw: str) -> str:
    """把模型给出的名称规范化为合法中文 slug；无效时回退 "图片"。

    参数:
        raw: 模型输出的文件名（可能含中文/空格/下划线）。

    返回值:
        仅含小写字母、数字、连字符的文件名主干（截断到 60 字符）。
    """
    slug = _FILENAME_INVALID_RE.sub("-", raw.strip().lower().replace("_", "-"))
    slug = re.sub(r"-{2,}", "-", slug).strip("-")[:60]
    return slug or "图片"


def _parse_recognition(content: str) -> dict[str, Any]:
    """从回复正文解析识别结果 JSON 并做字段规范化。

    参数:
        content: assistant 消息正文（JSON 或夹杂 JSON 的文本）。

    返回值:
        含 category/subject/clothing/occupation/gender/scene/description/
        filename 的字典；解析失败时抛出 UPSTREAM_UNAVAILABLE。

    异常:
        AppError: 无法解析出 JSON 对象时抛出 UPSTREAM_UNAVAILABLE。
    """
    # 部分网关直接把 content 解析为 dict 返回；若为 {"text": "..."} 包装则解包。
    if isinstance(content, dict):
        if "category" not in content and isinstance(content.get("text"), str):
            content = content["text"]
        else:
            payload: Any = content
            return _normalize_payload(payload)
    if isinstance(content, str):
        payload = _load_payload_text(content)
        return _normalize_payload(payload)
    raise AppError(
        ErrorCode.UPSTREAM_UNAVAILABLE,
        "Image recognition service returned unparsable content",
    )


def _load_payload_text(content: str) -> Any:
    """把回复正文解析为 JSON 对象；容忍 markdown 围栏与夹杂文本。"""
    try:
        return json.loads(content)
    except json.JSONDecodeError as error:
        matched = _JSON_RE.search(content)
        if not matched:
            raise AppError(
                ErrorCode.UPSTREAM_UNAVAILABLE,
                "Image recognition service returned unparsable content",
            ) from error
        try:
            return json.loads(matched.group(0))
        except json.JSONDecodeError as inner:
            raise AppError(
                ErrorCode.UPSTREAM_UNAVAILABLE,
                "Image recognition service returned unparsable content",
            ) from inner


def _normalize_payload(payload: Any) -> dict[str, Any]:
    """把已解析的 JSON 载荷规范化为识别结果字段字典。"""
    if not isinstance(payload, dict):
        raise AppError(
            ErrorCode.UPSTREAM_UNAVAILABLE,
            "Image recognition service returned unparsable content",
        )
    category = str(payload.get("category") or "").strip().lower()
    if category not in _CATEGORIES:
        category = "其他"
    filename = _sanitize_filename(str(payload.get("filename") or ""))
    subcategory = _sanitize_filename(str(payload.get("subcategory") or ""))
    if subcategory == "图片":
        # 文件名规范化兜底值，作为目录名无意义。
        subcategory = "通用"
    legwear = _sanitize_filename(str(payload.get("legwear") or ""))
    figure = _sanitize_filename(str(payload.get("figure") or ""))
    exposure = str(payload.get("exposure") or "").strip()
    if exposure not in _EXPOSURE_LEVELS:
        # 未返回或非法取值时保守视为无裸露，避免误分支。
        exposure = _EXPOSURE_NONE
    if exposure != _EXPOSURE_NONE:
        # 裸露图片强制归入私密写真子类，裸露程度写进文件名前缀。
        subcategory = _EXPOSURE_SUBCATEGORY
    return {
        "category": category,
        "subcategory": subcategory,
        "legwear": legwear,
        "figure": figure,
        "exposure": exposure,
        "subject": payload.get("subject"),
        "clothing": payload.get("clothing"),
        "occupation": payload.get("occupation"),
        "gender": payload.get("gender"),
        "scene": payload.get("scene"),
        "description": payload.get("description"),
        # 特征拼接进最终文件名：<裸露>-<主干>-<袜装>-<身材>，不明显项省略。
        "filename": _compose_filename(filename, legwear, figure, exposure),
    }


def _legwear_dir(recognition: dict[str, Any]) -> str:
    """返回袜装子目录段（含尾部斜杠）；不明显时为空串。

    参数:
        recognition: 规范化后的识别结果。

    返回值:
        形如 ``黑丝/`` 的目录段或空串。
    """
    legwear = str(recognition.get("legwear") or "")
    if legwear and legwear not in _LEGWEAR_SKIP:
        return f"{legwear}/"
    return ""


def _compose_filename(stem: str, legwear: str, figure: str, exposure: str = "") -> str:
    """把裸露/袜装/身材特征拼接为最终文件名主干。

    参数:
        stem: 内容主干（模型给出的文件名）。
        legwear: 规范化后的袜装类型。
        figure: 规范化后的身材类型。
        exposure: 裸露程度；非"无裸露"时作为前缀。

    返回值:
        形如 ``全裸湖边女性-裸腿-纤细`` / ``医院走廊护士-黑丝-高挑`` 的文件名主干；
        不明显项自动省略。
    """
    parts: list[str] = []
    if exposure and exposure != _EXPOSURE_NONE:
        parts.append(exposure)
    parts.append(stem)
    if legwear and legwear not in _LEGWEAR_SKIP:
        parts.append(legwear)
    if figure and figure not in _FIGURE_SKIP:
        parts.append(figure)
    return "-".join(part for part in parts if part)


def _recognize(config: ImageVisionConfig, data: bytes) -> dict[str, Any]:
    """压缩 + 调用模型识别一段图片字节，返回规范化后的识别结果。"""
    image_b64 = _compress_image(data, config.max_image_edge)
    content = _request_recognition(config, image_b64)
    return _parse_recognition(content)


def _ext_from_key(key: str) -> str | None:
    """从对象 key 推断图片扩展名；非图片返回 None。"""
    lowered = key.lower()
    for ext in _IMAGE_EXTS:
        if lowered.endswith(ext):
            # 统一 jpeg → jpg，避免 .jpeg/.jpg 混用。
            return ".jpg" if ext == ".jpeg" else ext
    return None


def _unique_target_key(target_dir: str, filename: str, ext: str, taken: set[str]) -> str:
    """在已占用 key 集合内推导不冲突的目标 key（冲突时追加 -2/-3）。

    参数:
        target_dir: 目标目录前缀（含尾部斜杠）。
        filename: 内容文件名主干。
        ext: 含点的扩展名。
        taken: 本次运行已产生（或已存在）的目标 key 集合。

    返回值:
        唯一的目标 key。
    """
    candidate = f"{target_dir}{filename}{ext}"
    index = 2
    while candidate in taken:
        candidate = f"{target_dir}{filename}-{index}{ext}"
        index += 1
    taken.add(candidate)
    return candidate


def register_image_vision_tools(
    mcp: FastMCP,
    *,
    config: ImageVisionConfig,
    storage: ObjectStorage,
) -> None:
    """把图像内容识别 tool 注册到给定 MCP 服务。

    参数:
        mcp: 待注册的 FastMCP 服务实例。
        config: 图像识别服务配置（base_url/api_key 必须已配置）。
        storage: 读取/整理图片对象的对象存储。
    """

    @mcp.tool
    def recognize_image(
        url: str | None = None,
        key: str | None = None,
    ) -> dict[str, Any]:
        """识别单张图片的内容（衣着、职业、场景等），返回结构化描述。

        url 与 key 二选一：url 为可下载的图片地址；key 为对象存储中的
        图片 key。识别维度包括画面主体、人物衣着、推测职业、性别、
        场景与整体描述；另给出建议的归档类目、袜装/身材特征与中文内容
        文件名。传入 key 时额外返回 suggested_key
        （图片/<类目>/<子类>/<袜装>/<文件名>.<ext>，袜装不明显时省略该层）。

        Args:
            url: 可下载的图片 URL，可选。
            key: 对象存储中的图片 key，可选。

        返回值:
            含 category/subcategory/legwear/figure/subject/clothing/occupation/
            gender/scene/description/filename，以及（传 key 时）key、
            suggested_key、url 的字典。
        """
        try:
            params = ImageRecognizeInput(url=url, key=key)
            if params.key is not None:
                data = _download(storage.download_url(params.key))
            else:
                data = _download(params.url)  # type: ignore[arg-type]
            result = _recognize(config, data)
            if params.key is not None:
                ext = _ext_from_key(params.key) or ".jpg"
                suggested = (
                    f"{_KEY_PREFIX}/{result['category']}/{result['subcategory']}/"
                    f"{_legwear_dir(result)}{result['filename']}{ext}"
                )
                return {
                    **result,
                    "key": params.key,
                    "suggested_key": suggested,
                    "url": storage.download_url(suggested),
                }
            return result
        except AppError as error:
            raise ToolError(error.public_message) from error
        except ValidationError as error:
            detail = error.errors()[0]["msg"] if error.errors() else "Invalid parameters"
            raise ToolError(str(detail)) from error

    @mcp.tool
    def organize_images(
        path: str = "",
        target_prefix: str = _KEY_PREFIX,
        limit: int | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """按内容批量归并整理 S3 图片：识别后移动到类目目录并重命名。

        列出 path 前缀下的图片对象，逐张调用视觉模型识别（衣着/职业/
        场景/裸露程度等），移动到
        ``<target_prefix>/<类目>/<子类>/<袜装>/<裸露>-<内容文件名>-<袜装>-<身材>.<ext>``；
        .jpeg 统一为 .jpg，目标冲突时追加 -2/-3 序号；与源路径相同的跳过。
        dry_run=True 只返回整理计划不实际移动。逐项尽力而为，结果汇总
        moved/failed 明细。

        Args:
            path: 待整理的目录前缀，空字符串表示 bucket 根。
            target_prefix: 归档根目录前缀，默认 "images"。
            limit: 本次最多处理的对象数（默认取配置 batch_limit=20）。
            dry_run: 仅生成计划不执行移动。

        返回值:
            含 plan/moved（每项含 source_key/target_key/category/filename/
            recognition）、skipped、failed（每项含 key/error）、moved_count、
            failed_count 的字典；dry_run 时计划在 plan 字段。
        """
        try:
            params = ImageOrganizeInput(
                path=path,
                target_prefix=target_prefix,
                limit=limit if limit is not None else config.batch_limit,
                dry_run=dry_run,
            )
            images = [
                entry.key
                for entry in storage.list(params.path)
                if _ext_from_key(entry.key) is not None
            ][: params.limit]
            prefix = params.target_prefix.rstrip("/") + "/"
            # 已占用的目标 key：目标目录下既有对象 + 本次已分配的。
            taken = {entry.key for entry in storage.list(prefix)}
            plan: list[dict[str, Any]] = []
            moved: list[dict[str, Any]] = []
            skipped: list[str] = []
            failed: list[dict[str, str]] = []
            for source_key in images:
                try:
                    data = _download(storage.download_url(source_key))
                    recognition = _recognize(config, data)
                except AppError as error:
                    failed.append({"key": source_key, "error": error.public_message})
                    continue
                ext = _ext_from_key(source_key) or ".jpg"
                target_dir = (
                    f"{prefix}{recognition['category']}/{recognition['subcategory']}/"
                    f"{_legwear_dir(recognition)}"
                )
                # 源对象移动后会腾空原路径，不参与目标冲突判定。
                taken.discard(source_key)
                target_key = _unique_target_key(target_dir, recognition["filename"], ext, taken)
                item = {
                    "source_key": source_key,
                    "target_key": target_key,
                    "category": recognition["category"],
                    "filename": recognition["filename"],
                    "recognition": {
                        field: recognition[field]
                        for field in (
                            "subcategory",
                            "legwear",
                            "figure",
                            "exposure",
                            "subject",
                            "clothing",
                            "occupation",
                            "gender",
                            "scene",
                            "description",
                        )
                    },
                }
                if target_key == source_key:
                    skipped.append(source_key)
                    taken.discard(target_key)
                    continue
                plan.append(item)
                if params.dry_run:
                    continue
                try:
                    storage.rename(source_key, target_key)
                    moved.append({**item, "url": storage.download_url(target_key)})
                except AppError as error:
                    failed.append({"key": source_key, "error": error.public_message})
            return {
                "plan": plan,
                "moved": moved,
                "skipped": skipped,
                "failed": failed,
                "moved_count": len(moved),
                "failed_count": len(failed),
                "dry_run": params.dry_run,
            }
        except AppError as error:
            raise ToolError(error.public_message) from error
        except ValidationError as error:
            detail = error.errors()[0]["msg"] if error.errors() else "Invalid parameters"
            raise ToolError(str(detail)) from error

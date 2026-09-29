"""批改作业图的三个节点: fetch_image / grade_llm / calc_verify。"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from PIL import Image

from agent_app.errors import AppError, ErrorCode
from agent_app.workflows.grade_homework.calc import (
    CalcError,
    assert_public_http_url,
    calc_one,
    numbers_match,
)
from agent_app.workflows.grade_homework.prompts import (
    GRADE_SYSTEM_PROMPT,
    GRADE_USER_PROMPT,
)
from agent_app.workflows.grade_homework.schemas import GradeResult, GradeState

logger = logging.getLogger(__name__)

_JSON_RE = re.compile(r"\{[\s\S]*\}")


def make_fetch_image_node(
    *,
    max_image_mb: int = 8,
    max_edge: int = 2048,
    timeout_seconds: float = 30.0,
) -> Callable[[GradeState], dict[str, Any]]:
    """返回下载图片并兜底压缩为 JPEG base64 的同步节点。"""

    def fetch_image(state: GradeState) -> dict[str, Any]:
        url = state.get("url", "")
        try:
            assert_public_http_url(url)
        except CalcError as error:
            raise AppError(ErrorCode.INVALID_PARAMETERS, str(error)) from error
        try:
            response = httpx.get(url, timeout=timeout_seconds, follow_redirects=True)
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise AppError(
                ErrorCode.INVALID_PARAMETERS,
                f"failed to download image: {error}",
            ) from error
        data = response.content
        if len(data) > max_image_mb * 1024 * 1024:
            raise AppError(ErrorCode.INVALID_PARAMETERS, "image exceeds size limit")
        try:
            image = Image.open(io.BytesIO(data)).convert("RGB")
            if max(image.size) > max_edge:
                image.thumbnail((max_edge, max_edge))
            buffer = io.BytesIO()
            image.save(buffer, "JPEG", quality=85)
        except Exception as error:  # noqa: BLE001 - 统一转为参数错误
            raise AppError(ErrorCode.INVALID_PARAMETERS, f"cannot decode image: {error}") from error
        return {"image_b64": base64.b64encode(buffer.getvalue()).decode()}

    return fetch_image


def make_grade_llm_node(
    model: BaseChatModel,
) -> Callable[[GradeState], Awaitable[dict[str, Any]]]:
    """返回调用 vision 模型结构化批改的异步节点。"""
    structured = model.with_structured_output(GradeResult)

    def _messages(image_b64: str) -> list[Any]:
        return [
            SystemMessage(content=GRADE_SYSTEM_PROMPT),
            HumanMessage(
                content=[
                    {"type": "text", "text": GRADE_USER_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                ]
            ),
        ]

    def _fallback_parse(text: str) -> GradeResult:
        """结构化输出不可用时,退回原始 JSON 文本解析。"""
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            match = _JSON_RE.search(text)
            if not match:
                raise
            payload = json.loads(match.group(0))
        return GradeResult.model_validate(payload)

    async def grade_llm(state: GradeState) -> dict[str, Any]:
        image_b64 = state.get("image_b64", "")
        if not image_b64:
            raise AppError(ErrorCode.EXECUTION_FAILED, "grade_llm: missing image")
        # 首选结构化输出;网关不支持 function call 时退回原始文本解析。
        try:
            result: GradeResult = await structured.ainvoke(_messages(image_b64))
        except Exception as structured_error:  # noqa: BLE001
            logger.warning("structured output failed, falling back: %s", structured_error)
            try:
                raw = await model.ainvoke(_messages(image_b64))
                result = _fallback_parse(raw.text)
            except Exception as error:  # noqa: BLE001
                raise AppError(
                    ErrorCode.EXECUTION_FAILED,
                    f"grading model failed: {error}",
                ) from error
        return {"result": result.model_dump()}

    return grade_llm


def _fix_stats(result: dict[str, Any]) -> dict[str, Any]:
    """以 questions 实际情况重算统计,防止模型数错。"""
    questions = result.get("questions") or []
    correct = sum(1 for q in questions if q.get("verdict") == "correct" and q.get("result_correct") is True)
    wrong = sum(1 for q in questions if q.get("verdict") == "wrong")
    total = len(questions)
    result["total_questions"] = total
    result["correct_count"] = correct
    result["wrong_count"] = wrong
    result["accuracy"] = round(correct / total, 4) if total else None
    if result.get("score") is not None and result.get("full_score"):
        try:
            result["accuracy"] = round(float(result["score"]) / float(result["full_score"]), 4)
        except (TypeError, ValueError, ZeroDivisionError):
            pass
    return result


def verify_math(result: dict[str, Any]) -> dict[str, Any]:
    """数学题用计算工具复算,不一致时以后端计算为准改写判定。"""
    if result.get("subject") != "math":
        return result
    for question in result.get("questions") or []:
        expressions = question.get("expressions") or []
        verified: list[dict[str, Any]] = []
        changed = False
        for item in expressions:
            raw = item.get("raw") if isinstance(item, dict) else None
            if not raw:
                continue
            outcome = calc_one(str(raw))
            claimed = item.get("claimed")
            outcome["claimed"] = claimed
            outcome["calc_verified"] = bool(
                outcome.get("ok")
                and claimed is not None
                and numbers_match(outcome.get("value"), claimed)
            )
            verified.append(outcome)
            if outcome.get("ok") and claimed is not None and not outcome["calc_verified"]:
                previous = question.get("verdict")
                question["result_correct"] = False
                if previous == "correct":
                    question["verdict"] = "wrong"
                    question["comment"] = "计算工具复验:结果不正确。"
                changed = True
                detail = question.get("error_detail") or {}
                detail.setdefault("error_type", "calculation")
                value = outcome.get("value")
                detail["expected"] = value.get("decimal") if isinstance(value, dict) else value
                detail.setdefault("reason", "经计算工具精确复算,等式不成立")
                question["error_detail"] = detail
        question["expressions"] = verified or None
        if changed:
            logger.info("calc corrected question %s", question.get("no"))
    return result


def make_calc_verify_node() -> Callable[[GradeState], dict[str, Any]]:
    """返回对结果做计算复算与统计修正的同步节点。"""

    def calc_verify(state: GradeState) -> dict[str, Any]:
        result = dict(state.get("result") or {})
        result = _fix_stats(result)
        result = verify_math(result)
        result = _fix_stats(result)  # calc 纠正后重算统计
        return {"result": result}

    return calc_verify

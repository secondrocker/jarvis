"""批改结果的服务端确定性后处理：契约校验、统计重算、复算纠错、人审标记。

agent 只负责识别与判断，凡是能确定性计算的都不交给模型：
统计以 questions 实际内容重算，数学题用 :mod:`.calc` 精确复算，
人工复核标记由 :func:`apply_review_policy` 统一裁定（对齐"AI 只出预批、
教师确认后生效"的合规要求）。
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from typing import Any

from pydantic import ValidationError

from agent_app.deep_agents.grade_homework.calc import (
    calc_one,
    numbers_match,
    numeric_value,
)
from agent_app.deep_agents.grade_homework.schemas import GradeResult
from agent_app.errors import AppError, ErrorCode

logger = logging.getLogger(__name__)

# 上报给调用方的复核原因上限（避免逐题罗列把响应撑大）。
_REVIEW_REASON_LIMIT = 8


def fix_stats(result: dict[str, Any]) -> dict[str, Any]:
    """以 questions 实际情况重算统计,防止模型数错。"""
    questions = result.get("questions") or []
    correct = sum(
        1 for q in questions if q.get("verdict") == "correct" and q.get("result_correct") is True
    )
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
    """数学题用计算工具复算,数值确不符时以后端计算为准改写判定。

    expressions 复核产物 ``calc_verified`` 三态:
    ``True`` 数值一致; ``False`` 确定不符(翻判为错); ``None`` 无法复算
    (算式本身不可求值,或 claimed 非数值)——无法复算不作为翻判依据,
    判定仍以模型的过程/结果判定为准。
    claimed 带单位(如 ``24（名）``)按纯数值比对,不算不符。
    """
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
            if not outcome.get("ok") or claimed is None or numeric_value(claimed) is None:
                # 算式不可求值或 claimed 非数值: 无法复算,不据此翻判
                outcome["calc_verified"] = None
                verified.append(outcome)
                continue
            outcome["calc_verified"] = numbers_match(outcome.get("value"), claimed)
            verified.append(outcome)
            if not outcome["calc_verified"]:
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


def apply_review_policy(result: dict[str, Any]) -> dict[str, Any]:
    """按复核策略裁定 needs_human_review / review_reasons（服务端唯一裁定权）。

    触发条件：学科无法确认、题目不可读或低置信、模型自行标记需复核、
    开放性/主观题、未识别到题目、执行过程降级。政策依据：教育部
    《关于加强义务教育学校作业管理的通知》（全批全改、及时反馈）与
    《教师生成式人工智能应用指引》（AI 不得直接将作文/开放性作业的
    自动批改结果作为学生最终评价）。
    """
    reasons: list[str] = []
    questions = result.get("questions") or []
    if result.get("subject") == "other":
        reasons.append("学科无法确认，建议人工确认学科归属")
    if not questions:
        reasons.append("未识别到可批改题目，请确认图片是否清晰、完整")
    for question in questions:
        no = question.get("no")
        if question.get("verdict") == "unreadable":
            reasons.append(f"第{no}题题干或字迹无法确认")
        if question.get("confidence") == "low":
            reasons.append(f"第{no}题识别置信度低")
        if question.get("answer_type") == "open":
            reasons.append(f"第{no}题为开放性/主观题，需教师确认后作为最终评价")
        if question.get("needs_review"):
            reasons.append(f"第{no}题：{question.get('review_reason') or '模型标记需人工确认'}")
    if result.get("degraded"):
        reasons.append("批改过程发生降级，结果可能不完整")

    deduped = list(dict.fromkeys(reasons))[:_REVIEW_REASON_LIMIT]
    result["needs_human_review"] = bool(deduped)
    result["review_reasons"] = deduped
    return result


def _format_errors(error: ValidationError) -> list[str]:
    """把校验错误压成不含原始输入的简洁描述（可安全回传给模型）。"""
    return [
        f"{'.'.join(str(part) for part in item.get('loc', ()))}: {item.get('msg', '')}"
        for item in error.errors()[:8]
    ]


def normalize_result(payload: Any) -> tuple[dict[str, Any] | None, list[str]]:
    """校验 payload 是否符合公开契约。

    参数:
        payload: 模型给出的批改结果（dict 或已构造的 GradeResult）。

    返回值:
        ``(规范化字典, [])`` 校验通过；``(None, [错误描述])`` 校验失败。
    """
    if isinstance(payload, GradeResult):
        return payload.model_dump(), []
    try:
        return GradeResult.model_validate(payload).model_dump(), []
    except ValidationError as error:
        return None, _format_errors(error)


def _postprocess(normalized: dict[str, Any]) -> dict[str, Any]:
    """对已通过契约校验的结果做全部确定性后处理。"""
    result = fix_stats(normalized)
    result = verify_math(result)
    result = fix_stats(result)  # calc 纠正后重算统计
    return apply_review_policy(result)


def finalize_result(payload: Any) -> dict[str, Any]:
    """校验并后处理一份批改结果；契约不合法时抛公开错误。

    参数:
        payload: 待校验的批改结果字典或 GradeResult。

    返回值:
        统计已重算、数学题已复算、人审标记已裁定的结果字典。

    异常:
        AppError: 契约校验失败时抛 EXECUTION_FAILED。
    """
    normalized, errors = normalize_result(payload)
    if normalized is None:
        logger.warning("grade result failed validation: %s", errors)
        raise AppError(ErrorCode.EXECUTION_FAILED, "grading result failed validation")
    return _postprocess(normalized)


def iter_json_objects(text: str) -> Iterator[Any]:
    """按出现顺序产出文本中所有可解析的 JSON 对象（跳过夹带的说明文字）。"""
    decoder = json.JSONDecoder()
    index = 0
    while True:
        start = text.find("{", index)
        if start < 0:
            return
        try:
            payload, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            index = start + 1
            continue
        yield payload
        index = max(end, start + 1)


def finalize_answer(answer: str) -> dict[str, Any]:
    """从 agent 最终回答里取出批改契约并后处理。

    回答中可能有多个 JSON 对象（中间推演、工具回显），取最后一个能通过
    契约校验的对象；一个都过不了则抛公开错误。

    参数:
        answer: agent 的最终文本回答。

    返回值:
        统计已重算、数学题已复算、人审标记已裁定的结果字典。

    异常:
        AppError: 回答中没有符合契约的结果时抛 EXECUTION_FAILED。
    """
    candidates = [payload for payload in iter_json_objects(answer) if isinstance(payload, dict)]
    last_errors: list[str] = []
    for payload in reversed(candidates):
        normalized, errors = normalize_result(payload)
        if normalized is not None:
            return _postprocess(normalized)
        last_errors = errors
    logger.warning(
        "grading answer has no valid result payload: %s", last_errors or "no json object"
    )
    raise AppError(ErrorCode.EXECUTION_FAILED, "grading agent returned no usable result")

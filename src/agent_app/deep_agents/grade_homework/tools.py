"""批改 agent 的确定性工具：算式精确复算与结果契约校验。

技能（SKILL.md）只提供流程与判断标准，无法保证确定性；数值复算与
契约校验必须由这里的代码完成——这也是 agent 与旧工作流共用的同一份
实现（执行器在返回前会再跑一次，见 :mod:`.postprocess`）。
"""

from typing import Any

from langchain_core.tools import BaseTool, tool

from agent_app.deep_agents.grade_homework.calc import calc_one, numbers_match, numeric_value
from agent_app.deep_agents.grade_homework.postprocess import finalize_result, normalize_result


@tool
def verify_expressions(expressions: list[dict[str, Any]]) -> dict[str, Any]:
    """用后端精确计算工具复算学生书写的算式，不要自己心算。

    传入你从图片逐字转写出的算式列表，每项形如
    ``{"raw": "12x12", "claimed": "154"}``：raw 为学生手写算式的原样转写
    （可含单位、等号、中文），claimed 为学生写的结果（纯数值，不要带单位）。
    返回每条的 ok/value/calc_verified；calc_verified 为 false 表示数值确定
    不符（该表达式判错），为 null 表示算式无法求值或 claimed 非数值
    （无法复算，不据此翻判）。

    Args:
        expressions: 待复算的算式列表，每项含 raw 与可选 claimed。
    """
    verified: list[dict[str, Any]] = []
    for item in expressions:
        raw = item.get("raw") if isinstance(item, dict) else None
        if not raw:
            continue
        outcome = calc_one(str(raw))
        claimed = item.get("claimed")
        outcome["claimed"] = claimed
        if not outcome.get("ok") or claimed is None or numeric_value(claimed) is None:
            outcome["calc_verified"] = None
        else:
            outcome["calc_verified"] = numbers_match(outcome.get("value"), claimed)
        verified.append(outcome)
    return {"results": verified}


@tool
def check_grading_result(result: dict[str, Any]) -> dict[str, Any]:
    """提交批改结果做契约校验与统计修正；必须在给出最终回答前调用。

    服务端会用这份结果重算统计（total/correct/wrong/accuracy）、复算数学题
    并对开放性题目打人工复核标记。校验不通过时返回 ok=false 与具体字段
    错误，请修正后再次调用；ok=true 时返回 normalized（最终会返回给调用方
    的完整结果），最终回答里必须原样包含该结果，不要再自行改动其中的数字。

    Args:
        result: 完整批改结果，字段见 grading-orchestration 技能的契约示例。
    """
    normalized, errors = normalize_result(result)
    if normalized is None:
        return {"ok": False, "errors": errors}
    final = finalize_result(normalized)
    return {
        "ok": True,
        "normalized": final,
        "needs_human_review": final["needs_human_review"],
        "review_reasons": final["review_reasons"],
    }


def create_grade_agent_tools() -> list[BaseTool]:
    """返回批改 agent 可用的确定性工具列表。

    返回值:
        算式复算与结果校验两个工具（技能文件负责流程，工具负责确定性）。
    """
    return [verify_expressions, check_grading_result]

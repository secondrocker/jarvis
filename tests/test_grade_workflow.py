"""grade_homework 工作流单元/集成测试(替身模型,不访问网络)。"""

from __future__ import annotations

import io
from typing import Any

import httpx
import pytest
from PIL import Image

from agent_app.workflows.grade_homework.schemas import GradeResult
from agent_app.workflows.grade_homework.nodes import (
    make_calc_verify_node,
    make_grade_llm_node,
)


class _FakeRunnable:
    """返回固定 GradeResult 的结构化输出替身。"""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.calls: list[Any] = []

    async def ainvoke(self, messages: Any, config: Any = None) -> GradeResult:
        self.calls.append(messages)
        return GradeResult.model_validate(self._payload)


class FakeGradeModel:
    """with_structured_output → FakeRunnable 的最小模型替身。"""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.runnable = _FakeRunnable(payload)

    def with_structured_output(self, schema: Any) -> _FakeRunnable:
        return self.runnable


def _jpeg_bytes() -> bytes:
    image = Image.new("RGB", (32, 32), "white")
    buffer = io.BytesIO()
    image.save(buffer, "JPEG")
    return buffer.getvalue()


MOCK_LLM_RESULT = {
    "subject": "math",
    "subject_confidence": 0.95,
    "summary": "整体不错。",
    "score": None,
    "full_score": None,
    "questions": [
        {
            "no": 1,
            "question_text": "25 × 4 = ?",
            "student_answer": "100",
            "result_correct": True,
            "verdict": "correct",
            "expressions": [{"raw": "25 × 4", "claimed": "100"}],
        },
        {
            "no": 2,
            "question_text": "12 × 12 = ?",
            "student_answer": "154",
            "result_correct": True,
            "verdict": "correct",
            "expressions": [{"raw": "12 × 12", "claimed": "154"}],
        },
        {
            "no": 3,
            "question_text": "3/4 + 1/2 = ?",
            "student_answer": "5/4",
            "result_correct": True,
            "verdict": "correct",
            "expressions": [{"raw": "3/4 + 1/2", "claimed": "5/4"}],
        },
    ],
}


async def test_grade_llm_then_calc_verify_corrects_llm_mistake():
    model = FakeGradeModel(MOCK_LLM_RESULT)
    grade = make_grade_llm_node(model)  # type: ignore[arg-type]
    state: dict[str, Any] = {"image_b64": "eHg="}
    update = await grade(state)
    assert update["result"]["subject"] == "math"
    assert len(update["result"]["questions"]) == 3

    verify = make_calc_verify_node()
    verified = verify({"result": update["result"]})["result"]
    questions = verified["questions"]
    # LLM 把 12*12=154 判正确 → calc 必须纠正
    assert questions[0]["verdict"] == "correct"
    assert questions[1]["verdict"] == "wrong"
    assert questions[1]["result_correct"] is False
    assert questions[1]["error_detail"]["expected"] == 144
    assert questions[2]["expressions"][0]["calc_verified"] is True
    # 统计以复算后为准
    assert verified["correct_count"] == 2
    assert verified["wrong_count"] == 1
    assert verified["accuracy"] == round(2 / 3, 4)


async def test_full_graph_with_fetched_image(monkeypatch: pytest.MonkeyPatch):
    from agent_app.workflows.grade_homework import build_grade_homework_graph

    def fake_get(url: str, **_: Any) -> httpx.Response:
        return httpx.Response(200, content=_jpeg_bytes(), request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", fake_get)
    graph = build_grade_homework_graph(FakeGradeModel(MOCK_LLM_RESULT))  # type: ignore[arg-type]
    final = await graph.ainvoke({"url": "https://example.com/hw.jpg"})
    assert final["result"]["questions"][1]["verdict"] == "wrong"
    assert final["result"]["correct_count"] == 2


def test_fetch_rejects_private_address():
    from agent_app.errors import AppError
    from agent_app.workflows.grade_homework.nodes import make_fetch_image_node

    fetch = make_fetch_image_node()
    with pytest.raises(AppError):
        fetch({"url": "http://127.0.0.1:8000/hw.jpg"})

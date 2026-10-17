"""批改 agent 与真实 deepagents 运行时的集成测试（ScriptedModel，不联网）。

锁定四件事：图片经执行器注入后 agent 能跑通完整工具回环、最终结果被服务端
收敛为公开契约、技能（主代理与子代理共用容器）注入 system prompt、
主代理可通过 task 工具派发职能子代理。
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from fakes import ScriptedModel
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from PIL import Image

from agent_app.deep_agents.grade_homework import create_grade_homework_agent
from agent_app.deep_agents.grade_homework.executor import GradeHomeworkExecutor
from agent_app.orchestration.executors import ExecutionContext
from agent_app.schemas.events import EventType

_SKILL_ROOT = Path(__file__).resolve().parents[2] / "src" / "agent_app" / "skills"

_GRADE_PAYLOAD: dict[str, Any] = {
    "subject": "math",
    "summary": "计算基本准确。",
    "questions": [
        {
            "no": 1,
            "question_text": "12×12=?",
            "student_answer": "154",
            "verdict": "correct",
            "result_correct": True,
            "expressions": [{"raw": "12×12", "claimed": "154"}],
        }
    ],
}


@pytest.fixture
def captured_systems(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """按调用顺序捕获每次模型调用的 system prompt（含子代理）。"""
    captured: list[str] = []

    def _capture(self, messages, stop=None, run_manager=None, **kwargs):
        from langchain_core.outputs import ChatGeneration, ChatResult

        for message in messages:
            if message.type == "system":
                content = message.content
                if isinstance(content, list):
                    content = "\n".join(
                        block.get("text", "") for block in content if isinstance(block, dict)
                    )
                captured.append(str(content))
                break
        self.calls.append(list(messages))
        script = self.scripts.pop(0)
        if isinstance(script, dict):
            from langchain_core.messages import AIMessage

            message = AIMessage(
                content=script.get("content", ""),
                tool_calls=[script["tool_call"]] if script.get("tool_call") else [],
            )
        else:
            from langchain_core.messages import AIMessage

            message = script if isinstance(script, AIMessage) else AIMessage(content=str(script))
        return ChatResult(generations=[ChatGeneration(message=message)])

    monkeypatch.setattr(ScriptedModel, "_generate", _capture)
    return captured


def _jpeg_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), "white").save(buffer, "JPEG")
    return buffer.getvalue()


def _context(message: str, emit=None) -> ExecutionContext:
    return ExecutionContext(
        message=message,
        messages=[HumanMessage(content=message)],
        parameters={},
        config={"configurable": {"thread_id": "grade-it-1"}},
        emit=emit or (lambda _: None),
    )


@pytest.mark.asyncio
async def test_executor_runs_real_agent_and_finalizes_contract(
    monkeypatch: pytest.MonkeyPatch, captured_systems: list[str]
) -> None:
    """真实运行时跑完"校验工具 → 最终回答"回环，服务端复算纠正模型判定。"""

    def fake_get(url: str, **_: Any) -> httpx.Response:
        return httpx.Response(200, content=_jpeg_bytes(), request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", fake_get)
    model = ScriptedModel.from_scripts(
        [
            {
                "tool_call": {
                    "name": "check_grading_result",
                    "args": {"result": _GRADE_PAYLOAD},
                    "id": "tc-check",
                }
            },
            "批改完成：\n\n```json\n" + json.dumps(_GRADE_PAYLOAD, ensure_ascii=False) + "\n```",
        ]
    )
    runtime = create_grade_homework_agent(
        model=model,
        checkpointer=MemorySaver(),
        skill_root=_SKILL_ROOT,
    )
    events: list[Any] = []
    executor = GradeHomeworkExecutor(runtime=runtime)

    result = await executor.run(_context("https://example.com/hw.jpg", emit=events.append))

    # 模型判 12×12=154 为正确 → 服务端精确复算翻判并重算统计。
    question = result["questions"][0]
    assert question["verdict"] == "wrong"
    assert question["error_detail"]["expected"] == 144
    assert result["wrong_count"] == 1
    assert result["accuracy"] == 0.0
    # 事件流保留工具回环（前端可见进度）。
    tool_events = [event for event in events if event.type is EventType.TOOL_STARTED]
    assert any(event.data["tool_name"] == "check_grading_result" for event in tool_events)
    # 技能被注入到主代理 system prompt。
    system = captured_systems[0]
    assert "## Skills System" in system
    assert "grading-orchestration" in system
    assert "math-grading" in system


@pytest.mark.asyncio
async def test_subagent_dispatch_uses_same_skill_container(
    captured_systems: list[str],
) -> None:
    """主代理经 task 工具派发 error_analyst，子代理读到同一技能容器。"""
    model = ScriptedModel.from_scripts(
        [
            {
                "tool_call": {
                    "name": "task",
                    "args": {"description": "诊断错因", "subagent_type": "error_analyst"},
                    "id": "tc-task",
                }
            },
            "错因：口诀记错（calculation）。",
            "批改完成：\n\n```json\n" + json.dumps(_GRADE_PAYLOAD, ensure_ascii=False) + "\n```",
        ]
    )
    runtime = create_grade_homework_agent(
        model=model,
        checkpointer=MemorySaver(),
        skill_root=_SKILL_ROOT,
    )
    history = [
        HumanMessage(
            content=[
                {"type": "text", "text": "批改这张作业图"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}},
            ]
        )
    ]
    executor = GradeHomeworkExecutor(runtime=runtime)

    result = await executor.run(
        ExecutionContext(
            message="看下这份作业",
            messages=history,
            parameters={},
            config={"configurable": {"thread_id": "grade-it-2"}},
            emit=lambda _: None,
        )
    )

    assert result["subject"] == "math"
    # 第一次是主代理、第二次是 error_analyst 子代理：都拿到了技能清单。
    assert len(captured_systems) >= 2
    assert "error-diagnosis" in captured_systems[1]
    assert "## Skills System" in captured_systems[1]

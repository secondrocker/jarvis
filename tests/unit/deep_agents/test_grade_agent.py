"""批改作业 agent 的单元测试（替身运行时/替身网络，不访问外部服务）。"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from PIL import Image
from pydantic import ValidationError

import agent_app
from agent_app.deep_agents import grade_homework as grade_mod
from agent_app.deep_agents.grade_homework import images as images_mod
from agent_app.deep_agents.grade_homework import postprocess as postprocess_mod
from agent_app.deep_agents.grade_homework.executor import GradeHomeworkExecutor
from agent_app.deep_agents.grade_homework.schemas import DiagnosisResult, FeedbackReport
from agent_app.deep_agents.grade_homework.subagents import build_grade_homework_subagents
from agent_app.deep_agents.grade_homework.tools import (
    check_grading_result,
    create_grade_agent_tools,
    verify_expressions,
)
from agent_app.errors import AppError, ErrorCode
from agent_app.orchestration.executors import ExecutionContext

MOCK_LLM_RESULT: dict[str, Any] = {
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


def _jpeg_bytes() -> bytes:
    image = Image.new("RGB", (32, 32), "white")
    buffer = io.BytesIO()
    image.save(buffer, "JPEG")
    return buffer.getvalue()


def _answer(payload: dict[str, Any]) -> str:
    return f"批改完成。\n\n```json\n{json.dumps(payload, ensure_ascii=False)}\n```\n"


class _FakeGradeRuntime:
    """产生固定回答的 Deep Agent runtime 替身（记录收到的输入）。"""

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.input_received: dict[str, Any] | None = None

    async def astream(self, input, config, *, stream_mode, **kwargs):
        self.input_received = input
        yield ("messages", (AIMessageChunk(content=self.answer), {}))


def _context(message: str, **overrides: Any) -> ExecutionContext:
    defaults: dict[str, Any] = {
        "message": message,
        "messages": [HumanMessage(content=message)],
        "parameters": {},
        "config": {"configurable": {"thread_id": "grade-1"}},
        "emit": lambda _: None,
    }
    return ExecutionContext(**{**defaults, **overrides})


# ---------------------------------------------------------------- 输入与图片


def test_extract_urls_splits_message_and_prefers_parameters() -> None:
    assert images_mod.extract_urls(
        "https://a.com/1.jpg,\nhttps://a.com/2.jpg  https://a.com/3.jpg"
    ) == ["https://a.com/1.jpg", "https://a.com/2.jpg", "https://a.com/3.jpg"]
    assert images_mod.extract_urls("x", {"urls": ["https://b.com/1.jpg"]}) == [
        "https://b.com/1.jpg"
    ]
    assert images_mod.extract_urls("x", {"url": "https://b.com/legacy.jpg"}) == [
        "https://b.com/legacy.jpg"
    ]
    assert images_mod.extract_urls("第二题错在哪") == []
    # 追问里夹带的非 URL 片段被忽略，真正的 URL 仍被取出。
    assert images_mod.extract_urls("看下 https://a.com/1.jpg 这道题") == ["https://a.com/1.jpg"]


def test_validate_urls_accepts_legacy_and_rejects_empty() -> None:
    assert images_mod.validate_urls(["https://a.com/1.jpg"]).urls == ["https://a.com/1.jpg"]
    with pytest.raises(ValidationError):
        images_mod.validate_urls([])
    with pytest.raises(ValidationError):
        images_mod.validate_urls([f"https://a.com/{index}.jpg" for index in range(9)])
    # 兼容旧契约的单图字段。
    legacy = grade_mod.GradeInput.model_validate({"url": "https://a.com/legacy.jpg"})
    assert legacy.urls == ["https://a.com/legacy.jpg"]


def test_fetch_image_data_urls_compresses_to_jpeg(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get(url: str, **_: Any) -> httpx.Response:
        return httpx.Response(200, content=_jpeg_bytes(), request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", fake_get)
    urls = ["https://example.com/hw-1.jpg", "https://example.com/hw-2.jpg"]
    data_urls = images_mod.fetch_image_data_urls(urls)
    assert len(data_urls) == 2
    assert all(url.startswith("data:image/jpeg;base64,") for url in data_urls)


def test_fetch_rejects_private_address_and_oversize(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(AppError) as error:
        images_mod.fetch_image_data_urls(["http://127.0.0.1:8000/hw.jpg"])
    assert error.value.code is ErrorCode.INVALID_PARAMETERS

    def fake_get(url: str, **_: Any) -> httpx.Response:
        return httpx.Response(200, content=b"x" * 2048, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", fake_get)
    with pytest.raises(AppError):
        images_mod.fetch_image_data_urls(["https://example.com/big.jpg"], max_image_mb=1)


def test_attach_images_replaces_last_human_message() -> None:
    messages = images_mod.attach_images([HumanMessage(content="https://a.com/1.jpg")], ["data:x"])
    assert len(messages) == 1
    blocks = messages[0].content
    assert blocks[0] == {"type": "text", "text": "https://a.com/1.jpg"}
    assert blocks[1]["image_url"]["url"] == "data:x"
    assert images_mod.messages_have_images(messages) is True


def test_attach_images_appends_when_last_is_not_human() -> None:
    messages = images_mod.attach_images([AIMessage(content="上一轮说明")], ["data:y"])
    assert len(messages) == 2
    assert isinstance(messages[-1], HumanMessage)
    assert images_mod.messages_have_images([AIMessage(content="无图")]) is False


# ---------------------------------------------------------------- 结果后处理


def test_finalize_result_recomputes_stats_and_corrects_math() -> None:
    result = postprocess_mod.finalize_result(MOCK_LLM_RESULT)
    questions = result["questions"]
    assert questions[0]["verdict"] == "correct"
    # 模型把 12*12=154 判正确 → 服务端复算必须纠正
    assert questions[1]["verdict"] == "wrong"
    assert questions[1]["result_correct"] is False
    assert questions[1]["error_detail"]["expected"] == 144
    assert questions[2]["expressions"][0]["calc_verified"] is True
    assert result["correct_count"] == 2
    assert result["wrong_count"] == 1
    assert result["accuracy"] == round(2 / 3, 4)


def test_finalize_answer_takes_last_valid_payload() -> None:
    chatty = '先看第一版 {"subject": "other"} 但字段不全。\n' + _answer(MOCK_LLM_RESULT)
    result = postprocess_mod.finalize_answer(chatty)
    assert result["subject"] == "math"
    assert result["total_questions"] == 3


def test_finalize_answer_without_contract_raises() -> None:
    with pytest.raises(AppError) as error:
        postprocess_mod.finalize_answer("这道题做对了，继续保持！")
    assert error.value.code is ErrorCode.EXECUTION_FAILED


def test_finalize_result_rejects_invalid_payload() -> None:
    with pytest.raises(AppError) as error:
        postprocess_mod.finalize_result({"subject": "physics", "questions": []})
    assert error.value.code is ErrorCode.EXECUTION_FAILED


def test_review_policy_flags_unreadable_open_and_other_subject() -> None:
    payload = {
        "subject": "other",
        "questions": [
            {
                "no": 1,
                "verdict": "unreadable",
                "confidence": "low",
                "needs_review": True,
                "review_reason": "字迹被涂改遮挡",
            },
            {"no": 2, "verdict": "partial", "answer_type": "open"},
        ],
    }
    result = postprocess_mod.finalize_result(payload)
    assert result["needs_human_review"] is True
    reasons = " ".join(result["review_reasons"])
    assert "学科无法确认" in reasons
    assert "第1题题干或字迹无法确认" in reasons
    assert "字迹被涂改遮挡" in reasons
    assert "第2题为开放性/主观题" in reasons


def test_review_policy_silent_for_clean_objective_result() -> None:
    result = postprocess_mod.finalize_result(MOCK_LLM_RESULT)
    assert result["needs_human_review"] is False
    assert result["review_reasons"] == []


def test_review_policy_flags_empty_questions() -> None:
    result = postprocess_mod.finalize_result({"subject": "math", "questions": []})
    assert result["needs_human_review"] is True
    assert any("未识别到可批改题目" in reason for reason in result["review_reasons"])


# ---------------------------------------------------------------- 确定性工具


def test_verify_expressions_tool_reports_mismatch() -> None:
    outcome = verify_expressions.invoke(
        {"expressions": [{"raw": "9×11", "claimed": "90（名）"}, {"raw": "25×4", "claimed": "100"}]}
    )
    results = outcome["results"]
    assert results[0]["calc_verified"] is False
    assert results[0]["value"] == 99
    assert results[1]["calc_verified"] is True


def test_check_grading_result_tool_reports_errors_then_normalized() -> None:
    bad = check_grading_result.invoke({"result": {"subject": "physics"}})
    assert bad["ok"] is False
    assert bad["errors"]

    good = check_grading_result.invoke({"result": MOCK_LLM_RESULT})
    assert good["ok"] is True
    assert good["normalized"]["wrong_count"] == 1
    assert good["needs_human_review"] is False


def test_create_grade_agent_tools_names() -> None:
    assert [tool.name for tool in create_grade_agent_tools()] == [
        "verify_expressions",
        "check_grading_result",
    ]


# ---------------------------------------------------------------- 执行器


@pytest.mark.asyncio
async def test_executor_injects_images_and_returns_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_get(url: str, **_: Any) -> httpx.Response:
        return httpx.Response(200, content=_jpeg_bytes(), request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", fake_get)
    runtime = _FakeGradeRuntime(_answer(MOCK_LLM_RESULT))
    executor = GradeHomeworkExecutor(runtime=runtime, max_pages=4)

    result = await executor.run(_context("https://example.com/hw.jpg"))

    assert result["correct_count"] == 2
    assert result["needs_human_review"] is False
    sent = runtime.input_received["messages"][-1]
    blocks = sent.content
    assert blocks[0]["text"] == "https://example.com/hw.jpg"
    assert blocks[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


@pytest.mark.asyncio
async def test_executor_requires_images_or_history() -> None:
    executor = GradeHomeworkExecutor(runtime=_FakeGradeRuntime(_answer(MOCK_LLM_RESULT)))
    with pytest.raises(AppError) as error:
        await executor.run(
            _context("这道题怎么做", messages=[HumanMessage(content="这道题怎么做")])
        )
    assert error.value.code is ErrorCode.INVALID_PARAMETERS


@pytest.mark.asyncio
async def test_executor_reuses_history_images_for_followup() -> None:
    history = [
        HumanMessage(
            content=[
                {"type": "text", "text": "https://example.com/hw.jpg"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}},
            ]
        ),
        HumanMessage(content="第二题错在哪"),
    ]
    runtime = _FakeGradeRuntime(_answer(MOCK_LLM_RESULT))
    executor = GradeHomeworkExecutor(runtime=runtime)

    result = await executor.run(_context("第二题错在哪", messages=history))

    assert result["total_questions"] == 3
    # 未重新提交图片：消息原样转发（历史里的图片对 agent 仍可见）。
    assert runtime.input_received["messages"] == history


@pytest.mark.asyncio
async def test_executor_enforces_page_limit() -> None:
    executor = GradeHomeworkExecutor(
        runtime=_FakeGradeRuntime(_answer(MOCK_LLM_RESULT)), max_pages=2
    )
    urls = [f"https://example.com/{index}.jpg" for index in range(3)]
    with pytest.raises(AppError) as error:
        await executor.run(_context("x", parameters={"urls": urls}))
    assert error.value.code is ErrorCode.INVALID_PARAMETERS


@pytest.mark.asyncio
async def test_executor_rejects_unparsable_answer() -> None:
    history = [
        HumanMessage(
            content=[
                {"type": "text", "text": "批改这张图"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}},
            ]
        )
    ]
    executor = GradeHomeworkExecutor(runtime=_FakeGradeRuntime("全部做对了！"))
    with pytest.raises(AppError) as error:
        await executor.run(_context("看下这份作业", messages=history))
    assert error.value.code is ErrorCode.EXECUTION_FAILED


# ---------------------------------------------------------------- 装配


def test_create_grade_homework_agent_forwards_configuration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    build_calls: list[dict[str, Any]] = []

    def fake_build_deep_agent(**kwargs: Any) -> object:
        build_calls.append(kwargs)
        return object()

    monkeypatch.setattr(grade_mod, "build_deep_agent", fake_build_deep_agent)
    model = object()
    runtime = grade_mod.create_grade_homework_agent(
        model=model,  # type: ignore[arg-type]
        checkpointer=object(),  # type: ignore[arg-type]
        skill_root=tmp_path,
    )
    assert runtime is not None
    call = build_calls[0]
    assert call["model"] is model
    assert call["skill_root"] == tmp_path
    assert call["skill_sources"] == [("/skills/grade-homework/", "Grade Homework")]
    assert "作业批改" in call["system_prompt"]
    assert [tool.name for tool in call["tools"]] == [
        "verify_expressions",
        "check_grading_result",
    ]
    assert [item["name"] for item in call["subagents"]] == ["error_analyst", "feedback_writer"]
    # 主图装配 task 限流，防止主代理反复派生子代理空转。
    from langchain.agents.middleware.tool_call_limit import ToolCallLimitMiddleware

    limit = call["middleware"][0]
    assert isinstance(limit, ToolCallLimitMiddleware)
    assert limit.tool_name == "task"


def test_subagents_are_text_only_with_structured_contracts() -> None:
    subagents = build_grade_homework_subagents()
    analyst, writer = subagents
    assert [item["name"] for item in subagents] == ["error_analyst", "feedback_writer"]
    # 子代理不接触图片：不继承主代理工具，只做文本推理。
    assert analyst["tools"] == []
    assert writer["tools"] == []
    assert analyst["skills"] == ["/skills/grade-homework/"]
    assert analyst["response_format"] is DiagnosisResult
    assert writer["response_format"] is FeedbackReport
    assert "model" not in analyst  # 继承主代理模型


def test_skill_layout_loadable() -> None:
    """技能源容器下每个一级子目录都必须是合法技能（name 与目录同名）。"""
    skills_root = Path(agent_app.__file__).resolve().parent / "skills"
    containers = [path for path, _ in grade_mod.SKILL_SOURCES]

    assert containers == ["/skills/grade-homework/"]
    container = skills_root / containers[0].removeprefix("/skills/").strip("/")
    skill_dirs = [d for d in container.iterdir() if d.is_dir() and (d / "SKILL.md").is_file()]

    assert {d.name for d in skill_dirs} == {
        "grading-orchestration",
        "math-grading",
        "chinese-grading",
        "english-grading",
        "error-diagnosis",
        "feedback-writing",
    }
    descriptions: list[str] = []
    for skill_dir in skill_dirs:
        text = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
        frontmatter = yaml.safe_load(text.split("---\n", 2)[1])
        assert frontmatter["name"] == skill_dir.name
        assert frontmatter["description"]
        assert len(frontmatter["description"]) <= 1024
        descriptions.append(frontmatter["description"])
    # description 是唯一触发键，必须互不重复。
    assert len(set(descriptions)) == len(descriptions)

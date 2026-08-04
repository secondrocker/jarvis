"""workflow 包级目录与统一适配器测试。"""

from types import SimpleNamespace

import pytest
from fakes import FakeObjectStorage

from agent_app import workflows as workflows_mod
from agent_app.errors import AppError, ErrorCode
from agent_app.orchestration.executors import ExecutionContext
from agent_app.schemas.tasks import SelectedMode
from agent_app.workflows import create_workflows


def _context(*, max_words: int = 100) -> ExecutionContext:
    return ExecutionContext(
        message=" Alpha.\n Beta. ",
        messages=[],
        parameters={"language": "zh-CN", "max_words": max_words, "unused": "ignored"},
        config={"configurable": {"thread_id": "thread-1"}},
        emit=lambda _: None,
    )


@pytest.mark.asyncio
async def test_create_workflows_returns_routable_executors(
    fake_summary_model,
    monkeypatch,
) -> None:
    selected_models = []

    def fake_create_chat_model(settings, *, model_name=None):
        selected_models.append(model_name)
        return fake_summary_model

    monkeypatch.setattr(workflows_mod, "create_chat_model", fake_create_chat_model)

    workflows = create_workflows(
        settings=SimpleNamespace(openai=SimpleNamespace(summary_model="summary-specialized-model")),
        storage=FakeObjectStorage(),
    )

    assert set(workflows) == {"summary", "pdf_to_image"}

    summary = workflows["summary"]
    assert summary.mode is SelectedMode.WORKFLOW
    assert summary.description
    assert summary.is_default is False

    pdf = workflows["pdf_to_image"]
    assert pdf.mode is SelectedMode.WORKFLOW
    assert pdf.description
    assert pdf.is_default is False

    # PDF 执行器不需要模型：模型仅按摘要专用配置创建一次。
    assert selected_models == ["summary-specialized-model"]

    result = await summary.executor.run(_context())

    assert result == {"summary": "测试摘要", "key_points": ["Alpha", "Beta"]}
    prompt = fake_summary_model.runnable.inputs[0]
    assert "Requested language: zh-CN." in prompt.to_messages()[0].content
    assert "Maximum words: 100." in prompt.to_messages()[0].content
    assert prompt.to_messages()[1].content == "Text to summarize:\nAlpha. Beta."


@pytest.mark.asyncio
async def test_summary_executor_maps_invalid_parameters_to_app_error(
    fake_summary_model,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        workflows_mod,
        "create_chat_model",
        lambda settings, *, model_name=None: fake_summary_model,
    )
    definition = create_workflows(
        settings=SimpleNamespace(openai=SimpleNamespace(summary_model=None)),
        storage=FakeObjectStorage(),
    )["summary"]

    with pytest.raises(AppError) as error:
        await definition.executor.run(_context(max_words=10))

    assert error.value.code is ErrorCode.INVALID_PARAMETERS
    assert error.value.public_message == "Invalid summary parameters"

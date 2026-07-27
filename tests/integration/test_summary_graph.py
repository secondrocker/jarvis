import httpx
import pytest
from openai import APITimeoutError

from agent_app.errors import AppError, ErrorCode
from agent_app.workflows.summary.graph import build_summary_graph
from agent_app.workflows.summary.schemas import SummaryResult


@pytest.mark.asyncio
async def test_summary_graph_returns_structured_result(fake_summary_model) -> None:
    graph = build_summary_graph(fake_summary_model)

    output = await graph.ainvoke(
        {"text": "Alpha. Beta.", "language": "zh-CN", "max_words": 100}
    )

    assert output["result"] == {
        "summary": "测试摘要",
        "key_points": ["Alpha", "Beta"],
    }


@pytest.mark.asyncio
async def test_summary_graph_collapses_text_before_sending_prompt(fake_summary_model) -> None:
    graph = build_summary_graph(fake_summary_model)

    await graph.ainvoke({"text": " Alpha.\n\t Beta.  ", "max_words": 100})

    prompt = fake_summary_model.runnable.inputs[0]
    assert fake_summary_model.structured_schema is SummaryResult
    assert "Requested language: none." in prompt.to_messages()[0].content
    assert "Maximum words: 100." in prompt.to_messages()[0].content
    assert prompt.to_messages()[1].content == "Text to summarize:\nAlpha. Beta."


@pytest.mark.asyncio
async def test_summary_graph_rejects_whitespace_only_text(fake_summary_model) -> None:
    graph = build_summary_graph(fake_summary_model)

    with pytest.raises(AppError) as error:
        await graph.ainvoke({"text": " \n\t "})

    assert error.value.code is ErrorCode.INVALID_PARAMETERS
    assert error.value.public_message == "Summary text is empty"


@pytest.mark.asyncio
async def test_summary_graph_maps_unexpected_model_error_to_safe_execution_error(
    fake_summary_model,
) -> None:
    fake_summary_model.runnable.error = RuntimeError("model response included a secret")
    graph = build_summary_graph(fake_summary_model)

    with pytest.raises(AppError) as error:
        await graph.ainvoke({"text": "Alpha. Beta."})

    assert error.value.code is ErrorCode.EXECUTION_FAILED
    assert error.value.public_message == "Summary generation failed"


@pytest.mark.asyncio
async def test_summary_graph_preserves_transient_openai_error_mapping(fake_summary_model) -> None:
    fake_summary_model.runnable.error = APITimeoutError(
        request=httpx.Request("POST", "https://api.openai.com/v1/responses")
    )
    graph = build_summary_graph(fake_summary_model)

    with pytest.raises(AppError) as error:
        await graph.ainvoke({"text": "Alpha. Beta."})

    assert error.value.code is ErrorCode.UPSTREAM_UNAVAILABLE
    assert error.value.public_message == "OpenAI is temporarily unavailable"

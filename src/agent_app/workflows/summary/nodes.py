"""结构化摘要图使用的节点。"""

import re
from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from agent_app.errors import AppError, ErrorCode, normalize_execution_error
from agent_app.workflows.summary.prompts import SUMMARY_PROMPT
from agent_app.workflows.summary.schemas import SummaryResult, SummaryState


def make_preprocess_node() -> Callable[[SummaryState], dict[str, Any]]:
    """返回无副作用的文本规范化节点。

    返回值:
        接收摘要状态并返回规范化文本字段的同步节点。
    """

    def preprocess(state: SummaryState) -> dict[str, Any]:
        """在输入文本进入模型前合并多余空白。

        参数:
            state: 当前摘要工作流状态。

        返回值:
            仅包含规范化 text 字段的状态更新。
        """
        normalized_text = re.sub(r"\s+", " ", state.get("text", "")).strip()
        if not normalized_text:
            raise AppError(ErrorCode.INVALID_PARAMETERS, "Summary text is empty")
        return {"normalized_text": normalized_text}

    return preprocess


def make_summarize_node(
    model: BaseChatModel,
) -> Callable[[SummaryState], Awaitable[dict[str, Any]]]:
    """返回绑定指定模型的异步结构化摘要节点。

    参数:
        model: 支持结构化输出绑定的聊天模型。

    返回值:
        接收摘要状态并返回结构化结果的异步节点。
    """
    structured_model = model.with_structured_output(SummaryResult)

    async def summarize(state: SummaryState) -> dict[str, Any]:
        """请求绑定模型的结构化响应，并转换为可安全写入状态的字典。

        参数:
            state: 已完成文本预处理的摘要状态。

        返回值:
            包含结构化摘要结果或标准化错误的状态更新。
        """
        try:
            prompt = SUMMARY_PROMPT.invoke(
                {
                    "text": state["normalized_text"],
                    "language": state.get("language") or "none",
                    "max_words": state.get("max_words", 200),
                }
            )
            result = await structured_model.ainvoke(prompt)
        except Exception as error:
            raise normalize_execution_error(
                error,
                fallback_code=ErrorCode.EXECUTION_FAILED,
                fallback_message="Summary generation failed",
            ) from error
        return {"result": result.model_dump()}

    return summarize

"""把批改 Deep Agent 适配为统一执行器。

职责边界：图片下载/压缩/注入是感知层（确定性），判分与错因由 agent 完成，
返回前再由 :func:`.postprocess.finalize_answer` 做契约校验、统计重算、
复算纠错与人审裁定——agent 输出的数字不直接决定对外结果。
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any

from pydantic import ValidationError

from agent_app.deep_agents.adapter import DeepAgentAdapter
from agent_app.deep_agents.grade_homework.images import (
    attach_images,
    extract_urls,
    fetch_image_data_urls,
    messages_have_images,
    validate_urls,
)
from agent_app.deep_agents.grade_homework.postprocess import finalize_answer
from agent_app.deep_agents.protocols import DeepAgentRuntime
from agent_app.errors import AppError, ErrorCode
from agent_app.orchestration.executors import ExecutionContext


class GradeHomeworkExecutor:
    """执行批改 agent，并在返回前收敛到公开批改契约。"""

    def __init__(
        self,
        *,
        runtime: DeepAgentRuntime,
        max_image_mb: int = 8,
        max_image_edge: int = 2048,
        max_pages: int = 4,
        timeout_seconds: float = 30.0,
    ) -> None:
        """保存运行时与图片限制配置。

        参数:
            runtime: 已装配技能、工具与子代理的批改 agent 运行时。
            max_image_mb: 单张图片允许的最大体积（MB）。
            max_image_edge: 兜底压缩的长边上限（像素）。
            max_pages: 单次提交允许的最大图片数（跨页作业）。
            timeout_seconds: 单张图片下载超时秒数。
        """
        self._adapter = DeepAgentAdapter(runtime=runtime)
        self._max_image_mb = max_image_mb
        self._max_image_edge = max_image_edge
        self._max_pages = max_pages
        self._timeout_seconds = timeout_seconds

    @property
    def max_pages(self) -> int:
        """返回单次提交允许的最大图片数。"""
        return self._max_pages

    @property
    def max_image_mb(self) -> int:
        """返回单张图片允许的最大体积（MB）。"""
        return self._max_image_mb

    @property
    def max_image_edge(self) -> int:
        """返回兜底压缩的长边上限（像素）。"""
        return self._max_image_edge

    async def run(self, context: ExecutionContext) -> dict[str, Any]:
        """下载作业图片、执行 agent 并返回校验后的批改结果。

        参数:
            context: 任务输入（图片 URL）、会话历史、运行配置与事件出口。

        返回值:
            符合 :class:`~agent_app.deep_agents.grade_homework.schemas.GradeResult`
            的批改结果字典。

        异常:
            AppError: 缺少图片、图片超限、下载失败或结果契约不合法时抛出。
        """
        enriched = await self._prepare_context(context)
        result = await self._adapter.run(enriched)
        answer = result.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            raise AppError(ErrorCode.EXECUTION_FAILED, "grading agent returned no answer")
        return finalize_answer(answer)

    async def _prepare_context(self, context: ExecutionContext) -> ExecutionContext:
        """把当前提交的图片注入用户消息；无新图片时复用会话历史中的作业。"""
        urls = extract_urls(context.message, context.parameters)
        if not urls:
            # 多轮追问：图片已在会话历史里（agent 可见），无需重新下载。
            if not messages_have_images(context.messages):
                raise AppError(ErrorCode.INVALID_PARAMETERS, "grade_homework: no image url")
            return context
        try:
            validate_urls(urls)
        except ValidationError as error:
            raise AppError(
                ErrorCode.INVALID_PARAMETERS,
                "Invalid grade homework parameters",
            ) from error
        if len(urls) > self._max_pages:
            raise AppError(
                ErrorCode.INVALID_PARAMETERS,
                f"grade_homework: at most {self._max_pages} images per submission",
            )
        data_urls = await asyncio.to_thread(
            fetch_image_data_urls,
            urls,
            max_image_mb=self._max_image_mb,
            max_image_edge=self._max_image_edge,
            timeout_seconds=self._timeout_seconds,
        )
        return replace(context, messages=attach_images(context.messages, data_urls))

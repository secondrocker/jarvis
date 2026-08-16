"""依赖模型的工作流测试所使用的测试替身。"""

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool

from agent_app.workflows.summary.schemas import SummaryResult


class FakeObjectStorage:
    """记录上传调用并返回稳定下载 URL 的内存对象存储替身。"""

    def __init__(self, *, base_url: str = "https://fake-s3.test") -> None:
        """绑定 fake 下载 URL 前缀并初始化上传记录。

        参数:
            base_url: 下载 URL 前缀。
        """
        self.base_url = base_url
        self.uploads: list[tuple[str, str, int]] = []

    def put(self, data: bytes, *, key: str, content_type: str) -> None:
        """记录上传的 key、content_type 与字节数。

        参数:
            data: 上传字节。
            key: 对象 key。
            content_type: 内容类型。
        """
        self.uploads.append((key, content_type, len(data)))

    def download_url(self, key: str) -> str:
        """返回基于 key 的稳定 fake 下载 URL。

        参数:
            key: 对象 key。

        返回值:
            形如 ``{base_url}/{key}`` 的 fake URL。
        """
        return f"{self.base_url}/{key}"

    def upload_url(self, key: str, *, content_type: str) -> str:
        """返回基于 key 的稳定 fake 上传 URL。

        参数:
            key: 对象 key。
            content_type: 内容类型（fake 不校验，仅记录）。

        返回值:
            形如 ``{base_url}/upload/{key}`` 的 fake URL。
        """
        return f"{self.base_url}/{key}"


class ScriptedModel(BaseChatModel):
    """按脚本弹出消息、不访问网络的聊天模型替身。

    供集成测试驱动真实 deepagents 运行时：每次模型调用按顺序弹出
    ``scripts`` 中的一个条目并整体返回（工具调用或纯文本），脚本耗尽后
    抛出断言错误。子代理与主代理可共用同一实例（共享脚本队列）。
    """

    model_name: str = "scripted-model"
    scripts: list[Any] = []

    @classmethod
    def from_scripts(cls, scripts: list[Any]) -> "ScriptedModel":
        """以受脚本列表构造替身。

        参数:
            scripts: 依次返回的消息或工具调用字典列表。
        """
        return cls(scripts=list(scripts))

    @property
    def _llm_type(self) -> str:
        return "scripted-model"

    def _get_ls_params(self, **kwargs: Any) -> dict[str, Any]:
        """伪装为 openai 提供方，使 harness 按 provider 级注册键命中。"""
        return {**super()._get_ls_params(**kwargs), "ls_provider": "openai"}

    @property
    def calls(self) -> list[list[BaseMessage]]:
        """记录每次模型调用的输入消息（含子代理调用）。"""
        return self.__dict__.setdefault("_calls", [])

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """按脚本弹出一条 AIMessage 并整体返回。"""
        self.calls.append(list(messages))
        script = self.scripts.pop(0)
        return ChatResult(generations=[ChatGeneration(message=_to_ai_message(script))])

    def bind_tools(self, tools: list[BaseTool | Any], **kwargs: Any) -> Any:
        """记录绑定的工具名（受限 profile 过滤后模型可见的集合）。"""
        bound = self.__dict__.setdefault("_bound_tools", [])
        bound.append([getattr(tool, "name", str(tool)) for tool in tools])
        return self

    @property
    def bound_tools(self) -> list[list[str]]:
        """历次 bind_tools 收到的工具名快照。"""
        return self.__dict__.setdefault("_bound_tools", [])


def _to_ai_message(script: Any) -> AIMessage:
    """把脚本条目归一为 AIMessage（纯文本或工具调用字典）。"""
    if isinstance(script, AIMessage):
        return script
    if isinstance(script, dict):
        return AIMessage(
            content=script.get("content", ""),
            tool_calls=[script["tool_call"]] if script.get("tool_call") else [],
        )
    return AIMessage(content=str(script))


class FakeStructuredSummaryRunnable:
    """返回固定且符合模型约束的摘要结果的异步可运行对象。"""

    def __init__(self) -> None:
        self.inputs: list[Any] = []
        self.error: Exception | None = None

    async def ainvoke(self, input_value: Any) -> SummaryResult:
        """记录提示词边界，并返回受控的模型响应。

        参数:
            input_value: 工作流渲染后发送给模型的提示词值。

        返回值:
            固定且符合模型约束的摘要结果。
        """
        self.inputs.append(input_value)
        if self.error is not None:
            raise self.error
        return SummaryResult(summary="测试摘要", key_points=["Alpha", "Beta"])


class FakeSummaryModel:
    """用于生成结构化摘要的轻量 LangChain 兼容替身。"""

    def __init__(self) -> None:
        self.structured_schema: type[SummaryResult] | None = None
        self.runnable = FakeStructuredSummaryRunnable()

    def with_structured_output(self, schema: type[SummaryResult]) -> FakeStructuredSummaryRunnable:
        """绑定请求的响应模型，且不访问外部服务。

        参数:
            schema: 调用方要求模型遵循的结构化响应类型。

        返回值:
            可记录输入并返回固定摘要的异步对象。
        """
        self.structured_schema = schema
        return self.runnable


class FakeTaskService:
    """记录调用并返回稳定契约的内存服务替身。"""

    def __init__(self, *, fail_with: Any = None) -> None:
        """初始化可选择成功或失败路径的服务替身。

        参数:
            fail_with: 调用时需要抛出的可选异常。
        """
        from agent_app.schemas.events import EventType, TaskEvent
        from agent_app.schemas.tasks import (
            ExecutionInfo,
            ExecutionMode,
            SelectedMode,
            TaskRequest,
            TaskResponse,
            TaskStatus,
        )

        self._EventType = EventType
        self._TaskEvent = TaskEvent
        self._TaskResponse = TaskResponse
        self._TaskStatus = TaskStatus
        self._ExecutionInfo = ExecutionInfo
        self._SelectedMode = SelectedMode
        self._ExecutionMode = ExecutionMode
        self._TaskRequest = TaskRequest
        self.invoke_calls: list[Any] = []
        self.stream_calls: list[Any] = []
        self._fail_with = fail_with

    def preflight(self, request: Any) -> None:
        """模拟无需额外校验的服务预检。

        参数:
            request: 接口传入的任务请求。
        """
        pass

    async def stream(self, request: Any) -> AsyncIterator[Any]:
        """产生固定顺序的任务事件。

        参数:
            request: 接口传入的任务请求。

        返回值:
            从开始到完成的异步任务事件流。
        """
        self.stream_calls.append(request)
        if self._fail_with is not None:
            raise self._fail_with
        seq = 0
        for etype in (
            self._EventType.TASK_STARTED,
            self._EventType.ROUTE_SELECTED,
            self._EventType.NODE_STARTED,
            self._EventType.NODE_COMPLETED,
            self._EventType.TASK_COMPLETED,
        ):
            seq += 1
            is_agent = request.execution_mode is self._ExecutionMode.DEEP_AGENT
            selected_mode = "deep_agent" if is_agent else "workflow"
            task_type = None if is_agent else request.task_type
            agent_type = (request.agent_type or "solution_planning") if is_agent else None
            data = (
                {"message": "fake"}
                if etype == self._EventType.TASK_STARTED
                else {
                    "selected_mode": selected_mode,
                    "task_type": task_type,
                    "agent_type": agent_type,
                }
            )
            if etype == self._EventType.TASK_COMPLETED:
                data = {
                    "selected_mode": selected_mode,
                    "task_type": task_type,
                    "agent_type": agent_type,
                    "route_reason": "explicit deep agent" if is_agent else "explicit workflow",
                    "result": (
                        {"answer": "fake plan"}
                        if is_agent
                        else {"summary": "fake", "key_points": []}
                    ),
                }
            yield self._TaskEvent(
                type=etype,
                task_id="fake-task",
                thread_id=request.thread_id or "fake-thread",
                sequence=seq,
                timestamp=datetime.now(UTC),
                data=data,
            )

    async def invoke(self, request: Any) -> Any:
        """返回固定的同步任务响应。

        参数:
            request: 接口传入的任务请求。

        返回值:
            符合公开响应契约的成功结果。
        """
        self.invoke_calls.append(request)
        if self._fail_with is not None:
            raise self._fail_with
        is_agent = request.execution_mode is self._ExecutionMode.DEEP_AGENT
        return self._TaskResponse(
            task_id="fake-task",
            thread_id=request.thread_id or "fake-thread",
            status=self._TaskStatus.COMPLETED,
            execution=self._ExecutionInfo(
                selected_mode=(
                    self._SelectedMode.DEEP_AGENT if is_agent else self._SelectedMode.WORKFLOW
                ),
                task_type=None if is_agent else request.task_type,
                agent_type=(request.agent_type or "solution_planning") if is_agent else None,
                route_reason="explicit deep agent" if is_agent else "explicit workflow",
            ),
            result=(
                {"answer": "测试方案"} if is_agent else {"summary": "测试摘要", "key_points": ["A"]}
            ),
        )

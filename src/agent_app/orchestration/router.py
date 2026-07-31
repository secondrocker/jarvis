"""结合显式目标、确定性规则与 LLM 的统一执行器路由。"""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from agent_app.errors import AppError, ErrorCode, normalize_execution_error
from agent_app.orchestration.registry import ExecutorRegistry
from agent_app.orchestration.schemas import LLMRouteDecision, RouteDecision
from agent_app.schemas.tasks import ExecutionMode, SelectedMode, TaskRequest

_SUMMARY_PHRASES = ("总结", "摘要", "概括", "summarize", "summary")

_ROUTING_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You route tasks to one registered executor. Choose only an executor from the "
            "capability list below and return its exact mode and executor_type. Set "
            "is_ambiguous to true when the intent is genuinely unclear. Keep the reason short, "
            "generic, and free of sensitive content.\n\nRegistered capabilities:\n{capabilities}",
        ),
        ("human", "{message}"),
    ]
)


class TaskRouter:
    """按稳定优先级选择已注册 workflow 或 Deep Agent。"""

    def __init__(self, registry: ExecutorRegistry, model: BaseChatModel) -> None:
        """保存统一执行器注册表和路由 LLM。"""
        self._registry = registry
        self._model = model

    async def route(self, request: TaskRequest) -> RouteDecision:
        """按显式模式、命名目标、规则、LLM 和安全回退依次路由。"""
        if request.execution_mode is ExecutionMode.WORKFLOW:
            executor_type = (request.task_type or "").strip().lower()
            self._registry.get(executor_type, mode=SelectedMode.WORKFLOW)
            return RouteDecision.workflow(
                executor_type,
                "Explicit workflow execution requested",
            )

        if request.execution_mode is ExecutionMode.DEEP_AGENT:
            executor_type = (
                request.agent_type or ""
            ).strip().lower() or self._registry.default_agent_type
            self._registry.get(executor_type, mode=SelectedMode.DEEP_AGENT)
            return RouteDecision.deep_agent(
                executor_type,
                "Explicit deep agent execution requested",
            )

        if request.task_type and request.task_type.strip():
            executor_type = request.task_type.strip().lower()
            self._registry.get(executor_type, mode=SelectedMode.WORKFLOW)
            return RouteDecision.workflow(executor_type, "Registered task type matched")

        if request.agent_type and request.agent_type.strip():
            executor_type = request.agent_type.strip().lower()
            self._registry.get(executor_type, mode=SelectedMode.DEEP_AGENT)
            return RouteDecision.deep_agent(executor_type, "Registered agent type matched")

        if any(phrase in request.message.lower() for phrase in _SUMMARY_PHRASES):
            self._registry.get("summary", mode=SelectedMode.WORKFLOW)
            return RouteDecision.workflow("summary", "Summary intent detected")

        return await self._route_with_llm(request.message)

    async def _route_with_llm(self, message: str) -> RouteDecision:
        """调用路由 LLM，并把无效或歧义选择回退到默认 agent。"""
        structured_model = self._model.with_structured_output(LLMRouteDecision)
        capabilities = "\n".join(
            f"- {option.executor_type} [{option.mode.value}]: {option.description}"
            for option in self._registry.routing_options()
        )
        try:
            prompt = _ROUTING_PROMPT.invoke({"message": message, "capabilities": capabilities})
            decision: LLMRouteDecision = await structured_model.ainvoke(prompt)
        except AppError:
            raise
        except Exception as error:
            raise normalize_execution_error(
                error,
                fallback_code=ErrorCode.EXECUTION_FAILED,
                fallback_message="Task routing failed",
            ) from error

        if not decision.is_ambiguous and decision.executor_type:
            executor_type = decision.executor_type.strip().lower()
            if self._registry.contains(executor_type, mode=decision.selected_mode):
                if decision.selected_mode is SelectedMode.WORKFLOW:
                    return RouteDecision.workflow(executor_type, decision.reason)
                return RouteDecision.deep_agent(executor_type, decision.reason)

        return RouteDecision.deep_agent(
            self._registry.default_agent_type,
            "No suitable registered executor for this task",
        )

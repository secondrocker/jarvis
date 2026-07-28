"""结合确定性规则与 LLM 辅助判断的混合任务路由器。"""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from agent_app.errors import AppError, ErrorCode, normalize_execution_error
from agent_app.orchestration.registry import WorkflowRegistry
from agent_app.orchestration.schemas import LLMRouteDecision, RouteDecision
from agent_app.schemas.tasks import ExecutionMode, SelectedMode, TaskRequest

# 能明确表示摘要意图的精确短语。
_SUMMARY_PHRASES = ("总结", "摘要", "概括", "summarize", "summary")

_ROUTING_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a task router for an agent service. Read the user message and "
            "decide whether it should run as a fixed workflow or a flexible deep agent. "
            "Choose 'workflow' only when the task clearly maps to a registered capability "
            "and you can name the task_type. Otherwise choose 'deep_agent'. Set "
            "is_ambiguous to true when the intent is genuinely unclear. Keep the reason "
            "short, generic, and free of any sensitive content.",
        ),
        ("human", "{message}"),
    ]
)


class TaskRouter:
    """按规格定义的优先级将 TaskRequest 解析为 RouteDecision。"""

    def __init__(self, registry: WorkflowRegistry, model: BaseChatModel) -> None:
        """保存工作流注册表和路由 LLM。

        参数:
            registry: 查询固定工作流的注册表。
            model: 在确定性规则无法决策时使用的聊天模型。
        """
        self._registry = registry
        self._model = model

    async def route(self, request: TaskRequest) -> RouteDecision:
        """依次应用显式模式、注册表、规则、LLM 和安全回退优先级。

        参数:
            request: 待选择执行路径的任务请求。

        返回值:
            包含执行模式、任务类型和决策原因的最终路由结果。
        """
        # 1. 显式执行模式拥有最高优先级。
        if request.execution_mode is ExecutionMode.WORKFLOW:
            task_type = request.task_type or ""
            self._registry.get(task_type)  # raises INVALID_TASK_TYPE if missing
            return RouteDecision.workflow(
                task_type=task_type.strip().lower(),
                reason="Explicit workflow execution requested",
            )

        if request.execution_mode is ExecutionMode.DEEP_AGENT:
            return RouteDecision.deep_agent(reason="Explicit deep agent execution requested")

        # 2. 已注册的 task_type 优先于启发式规则和 LLM。
        if request.task_type and request.task_type.strip():
            normalized = request.task_type.strip().lower()
            if self._registry.contains(normalized):
                return RouteDecision.workflow(
                    task_type=normalized,
                    reason="Registered task type matched",
                )

        # 3. 使用确定性的摘要意图规则。
        message_lower = request.message.lower()
        if any(phrase in message_lower for phrase in _SUMMARY_PHRASES):
            return RouteDecision.workflow(
                task_type="summary",
                reason="Summary intent detected",
            )

        # 4. 使用 LLM 辅助路由。
        return await self._route_with_llm(request.message)

    async def _route_with_llm(self, message: str) -> RouteDecision:
        """调用路由 LLM，并应用安全回退规则。

        参数:
            message: 需要分类的用户消息。

        返回值:
            经注册表校验和歧义处理后的路由结果。

        异常:
            AppError: 上游模型出现可识别的瞬时故障时抛出。
        """
        structured_model = self._model.with_structured_output(LLMRouteDecision)
        try:
            prompt = _ROUTING_PROMPT.invoke({"message": message})
            decision: LLMRouteDecision = await structured_model.ainvoke(prompt)
        except AppError:
            raise
        except Exception as error:
            raise normalize_execution_error(
                error,
                fallback_code=ErrorCode.EXECUTION_FAILED,
                fallback_message="Task routing failed",
            ) from error

        # 5. 判断模糊或无效时，安全回退到 Deep Agent。
        if decision.is_ambiguous:
            return RouteDecision.deep_agent(reason="Task intent is ambiguous")

        if decision.selected_mode is SelectedMode.DEEP_AGENT:
            return RouteDecision.deep_agent(reason=decision.reason)

        # 6. LLM 选择工作流时，必须给出已注册的任务类型。
        if decision.task_type and self._registry.contains(decision.task_type):
            return RouteDecision.workflow(
                task_type=decision.task_type.strip().lower(),
                reason=decision.reason,
            )

        return RouteDecision.deep_agent(reason="No suitable registered workflow for this task")

"""Hybrid task router with deterministic and LLM-assisted precedence."""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from agent_app.errors import AppError, ErrorCode, normalize_execution_error
from agent_app.orchestration.registry import WorkflowRegistry
from agent_app.orchestration.schemas import LLMRouteDecision, RouteDecision
from agent_app.schemas.tasks import ExecutionMode, SelectedMode, TaskRequest

# Exact phrases that indicate a clear summary intent.
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
    """Resolve a TaskRequest into a RouteDecision using spec-defined precedence."""

    def __init__(self, registry: WorkflowRegistry, model: BaseChatModel) -> None:
        """Store the workflow registry and routing LLM."""
        self._registry = registry
        self._model = model

    async def route(self, request: TaskRequest) -> RouteDecision:
        """Apply explicit, registry, rule, LLM, and safe-fallback precedence."""
        # 1. Explicit modes override everything.
        if request.execution_mode is ExecutionMode.WORKFLOW:
            task_type = request.task_type or ""
            self._registry.get(task_type)  # raises INVALID_TASK_TYPE if missing
            return RouteDecision.workflow(
                task_type=task_type.strip().lower(),
                reason="Explicit workflow execution requested",
            )

        if request.execution_mode is ExecutionMode.DEEP_AGENT:
            return RouteDecision.deep_agent(reason="Explicit deep agent execution requested")

        # 2. Registered task_type wins over heuristic/LLM.
        if request.task_type and request.task_type.strip():
            normalized = request.task_type.strip().lower()
            if self._registry.contains(normalized):
                return RouteDecision.workflow(
                    task_type=normalized,
                    reason="Registered task type matched",
                )

        # 3. Deterministic summary rule.
        message_lower = request.message.lower()
        if any(phrase in message_lower for phrase in _SUMMARY_PHRASES):
            return RouteDecision.workflow(
                task_type="summary",
                reason="Summary intent detected",
            )

        # 4. LLM-assisted routing.
        return await self._route_with_llm(request.message)

    async def _route_with_llm(self, message: str) -> RouteDecision:
        """Call the routing LLM and apply safe-fallback rules."""
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

        # 5. Ambiguous or invalid → safe Deep Agent fallback.
        if decision.is_ambiguous:
            return RouteDecision.deep_agent(reason="Task intent is ambiguous")

        if decision.selected_mode is SelectedMode.DEEP_AGENT:
            return RouteDecision.deep_agent(reason=decision.reason)

        # 6. LLM selected workflow — must name a registered task type.
        if decision.task_type and self._registry.contains(decision.task_type):
            return RouteDecision.workflow(
                task_type=decision.task_type.strip().lower(),
                reason=decision.reason,
            )

        return RouteDecision.deep_agent(reason="No suitable registered workflow for this task")

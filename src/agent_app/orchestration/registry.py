"""统一管理 workflow 与 Deep Agent 的执行器注册表。"""

from collections.abc import Mapping

from agent_app.errors import AppError, ErrorCode
from agent_app.orchestration.executors import ExecutorDefinition, RoutingOption
from agent_app.schemas.tasks import SelectedMode


def _normalize(executor_type: str) -> str:
    """去除名称首尾空白并转换为小写。"""
    return executor_type.strip().lower()


class ExecutorRegistry:
    """保存可路由执行器并维护唯一的默认 Deep Agent。"""

    def __init__(self, *catalogs: Mapping[str, ExecutorDefinition]) -> None:
        """合并目录、规范化名称并校验默认执行器约束。"""
        self._executors: dict[str, ExecutorDefinition] = {}
        default_agents: list[str] = []

        for catalog in catalogs:
            for raw_name, definition in catalog.items():
                name = _normalize(raw_name)
                if not name:
                    raise ValueError("Executor type must not be blank")
                if name in self._executors:
                    raise ValueError(f"Duplicate executor type after normalization: {name}")
                if definition.is_default and definition.mode is not SelectedMode.DEEP_AGENT:
                    raise ValueError("Only a Deep Agent can be the default executor")
                self._executors[name] = definition
                if definition.is_default:
                    default_agents.append(name)

        if len(default_agents) != 1:
            raise ValueError("Executor registry requires exactly one default Deep Agent")
        self._default_agent_type = default_agents[0]

    @property
    def default_agent_type(self) -> str:
        """返回默认 Deep Agent 的规范化名称。"""
        return self._default_agent_type

    def contains(self, executor_type: str, *, mode: SelectedMode | None = None) -> bool:
        """返回名称是否存在且与可选执行模式匹配。"""
        definition = self._executors.get(_normalize(executor_type))
        return definition is not None and (mode is None or definition.mode is mode)

    def get(
        self,
        executor_type: str,
        *,
        mode: SelectedMode | None = None,
    ) -> ExecutorDefinition:
        """返回匹配的执行器，否则抛出模式对应的公开错误。"""
        definition = self._executors.get(_normalize(executor_type))
        if definition is None or (mode is not None and definition.mode is not mode):
            if mode is SelectedMode.DEEP_AGENT:
                raise AppError(ErrorCode.INVALID_AGENT_TYPE, "Unknown agent type")
            raise AppError(ErrorCode.INVALID_TASK_TYPE, "Unknown task type")
        return definition

    def names(self, mode: SelectedMode | None = None) -> tuple[str, ...]:
        """返回全部或指定模式下排序后的执行器名称。"""
        return tuple(
            sorted(
                name
                for name, definition in self._executors.items()
                if mode is None or definition.mode is mode
            )
        )

    def routing_options(self) -> tuple[RoutingOption, ...]:
        """返回按名称排序的 LLM 路由能力描述。"""
        return tuple(
            RoutingOption(
                executor_type=name,
                mode=definition.mode,
                description=definition.description,
            )
            for name, definition in sorted(self._executors.items())
        )

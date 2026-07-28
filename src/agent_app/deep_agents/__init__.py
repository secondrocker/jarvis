"""受限 Deep Agent 适配层，用于隔离第三方 deepagents 类型。"""

from agent_app.deep_agents.adapter import DeepAgentAdapter
from agent_app.deep_agents.factory import create_restricted_deep_agent
from agent_app.deep_agents.protocols import DeepAgentRuntime

__all__ = [
    "DeepAgentAdapter",
    "DeepAgentRuntime",
    "create_restricted_deep_agent",
]

"""Factory creating a restricted deep agent with only approved capabilities."""

from pathlib import Path

from deepagents import HarnessProfile, create_deep_agent, register_harness_profile
from deepagents.backends import StateBackend
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver

from agent_app.deep_agents.protocols import DeepAgentRuntime

# Built-in tools that are explicitly excluded for the restricted demo agent.
# `execute` (shell) and `task` (subagent dispatch) are removed per spec.
_EXCLUDED_TOOLS = ("execute", "task")


def create_restricted_deep_agent(
    *,
    model: BaseChatModel,
    checkpointer: BaseCheckpointSaver,
    skill_root: Path,
) -> DeepAgentRuntime:
    """Create a runtime with only approved skills and in-memory capabilities."""
    profile = HarnessProfile(excluded_tools=frozenset(_EXCLUDED_TOOLS))
    register_harness_profile("restricted-demo", profile)

    skill_path = str(skill_root / "solution_planning")

    return create_deep_agent(
        model=model,
        tools=None,
        harness_profile="restricted-demo",
        skills=[skill_path],
        backend=StateBackend(),
        checkpointer=checkpointer,
    )

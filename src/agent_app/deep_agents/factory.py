"""Factory creating a restricted deep agent with only approved capabilities."""

from pathlib import Path

from deepagents import (
    GeneralPurposeSubagentProfile,
    HarnessProfile,
    create_deep_agent,
    register_harness_profile,
)
from deepagents.backends import StateBackend
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver

from agent_app.deep_agents.protocols import DeepAgentRuntime

# Shell tool removed per spec. The `task` (subagent dispatch) tool is dropped
# separately by disabling the auto-added general-purpose subagent: the harness
# rejects excluding `SubAgentMiddleware` directly with `ValueError`.
_EXCLUDED_TOOLS = ("execute",)

# Provider key the OpenAI chat model (see infrastructure/llm.py) resolves to.
# `create_deep_agent` looks harness profiles up by the model's provider, so a
# provider-level registration is the match path for a pre-built ChatOpenAI.
_RESTRICTED_PROVIDER_KEY = "openai"


def create_restricted_deep_agent(
    *,
    model: BaseChatModel,
    checkpointer: BaseCheckpointSaver,
    skill_root: Path,
) -> DeepAgentRuntime:
    """Create a runtime with only approved skills and in-memory capabilities."""
    profile = HarnessProfile(
        excluded_tools=frozenset(_EXCLUDED_TOOLS),
        general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
    )
    register_harness_profile(_RESTRICTED_PROVIDER_KEY, profile)

    skill_path = str(skill_root / "solution_planning")

    return create_deep_agent(
        model=model,
        tools=None,
        skills=[skill_path],
        backend=StateBackend(),
        checkpointer=checkpointer,
    )

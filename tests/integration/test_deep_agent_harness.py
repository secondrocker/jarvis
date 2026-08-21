"""harness.build_deep_agent 与真实 deepagents 的集成测试（不联网）。

用 ScriptedModel + tmp_path 技能目录驱动真实运行时，锁定：
技能从磁盘注入 system prompt、受限 profile 生效（无 execute 工具）、
显式 subagents 与禁用通用子代理共存（task 工具存在）、技能目录写保护。
"""

from pathlib import Path

import pytest
from fakes import ScriptedModel
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from agent_app.deep_agents.harness import build_deep_agent
from agent_app.deep_agents.solution_planning import create_solution_planning_agent

_SKILL_ROOT = Path(__file__).resolve().parents[2] / "src" / "agent_app" / "skills"


def _build_skill_dir(root: Path, name: str, description: str) -> Path:
    """在临时目录创建一个最小可加载的技能目录。"""
    skill_dir = root / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n# {name}\n技能正文。\n",
        encoding="utf-8",
    )
    return root


@pytest.fixture
def captured_prompt(monkeypatch):
    """捕获发送给模型的最终 system prompt。"""
    captured: dict[str, str] = {}

    def _capture(self, messages, stop=None, run_manager=None, **kwargs):
        from langchain_core.outputs import ChatGeneration, ChatResult

        for message in messages:
            if message.type == "system":
                captured["system"] = _flatten_content(message.content)
                break
        self.calls.append(list(messages))
        script = self.scripts.pop(0)
        return ChatResult(
            generations=[ChatGeneration(message=_to_ai(script))],
        )

    monkeypatch.setattr(ScriptedModel, "_generate", _capture)
    return captured


def _flatten_content(content) -> str:
    """把 system 消息内容（字符串或 content blocks 列表）拍平为字符串。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(block.get("text", "") for block in content if isinstance(block, dict))
    return str(content)


def _to_ai(script) -> object:
    from langchain_core.messages import AIMessage

    if isinstance(script, AIMessage):
        return script
    if isinstance(script, dict):
        return AIMessage(
            content=script.get("content", ""),
            tool_calls=[script["tool_call"]] if script.get("tool_call") else [],
        )
    return AIMessage(content=str(script))


@pytest.mark.asyncio
async def test_skills_loaded_from_disk_into_system_prompt(tmp_path, captured_prompt) -> None:
    """技能经 CompositeBackend 从磁盘加载并注入 system prompt。"""
    _build_skill_dir(tmp_path, "demo-skill", "Demo skill for tests.")
    model = ScriptedModel.from_scripts(["ok"])
    runtime = build_deep_agent(
        model=model,
        checkpointer=MemorySaver(),
        skill_root=tmp_path,
        # source 须为容器目录（技能是其一级子目录），指向技能目录本身注入为空。
        skill_sources=[("/skills/", "Demo Skill")],
    )
    await runtime.ainvoke(
        {"messages": [HumanMessage(content="你好")]},
        config={"configurable": {"thread_id": "t-skills"}},
    )
    assert "## Skills System" in captured_prompt["system"]
    # 断言技能条目本身（而非 source 路径行），防止"注入为空"被路径子串误命中掩盖。
    assert "- **demo-skill**: Demo skill for tests." in captured_prompt["system"]
    assert "Read `/skills/demo-skill/SKILL.md`" in captured_prompt["system"]


@pytest.mark.asyncio
async def test_solution_planning_agent_excludes_shell_and_task_tools(tmp_path) -> None:
    """受限 profile 生效：无 execute 工具、无子代理时无 task 工具。"""
    _build_skill_dir(tmp_path, "solution-planning", "Guides structured planning.")
    model = ScriptedModel.from_scripts(
        [
            {
                "tool_call": {
                    "name": "write_todos",
                    "args": {"todos": []},
                    "id": "tc-1",
                }
            },
            "方案完成。",
        ]
    )
    runtime = create_solution_planning_agent(
        model=model,
        checkpointer=MemorySaver(),
        skill_root=tmp_path,
    )
    result = await runtime.ainvoke(
        {"messages": [HumanMessage(content="制定方案")]},
        config={"configurable": {"thread_id": "t-restricted"}},
    )
    assert "方案完成" in str(result)
    # 模型可见的工具名集合：bind_tools 快照的最后一份。
    tool_names = model.bound_tools[-1]
    assert "execute" not in tool_names
    assert "task" not in tool_names
    assert "write_todos" in tool_names  # task list 能力保留


@pytest.mark.asyncio
async def test_explicit_subagents_keep_task_tool(tmp_path) -> None:
    """GP 子代理禁用时，显式 subagents 仍装配 task 工具。"""
    _build_skill_dir(tmp_path, "demo-skill", "Demo skill.")
    subagent = {
        "name": "researcher",
        "description": "查询数据并返回结果",
        "system_prompt": "你是数据查询专员。",
        "tools": [],
    }
    model = ScriptedModel.from_scripts(
        [
            {
                "tool_call": {
                    "name": "task",
                    "args": {
                        "description": "查询数据",
                        "subagent_type": "researcher",
                    },
                    "id": "tc-task",
                }
            },
            "数据已获取。",
            "最终报告。",
        ]
    )
    runtime = build_deep_agent(
        model=model,
        checkpointer=MemorySaver(),
        skill_root=tmp_path,
        skill_sources=[("/skills/demo-skill/", "Demo Skill")],
        subagents=[subagent],
    )
    result = await runtime.ainvoke(
        {"messages": [HumanMessage(content="查询数据并出报告")]},
        config={"configurable": {"thread_id": "t-task-tool"}},
    )
    # task 调用成功走通（脚本队列被顺序消费）即证明 task 工具存在且可用。
    assert "最终报告" in str(result)


@pytest.mark.asyncio
async def test_skill_directory_is_write_protected(tmp_path) -> None:
    """对 /skills/ 的写入被 permissions 拒绝，磁盘文件不被改动。"""
    _build_skill_dir(tmp_path, "demo-skill", "Demo skill.")
    original = (tmp_path / "demo-skill" / "SKILL.md").read_text(encoding="utf-8")
    model = ScriptedModel.from_scripts(
        [
            {
                "tool_call": {
                    "name": "write_file",
                    "args": {
                        "file_path": "/skills/demo-skill/SKILL.md",
                        "content": "被篡改的内容",
                    },
                    "id": "tc-w",
                }
            },
            "ok",
        ]
    )
    runtime = build_deep_agent(
        model=model,
        checkpointer=MemorySaver(),
        skill_root=tmp_path,
        skill_sources=[("/skills/demo-skill/", "Demo Skill")],
    )
    await runtime.ainvoke(
        {"messages": [HumanMessage(content="写入技能文件")]},
        config={"configurable": {"thread_id": "t-write"}},
    )
    assert (tmp_path / "demo-skill" / "SKILL.md").read_text(encoding="utf-8") == original

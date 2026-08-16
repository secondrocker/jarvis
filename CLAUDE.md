# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

FastAPI 服务，根据任务的明确程度与复杂度，将用户任务路由到**固定 LangGraph 工作流**或**受限 Deep Agents**。Python 3.12，包管理器为 `uv`。完整设计规格见 `main-spec.md`（中文），它是所有架构决策的权威依据。

## 常用命令

所有命令通过 `make` 封装，底层均走 `uv run`：

```bash
make install    # uv sync --all-groups（安装依赖）
make run        # uvicorn agent_app.main:create_app --factory --reload，监听 :8000
make test       # uv run pytest（单元 + 集成 + 契约，全部不联网）
make lint       # ruff check . && ruff format --check .
make smoke      # scripts/smoke.sh（端到端，需要 config.yaml 中配置真实的 openai 节）
```

运行单个测试 / 子集：

```bash
uv run pytest tests/unit/orchestration/test_router.py            # 单个文件
uv run pytest tests/unit/orchestration/test_router.py::test_name # 单个用例
uv run pytest -k "router"                                        # 关键字过滤
```

修复格式 / lint：

```bash
uv run ruff format .
uv run ruff check . --fix
```

配置从项目根 `config.yaml` 读取（从 `config.example.yaml` 复制）：`openai` 节必填 `api_key` 与 `model`。除 `make smoke` 外，所有测试都不联网。

## 架构（必须跨文件理解的全局设计）

### 统一执行器抽象（核心抽象）

工作流与 Deep Agent **不是**两个并行体系，而是同一个 `Executor` 协议（`orchestration/executors.py`）的两种实现：都暴露 `async run(context: ExecutionContext) -> dict`。每个执行器以 `ExecutorDefinition(mode, description, executor, is_default)` 形式注册。

因此顶层编排图（`orchestration/graph.py`）只有一个 `execute` 节点，按路由结果从注册表取出任意执行器执行。**新增执行器不需要修改图结构**——这是最关键的架构不变量。

### 顶层编排图

线性流程 `START → normalize_input → select_route → execute → END`：

- `normalize_input`：把当前用户消息追加到检查点历史（多轮会话基础）。
- `select_route`：调用 `TaskRouter` 选择 `selected_mode` + `selected_executor_type`。
- `execute`：从 `ExecutorRegistry.get(executor_type, mode=...)` 取执行器，构造 `ExecutionContext` 调用 `run()`。Deep Agent 返回的 `answer` 会被追加为 `AIMessage` 写回历史。

### 路由优先级（`orchestration/router.py`，从高到低）

1. 显式 `execution_mode`（`workflow` 必须带 `task_type`；`deep_agent` 可省略 `agent_type` 走默认）
2. AUTO 请求中含已注册的 `task_type` / `agent_type`
3. 确定性关键词规则（"总结/摘要/summarize" → summary workflow）
4. LLM 按注册表能力描述结构化分类
5. 歧义或无效选择 → 安全回退到默认 Deep Agent

`task_type` 与 `agent_type` **互斥**；显式填写但未注册的目标在 `TaskService.preflight` 阶段直接返回 422（拼写错误不会被静默路由）。

### 单一事件源（`services/task_service.py` + `schemas/events.py`）

`invoke()` 消费 `stream()`——同步响应是流式事件的聚合。`TaskService` 是唯一的事件序列生成点：图节点通过 LangGraph `get_stream_writer()`（`custom` 流模式）发出 `PendingEvent`，`TaskService` 用 `EventSequencer` 给每个事件附上 `task_id`、`thread_id`、单调递增 `sequence` 和时间戳，产出固定序列 `task.started → route.selected → node.started → node.completed → task.completed`，失败终态为 `task.failed`。新增/修改事件类型需同时更新 `schemas/events.py` 的 `EventType`。

### 新增执行器

在所属目录的创建函数里登记（**而非**改图）：

- 工作流：`workflows/__init__.py` 的 `create_workflows()`，提供 `WorkflowExecutor` + `ExecutorDefinition`。
- Deep Agent：新建 `deep_agents/<name>/` 模块（prompt/subagents/技能声明，调 `harness.build_deep_agent`），再在 `deep_agents/catalog.py` 的 `create_agents()` 登记。

注册后自动出现在路由候选中。Deep Agent 目录必须有且仅有一个 `is_default=True` 项（注册表构造时强制校验）。

当前 Deep Agent：`solution_planning`（默认，结构化方案规划）、`info_price`（信息价查询/比价/趋势分析，researcher+analyst 职能子代理，图表经 `tools/chart_tools.py` 上传对象存储）。

### Deep Agent 装配（`deep_agents/harness.py`，非显而易见）

每个 agent 是独立模块（`deep_agents/<name>/`，含 prompt、子代理、技能声明），`harness.py` 只沉淀公共装配：受限 profile（排除 `execute` Shell 工具 + `GeneralPurposeSubagentProfile(enabled=False)` 禁用通用子代理——不能直接排除 `SubAgentMiddleware`，否则 harness 抛 `ValueError`；需要子代理的 agent 显式传 `subagents=`，`task` 工具照常装配）+ `CompositeBackend` + `/skills/**` 写保护。

**关键陷阱**：
- `create_deep_agent` 按**模型提供方**查找 harness profile，对预构建的 `ChatOpenAI` 必须用 `register_harness_profile("openai", profile)` 在提供方级别注册（harness.py 以模块级标志幂等注册，注册表是进程级全局 dict 且 additive merge）。
- `SkillsMiddleware` 只走 backend API（无磁盘回退），`StateBackend` 读不到磁盘技能——必须 `CompositeBackend(default=StateBackend(), routes={"/skills/": FilesystemBackend(root_dir=skill_root, virtual_mode=True)})`，并用 `FilesystemPermission(operations=["write"], paths=["/skills/**"], mode="deny")` 挡住写穿。
- adapter 的多流模式必须传 **list**（`stream_mode=["messages", "updates"]`）：langgraph `_output()` 只在 `isinstance(stream_mode, list)` 时产出 `(mode, payload)`，tuple 会退化为裸 payload 导致解包崩溃。嵌套子图（子代理）的中间输出靠 checkpoint_ns 含 `|` 过滤。
- `event_mapper` 同时处理 `AIMessage` 与 `AIMessageChunk`（messages 流模式下非流式模型产出完整 AIMessage）。
- task list（`TodoListMiddleware`/`write_todos`）与自动压缩（`SummarizationMiddleware`）默认装配，勿通过 profile 排除。

### 配置与模型分配（`config.py`）

`openai.model` 是路由器及所有未单独配置执行器的默认模型。每个执行器在所属创建函数里选自己的可选字段：当前有 `summary_model`、`solution_planning_model`、`info_price_model`，未设置时回退到 `openai.model`。新增执行器沿用此模式。

### 会话与检查点

仅进程内存储：`MemorySaver`（编排图）+ `StateBackend`（Deep Agent）均为内存实例，**重启即丢失**。`thread_id` 是会话连续性的键。

## 错误模型（`errors.py`）

`AppError(code, public_message)` 是唯一跨边界的异常，`public_message` 必须对调用方安全。`ErrorCode` StrEnum 经 `error_http_status()` 映射到 HTTP 状态（校验类→422，`UPSTREAM_UNAVAILABLE`→503，其余→500）。`normalize_execution_error()` 把 OpenAI 瞬时故障（超时/连接/限流）统一映射为 `UPSTREAM_UNAVAILABLE`，并保留已封装的 `AppError`。图执行层捕获异常后写入 `state["error"]`，由 `TaskService` 转为 `task.failed` 事件——**不要让异常越过图边界**。

## 测试约定

- `pytest` 配置 `asyncio_mode = "auto"`，但测试仍显式标注 `@pytest.mark.asyncio`。
- `tests/fakes.py` 位于测试根目录，通过裸 `from fakes import ...` 导入（`testpaths=["tests"]`，根目录在仓库根）。
- **三层隔离**：`unit`（纯替身）、`integration`（用替身装配完整图，端到端验收见 `tests/integration/test_acceptance.py`）、`contract`（向 `create_app(settings=..., service=FakeTaskService())` 注入假服务，验证 HTTP 契约）。
- 集成/验收测试通过 `unittest.mock.patch` 替换 `create_chat_model` 并用 `_FakeDeepAgentRuntime` 等替身运行真实编排图，**完全不访问网络**。改 LLM 工厂签名时这些替身需同步更新。
- `make smoke` 是唯一会真实调用 OpenAI 的脚本。

## 代码风格约定

- 源码与测试注释、docstring 均为**中文**（见 git 历史中的中文化提交）；新增代码沿用中文注释。
- Ruff：`target-version = "py312"`，`line-length = 100`，规则集 `E F I UP B ASYNC`。
- 技能 markdown（`src/agent_app/skills/**`）通过 hatch `artifacts` 打包进 wheel，删除前确认 `pyproject.toml` 中对应条目。

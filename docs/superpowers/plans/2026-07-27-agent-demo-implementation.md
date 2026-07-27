# 通用任务 Agent Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个 Python 3.12 FastAPI demo，通过顶层 LangGraph 将固定摘要任务路由到 Summary Subgraph，将复杂开放式任务路由到加载了 `solution_planning` skill 的受限 Deep Agent，并提供统一同步与 SSE API。

**Architecture:** 采用模块化单体。FastAPI 仅负责传输，`TaskService` 统一消费顶层 LangGraph 的内部事件流；顶层图负责会话状态和路由；摘要子图与 `DeepAgentAdapter` 是可独立测试的执行分支。外部库类型被限制在 `infrastructure`、`workflows` 和 `deep_agents` 内部，公共 DTO 不暴露 LangChain/Deep Agents 对象。

**Tech Stack:** Python 3.12、uv、FastAPI、Pydantic v2、pydantic-settings、LangChain、langchain-openai、LangGraph、Deep Agents、sse-starlette、structlog、pytest、pytest-asyncio、HTTPX、Ruff。

## Global Constraints

- 使用 Python 3.12 和 `src/agent_app` 布局。
- OpenAI 模型只能通过 `OPENAI_MODEL` 配置；在写 `.env.example` 前使用已安装的 OpenAI 官方开发文档 MCP 核定示例模型值。
- API 只暴露项目 Pydantic DTO，不暴露 LangChain 或 Deep Agents 内部类型。
- 首版 checkpoint 和 Deep Agent 工作区仅在进程内保存；进程重启后状态丢失。
- Deep Agent 只允许白名单 skill、规划/待办和进程内虚拟文件系统；禁止 Shell、宿主机写入、任意 HTTP、浏览器及动态工具注入。
- 默认测试不得读取真实 `OPENAI_API_KEY` 或访问网络；真实 OpenAI 调用只能由显式 smoke 命令触发。
- 单次 LLM 调用超时默认 60 秒；只对限流和瞬时错误最多重试 2 次；任务总超时默认 300 秒。
- 摘要 `max_words` 默认 200，允许 50 至 1000（含边界）。
- SSE 事件 `sequence` 在单次任务内从 1 开始递增；成功以 `task.completed` 结束，建连后失败以 `task.failed` 结束。
- 默认日志不得记录完整用户输入、完整模型输出、系统 prompt、密钥或 skill 工作文件。
- 不实现鉴权、数据库、Redis、队列、取消、断线续传、幂等、RAG、动态插件或多 Agent 协作。

## Target File Map

### Project and configuration

- `pyproject.toml`：项目元数据、运行依赖、开发依赖和 pytest/Ruff 配置。
- `uv.lock`：由 uv 生成的精确依赖锁。
- `.python-version`：固定 Python 3.12。
- `.env.example`：无密钥的运行配置示例。
- `.gitignore`：忽略 `.env`、虚拟环境、缓存和 companion 文件。
- `Makefile`：安装、运行、测试、lint 和显式 smoke 命令。
- `src/agent_app/config.py`：Pydantic Settings 及配置验证。

### Stable application contracts

- `src/agent_app/schemas/tasks.py`：请求、响应、执行信息和错误 DTO。
- `src/agent_app/schemas/events.py`：统一事件类型、事件 payload 和 sequence 生成器。
- `src/agent_app/errors.py`：稳定应用错误码、异常类型、OpenAI 瞬时错误归一化及 HTTP 映射。
- `src/agent_app/logging.py`：structlog 安全字段配置。

### Fixed workflow

- `src/agent_app/workflows/summary/schemas.py`：`SummaryInput` 和 `SummaryResult`。
- `src/agent_app/workflows/summary/prompts.py`：摘要 system prompt。
- `src/agent_app/workflows/summary/nodes.py`：预处理与结构化摘要节点。
- `src/agent_app/workflows/summary/graph.py`：编译 Summary Subgraph。

### Routing and orchestration

- `src/agent_app/orchestration/state.py`：顶层 `AgentState`。
- `src/agent_app/orchestration/registry.py`：显式 workflow 注册表。
- `src/agent_app/orchestration/router.py`：显式覆盖、注册任务、规则、LLM 和回退逻辑。
- `src/agent_app/orchestration/graph.py`：顶层 LangGraph 构建和分支连接。
- `src/agent_app/infrastructure/checkpoint.py`：共享进程内 checkpointer 工厂。
- `src/agent_app/infrastructure/llm.py`：OpenAI chat model 工厂。

### Deep Agent boundary

- `src/agent_app/skills/solution_planning/SKILL.md`：方案制定行为规范。
- `src/agent_app/deep_agents/protocols.py`：供应用层依赖的最小 runtime Protocol。
- `src/agent_app/deep_agents/factory.py`：受限 Deep Agent 构造。
- `src/agent_app/deep_agents/event_mapper.py`：第三方事件到内部事件的映射。
- `src/agent_app/deep_agents/adapter.py`：输入、流式事件和最终结果适配。

### Service and transport

- `src/agent_app/services/task_service.py`：统一流式源和同步聚合。
- `src/agent_app/api/dependencies.py`：应用装配及依赖注入。
- `src/agent_app/api/routes/health.py`：健康检查。
- `src/agent_app/api/routes/tasks.py`：invoke 与 stream 路由。
- `src/agent_app/main.py`：FastAPI app factory 和生命周期。

### Tests and docs

- `tests/unit/`：纯路由、schema、节点、mapper、错误和服务测试。
- `tests/integration/`：摘要子图、顶层图、两条分支和 checkpoint 测试。
- `tests/contract/`：HTTP/JSON/SSE 契约测试。
- `tests/fakes.py`：不访问网络的 fake model、fake graph 和 fake Deep Agent。
- `README.md`：安装、运行、curl 示例、安全边界和限制。
- `scripts/smoke.sh`：只有显式调用才执行真实 OpenAI 请求。

---

### Task 1: Bootstrap the Python package and validated settings

**Files:**
- Create: `pyproject.toml`
- Create: `uv.lock`
- Create: `.python-version`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `Makefile`
- Create: `src/agent_app/__init__.py`
- Create: `src/agent_app/config.py`
- Create: `tests/unit/test_config.py`

**Interfaces:**
- Consumes: no application code.
- Produces: `Settings`, `get_settings() -> Settings`, and dependency lock used by every later task.

- [ ] **Step 1: Resolve the current OpenAI example model from official docs**

After restarting Codex so the installed `openaiDeveloperDocs` MCP is visible, fetch the official latest-model guide and record its recommended general-purpose API model ID. Use that ID only for `OPENAI_MODEL` in `.env.example`; keep `Settings.openai_model` required so production behavior never silently depends on the example.

Evidence to retain in the task notes: official page URL, fetched heading, and selected model ID. If the official MCP remains unavailable, stop this task rather than guessing a model ID.

- [ ] **Step 2: Create package metadata and lock dependencies**

Create `pyproject.toml` with package name `agent-demo`, `requires-python = ">=3.12,<3.13"`, the runtime dependencies listed in Tech Stack, and dev dependencies `pytest`, `pytest-asyncio`, `httpx`, and `ruff`. Configure pytest with `asyncio_mode = "auto"`, `testpaths = ["tests"]`; configure Ruff target `py312`, line length 100, and rules `E`, `F`, `I`, `UP`, `B`, `ASYNC`.

Configure the build backend so `src/agent_app` is installed as the package, and include `skills/**/*.md` plus `skills/**/references/*` as package data. This is required because Task 8 resolves skill files from the installed package rather than the current working directory.

Run:

```bash
uv lock
uv sync --all-groups
```

Expected: exit 0, `uv.lock` is created, and the environment resolves on Python 3.12.

- [ ] **Step 3: Write failing settings tests**

Create `tests/unit/test_config.py`:

```python
import pytest
from pydantic import ValidationError

from agent_app.config import Settings


def test_settings_require_openai_credentials_and_model() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_settings_apply_demo_timeout_defaults() -> None:
    settings = Settings(
        openai_api_key="test-key",
        openai_model="test-model",
        _env_file=None,
    )

    assert settings.llm_timeout_seconds == 60.0
    assert settings.llm_max_retries == 2
    assert settings.task_timeout_seconds == 300.0
    assert settings.log_level == "INFO"
```

- [ ] **Step 4: Run tests to verify the expected failure**

Run:

```bash
uv run pytest tests/unit/test_config.py -v
```

Expected: collection fails with `ModuleNotFoundError: No module named 'agent_app.config'`.

- [ ] **Step 5: Implement validated settings**

Create `src/agent_app/config.py` with this public contract:

```python
from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    openai_api_key: SecretStr
    openai_model: str = Field(min_length=1)
    llm_timeout_seconds: float = Field(default=60.0, gt=0)
    llm_max_retries: int = Field(default=2, ge=0, le=5)
    task_timeout_seconds: float = Field(default=300.0, gt=0)
    log_level: str = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
```

Create `.env.example` with `OPENAI_API_KEY=` and the officially resolved `OPENAI_MODEL` value. Add timeout and log defaults. `.gitignore` must include `.env`, `.venv/`, `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`, `.superpowers/`, and coverage artifacts.

The `Makefile` must expose exact targets:

```make
install:
	uv sync --all-groups
run:
	uv run uvicorn agent_app.main:create_app --factory --reload
test:
	uv run pytest
lint:
	uv run ruff check .
	uv run ruff format --check .
smoke:
	bash scripts/smoke.sh
```

- [ ] **Step 6: Verify settings and package quality**

Run:

```bash
uv run pytest tests/unit/test_config.py -v
uv run ruff check src/agent_app/config.py tests/unit/test_config.py
uv run ruff format --check src/agent_app/config.py tests/unit/test_config.py
```

Expected: all commands exit 0; two tests pass.

- [ ] **Step 7: Commit the foundation**

```bash
git add pyproject.toml uv.lock .python-version .gitignore .env.example Makefile src/agent_app tests/unit/test_config.py
git commit -m "build: bootstrap Python agent service"
```

### Task 2: Define stable task, event, error, and logging contracts

**Files:**
- Create: `src/agent_app/schemas/__init__.py`
- Create: `src/agent_app/schemas/tasks.py`
- Create: `src/agent_app/schemas/events.py`
- Create: `src/agent_app/errors.py`
- Create: `src/agent_app/logging.py`
- Create: `tests/unit/test_task_schemas.py`
- Create: `tests/unit/test_events.py`
- Create: `tests/unit/test_errors.py`

**Interfaces:**
- Consumes: `Settings` from Task 1 for logging setup only.
- Produces: `TaskRequest`, `TaskResponse`, `ExecutionInfo`, `PendingEvent`, `TaskEvent`, `EventSequencer`, `AppError`, `ErrorCode`, and `configure_logging()` used by all later tasks.

- [ ] **Step 1: Write failing schema and event tests**

Create tests that establish the public contract:

```python
from pydantic import ValidationError
import pytest

from agent_app.schemas.tasks import ExecutionMode, TaskRequest


def test_task_request_defaults_to_auto_and_empty_parameters() -> None:
    request = TaskRequest(message="制定一个发布方案")
    assert request.execution_mode is ExecutionMode.AUTO
    assert request.task_type is None
    assert request.thread_id is None
    assert request.parameters == {}


def test_task_request_rejects_blank_message() -> None:
    with pytest.raises(ValidationError):
        TaskRequest(message="   ")
```

```python
from agent_app.schemas.events import EventSequencer, EventType


def test_event_sequencer_starts_at_one_and_increments() -> None:
    sequencer = EventSequencer(task_id="task-1", thread_id="thread-1")
    first = sequencer.next(EventType.TASK_STARTED, {})
    second = sequencer.next(EventType.ROUTE_SELECTED, {"selected_mode": "workflow"})
    assert [first.sequence, second.sequence] == [1, 2]
    assert first.task_id == second.task_id == "task-1"
```

```python
from agent_app.errors import AppError, ErrorCode, error_http_status


def test_error_codes_have_stable_http_mapping() -> None:
    error = AppError(ErrorCode.UPSTREAM_UNAVAILABLE, "OpenAI unavailable")
    assert error_http_status(error.code) == 503
    assert error.public_message == "OpenAI unavailable"
```

- [ ] **Step 2: Run the focused tests and confirm failure**

Run:

```bash
uv run pytest tests/unit/test_task_schemas.py tests/unit/test_events.py tests/unit/test_errors.py -v
```

Expected: imports fail because the contract modules do not exist.

- [ ] **Step 3: Implement task DTOs**

Implement these names in `schemas/tasks.py`:

```python
class ExecutionMode(StrEnum):
    AUTO = "auto"
    WORKFLOW = "workflow"
    DEEP_AGENT = "deep_agent"


class SelectedMode(StrEnum):
    WORKFLOW = "workflow"
    DEEP_AGENT = "deep_agent"


class TaskStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"


class TaskRequest(BaseModel):
    message: str
    execution_mode: ExecutionMode = ExecutionMode.AUTO
    task_type: str | None = None
    thread_id: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class ExecutionInfo(BaseModel):
    selected_mode: SelectedMode
    task_type: str | None
    route_reason: str


class TaskResponse(BaseModel):
    task_id: str
    thread_id: str
    status: Literal["completed"] = "completed"
    execution: ExecutionInfo
    result: dict[str, Any]
```

Use a field validator to strip `message` and reject an empty result. Limit caller-supplied `thread_id` to 128 characters and `task_type` to 64 characters.

Add a model validator: `execution_mode="workflow"` requires a non-empty `task_type`. This catches the shape error at FastAPI/Pydantic validation time; registry membership is checked by `TaskService.preflight()` in Task 7.

- [ ] **Step 4: Implement events, errors, and safe logging**

`schemas/events.py` must define `EventType` with all nine spec values and:

```python
class PendingEvent(BaseModel):
    type: EventType
    data: dict[str, Any] = Field(default_factory=dict)


class TaskEvent(BaseModel):
    type: EventType
    task_id: str
    thread_id: str
    sequence: int = Field(ge=1)
    timestamp: datetime
    data: dict[str, Any] = Field(default_factory=dict)


class EventSequencer:
    def __init__(self, task_id: str, thread_id: str) -> None:
        self._task_id = task_id
        self._thread_id = thread_id
        self._sequence = 0

    def next(self, event_type: EventType, data: dict[str, Any]) -> TaskEvent:
        self._sequence += 1
        return TaskEvent(
            type=event_type,
            task_id=self._task_id,
            thread_id=self._thread_id,
            sequence=self._sequence,
            timestamp=datetime.now(UTC),
            data=data,
        )
```

`errors.py` must define exact codes `VALIDATION_ERROR`, `INVALID_PARAMETERS`, `INVALID_TASK_TYPE`, `UPSTREAM_UNAVAILABLE`, `EXECUTION_FAILED`, `INTERNAL_ERROR`; `AppError` carries only `code`, `public_message`, and optional safe `details`. Map the first three to 422, upstream to 503, and the final two to 500.

Also expose:

```python
def normalize_execution_error(
    error: Exception,
    *,
    fallback_code: ErrorCode,
    fallback_message: str,
) -> AppError:
    """Preserve AppError and map OpenAI transient failures to UPSTREAM_UNAVAILABLE."""
```

Return an existing `AppError` unchanged. Map OpenAI `APITimeoutError`, `APIConnectionError`, and `RateLimitError` to `AppError(ErrorCode.UPSTREAM_UNAVAILABLE, "OpenAI is temporarily unavailable")`. All other exceptions use the supplied safe fallback. Unit tests must construct each upstream exception supported by the locked OpenAI SDK and prove that raw exception messages never enter `public_message` or `details`.

`logging.py` must expose `configure_logging(log_level: str) -> None` and configure JSON output with timestamps and log level. Do not install a processor that serializes arbitrary model/request objects.

- [ ] **Step 5: Run contract unit tests and lint**

Run:

```bash
uv run pytest tests/unit/test_task_schemas.py tests/unit/test_events.py tests/unit/test_errors.py -v
uv run ruff check src/agent_app/schemas src/agent_app/errors.py src/agent_app/logging.py tests/unit
```

Expected: all focused tests pass; Ruff exits 0.

- [ ] **Step 6: Commit stable contracts**

```bash
git add src/agent_app/schemas src/agent_app/errors.py src/agent_app/logging.py tests/unit
git commit -m "feat: define task and event contracts"
```

### Task 3: Build the Summary Subgraph with structured output

**Files:**
- Create: `src/agent_app/workflows/__init__.py`
- Create: `src/agent_app/workflows/summary/__init__.py`
- Create: `src/agent_app/workflows/summary/schemas.py`
- Create: `src/agent_app/workflows/summary/prompts.py`
- Create: `src/agent_app/workflows/summary/nodes.py`
- Create: `src/agent_app/workflows/summary/graph.py`
- Create: `tests/fakes.py`
- Create: `tests/conftest.py`
- Create: `tests/unit/workflows/test_summary_schemas.py`
- Create: `tests/integration/test_summary_graph.py`

**Interfaces:**
- Consumes: a LangChain-compatible chat model supporting `with_structured_output(SummaryResult)`.
- Produces: `SummaryInput`, `SummaryResult`, `SummaryState`, and `build_summary_graph(model) -> CompiledStateGraph`.

- [ ] **Step 1: Write failing validation and graph tests**

Create tests:

```python
import pytest
from pydantic import ValidationError

from agent_app.workflows.summary.schemas import SummaryInput


def test_summary_input_defaults_and_bounds() -> None:
    value = SummaryInput(text="A useful source text", language=None)
    assert value.max_words == 200

    with pytest.raises(ValidationError):
        SummaryInput(text="A useful source text", max_words=49)

    with pytest.raises(ValidationError):
        SummaryInput(text="A useful source text", max_words=1001)
```

```python
import pytest

from agent_app.workflows.summary.graph import build_summary_graph


@pytest.mark.asyncio
async def test_summary_graph_returns_structured_result(fake_summary_model) -> None:
    graph = build_summary_graph(fake_summary_model)
    output = await graph.ainvoke(
        {"text": "Alpha. Beta.", "language": "zh-CN", "max_words": 100}
    )
    assert output["result"] == {
        "summary": "测试摘要",
        "key_points": ["Alpha", "Beta"],
    }
```

Implement `FakeSummaryModel` in `tests/fakes.py` as a small object whose `with_structured_output()` returns an async runnable recording its input and returning `SummaryResult(summary="测试摘要", key_points=["Alpha", "Beta"])`. Export the `fake_summary_model` pytest fixture from root `tests/conftest.py`; root fixtures must not import the FastAPI app or require environment credentials.

- [ ] **Step 2: Run focused tests and confirm failure**

Run:

```bash
uv run pytest tests/unit/workflows/test_summary_schemas.py tests/integration/test_summary_graph.py -v
```

Expected: imports fail for the missing summary modules.

- [ ] **Step 3: Implement summary schemas and prompt**

Define:

```python
class SummaryInput(BaseModel):
    text: str = Field(min_length=1)
    language: str | None = Field(default=None, max_length=32)
    max_words: int = Field(default=200, ge=50, le=1000)


class SummaryResult(BaseModel):
    summary: str = Field(min_length=1)
    key_points: list[str] = Field(min_length=1, max_length=10)


class SummaryState(TypedDict, total=False):
    text: str
    language: str | None
    max_words: int
    normalized_text: str
    result: dict[str, Any]
```

The system prompt must instruct the model to summarize only supplied text, preserve facts, use the requested language when present, respect `max_words`, and return `summary` plus concise `key_points`. It must not claim external research.

- [ ] **Step 4: Implement nodes and graph minimally**

Expose:

```python
def make_preprocess_node() -> Callable[[SummaryState], dict[str, Any]]:
    """Return the pure text-normalization node."""


def make_summarize_node(
    model: BaseChatModel,
) -> Callable[[SummaryState], Awaitable[dict[str, Any]]]:
    """Return the async structured-summary node bound to model."""


def build_summary_graph(model: BaseChatModel) -> CompiledStateGraph:
    """Compile START -> preprocess -> summarize -> END."""
```

`preprocess` strips and collapses repeated whitespace; it raises `AppError(ErrorCode.INVALID_PARAMETERS, "Summary text is empty")` if the normalized text is empty. `summarize` calls `model.with_structured_output(SummaryResult)` and stores `SummaryResult.model_dump()` under `result`. Catch model exceptions and raise `normalize_execution_error(error, fallback_code=ErrorCode.EXECUTION_FAILED, fallback_message="Summary generation failed")`. Build `START → preprocess → summarize → END`.

- [ ] **Step 5: Verify the subgraph**

Run:

```bash
uv run pytest tests/unit/workflows/test_summary_schemas.py tests/integration/test_summary_graph.py -v
uv run ruff check src/agent_app/workflows tests/fakes.py tests/unit/workflows tests/integration/test_summary_graph.py
```

Expected: all focused tests pass and no network call occurs.

- [ ] **Step 6: Commit the fixed workflow**

```bash
git add src/agent_app/workflows tests/fakes.py tests/unit/workflows tests/integration/test_summary_graph.py
git commit -m "feat: add structured summary workflow"
```

### Task 4: Implement deterministic and LLM-assisted routing

**Files:**
- Create: `src/agent_app/orchestration/__init__.py`
- Create: `src/agent_app/orchestration/registry.py`
- Create: `src/agent_app/orchestration/router.py`
- Create: `tests/unit/orchestration/test_registry.py`
- Create: `tests/unit/orchestration/test_router.py`

**Interfaces:**
- Consumes: `ExecutionMode`, `SelectedMode`, registered workflow names, and an optional structured-output router model.
- Produces: `WorkflowRegistry`, `RouteDecision`, `LLMRouteDecision`, and `TaskRouter.route(request) -> RouteDecision`.

- [ ] **Step 1: Write failing routing tests for every precedence rule**

Create parameterized tests plus these essential cases:

```python
import pytest

from agent_app.errors import AppError, ErrorCode
from agent_app.orchestration.router import TaskRouter
from agent_app.schemas.tasks import ExecutionMode, SelectedMode, TaskRequest


@pytest.mark.asyncio
async def test_explicit_workflow_requires_registered_task_type(registry, router_model) -> None:
    router = TaskRouter(registry=registry, model=router_model)
    with pytest.raises(AppError) as caught:
        await router.route(TaskRequest(message="do it", execution_mode=ExecutionMode.WORKFLOW))
    assert caught.value.code is ErrorCode.INVALID_TASK_TYPE


@pytest.mark.asyncio
async def test_registered_task_type_wins_without_llm(registry, recording_router_model) -> None:
    router = TaskRouter(registry=registry, model=recording_router_model)
    result = await router.route(TaskRequest(message="text", task_type="summary"))
    assert result.selected_mode is SelectedMode.WORKFLOW
    assert result.task_type == "summary"
    assert recording_router_model.calls == []


@pytest.mark.asyncio
async def test_ambiguous_llm_decision_falls_back_to_deep_agent(registry, ambiguous_router_model) -> None:
    router = TaskRouter(registry=registry, model=ambiguous_router_model)
    result = await router.route(TaskRequest(message="帮我想想下一步"))
    assert result.selected_mode is SelectedMode.DEEP_AGENT
    assert result.task_type is None
```

Also test explicit `deep_agent`, clear Chinese and English summary phrases, invalid LLM-selected workflow names, and whitespace/case normalization for task types.

- [ ] **Step 2: Run tests and confirm imports fail**

Run:

```bash
uv run pytest tests/unit/orchestration/test_registry.py tests/unit/orchestration/test_router.py -v
```

Expected: missing orchestration modules.

- [ ] **Step 3: Implement the explicit registry**

Define:

```python
class Workflow(Protocol):
    async def ainvoke(
        self,
        input: dict[str, Any],
        config: RunnableConfig | None = None,
    ) -> dict[str, Any]:
        """Execute a registered workflow."""


class WorkflowRegistry:
    def __init__(self, workflows: Mapping[str, Workflow]) -> None:
        """Normalize and validate explicit workflow registrations."""

    def contains(self, task_type: str) -> bool:
        """Return whether a normalized task type is registered."""

    def get(self, task_type: str) -> Workflow:
        """Return a workflow or raise INVALID_TASK_TYPE."""

    def names(self) -> tuple[str, ...]:
        """Return sorted normalized task type names."""
```

Normalize registry keys with `strip().lower()`. Duplicate normalized keys raise `ValueError`; missing keys raise `AppError(ErrorCode.INVALID_TASK_TYPE, "Unknown task type")`.

- [ ] **Step 4: Implement routing DTOs and precedence**

Define:

```python
class LLMRouteDecision(BaseModel):
    selected_mode: SelectedMode
    task_type: str | None = None
    is_ambiguous: bool
    reason: str = Field(min_length=1, max_length=500)


class RouteDecision(BaseModel):
    selected_mode: SelectedMode
    task_type: str | None
    reason: str


class TaskRouter:
    async def route(self, request: TaskRequest) -> RouteDecision:
        """Apply explicit, registry, rule, LLM, and safe-fallback precedence."""
```

Implement exact precedence from the spec. Deterministic summary rules are limited to explicit phrases such as `总结`, `摘要`, `概括`, `summarize`, `summary`; do not use broad words like `分析`. If structured output is invalid, `is_ambiguous` is true, or the LLM chooses an unregistered workflow, return Deep Agent with a safe reason. Do not propagate raw model output in the reason. Model timeout, connection, and rate-limit exceptions must pass through `normalize_execution_error` and become `UPSTREAM_UNAVAILABLE`; they must not be misclassified as ambiguous tasks.

- [ ] **Step 5: Verify routing behavior and lint**

Run:

```bash
uv run pytest tests/unit/orchestration/test_registry.py tests/unit/orchestration/test_router.py -v
uv run ruff check src/agent_app/orchestration tests/unit/orchestration
```

Expected: all precedence cases pass; fake model call counts prove deterministic paths bypass the LLM.

- [ ] **Step 6: Commit routing**

```bash
git add src/agent_app/orchestration tests/unit/orchestration
git commit -m "feat: add hybrid task routing"
```

### Task 5: Add a restricted skill-backed Deep Agent adapter

**Files:**
- Create: `src/agent_app/skills/solution_planning/SKILL.md`
- Create: `src/agent_app/deep_agents/__init__.py`
- Create: `src/agent_app/deep_agents/protocols.py`
- Create: `src/agent_app/deep_agents/factory.py`
- Create: `src/agent_app/deep_agents/event_mapper.py`
- Create: `src/agent_app/deep_agents/adapter.py`
- Create: `tests/unit/deep_agents/test_event_mapper.py`
- Create: `tests/unit/deep_agents/test_adapter.py`
- Create: `tests/integration/test_deep_agent_factory.py`

**Interfaces:**
- Consumes: a LangChain model, process-local checkpointer, current message/history, `thread_id`, shared `PendingEvent`, and internal event sink.
- Produces: `DeepAgentRuntime` Protocol, `create_restricted_deep_agent`, `map_deep_agent_event`, and `DeepAgentAdapter.run` with stable shape `{"answer": "final response"}`.

- [ ] **Step 1: Write the approved skill content**

Create `SKILL.md` with YAML frontmatter `name: solution-planning` and instructions that require the Agent to:

1. Restate the goal and success criteria.
2. Separate known facts from assumptions.
3. Decompose the task into ordered, verifiable actions.
4. Identify dependencies, risks, mitigations, owners/roles, and acceptance checks.
5. Ask for missing information when it materially changes the plan; otherwise make an explicit bounded assumption.
6. Never claim external research, shell execution, or host-file changes.
7. End with a concise proposed plan and unresolved decisions.

- [ ] **Step 2: Write failing adapter and event mapping tests**

Create a fake runtime yielding library-neutral dictionaries and assert stable mapping:

```python
import pytest

from agent_app.deep_agents.adapter import DeepAgentAdapter
from agent_app.schemas.events import EventType


@pytest.mark.asyncio
async def test_adapter_maps_events_and_returns_answer(fake_deep_agent_runtime) -> None:
    emitted = []
    adapter = DeepAgentAdapter(runtime=fake_deep_agent_runtime)
    result = await adapter.run(
        message="制定发布计划",
        messages=[],
        config={"configurable": {"thread_id": "thread-1"}},
        emit=emitted.append,
    )
    assert result == {"answer": "发布方案"}
    assert [event.type for event in emitted] == [
        EventType.TOOL_STARTED,
        EventType.TOOL_COMPLETED,
        EventType.CONTENT_DELTA,
    ]
```

Test unknown third-party events are ignored, failed tool events map to `tool.completed` with `status="error"`, and missing final answer raises `AppError(ErrorCode.EXECUTION_FAILED, "Deep Agent returned no answer")`.

- [ ] **Step 3: Run tests and confirm failure**

Run:

```bash
uv run pytest tests/unit/deep_agents -v
```

Expected: modules do not exist.

- [ ] **Step 4: Define the library boundary and adapter**

In `protocols.py`, define:

```python
class DeepAgentRuntime(Protocol):
    def astream(
        self,
        input: dict[str, Any],
        config: RunnableConfig,
        *,
        stream_mode: tuple[str, ...],
    ) -> AsyncIterator[Any]:
        """Stream message and update chunks from the restricted runtime."""
```

Import the shared `PendingEvent(type: EventType, data: dict[str, Any])` from `schemas/events.py`. `event_mapper.py` accepts only the specific Deep Agents event shapes observed in the dependency version locked by Task 1; map tool start/end and message token deltas, ignore unknown event kinds, and never include full prompts or virtual file bodies.

`DeepAgentAdapter.run()` consumes `runtime.astream(input, config, stream_mode=("messages", "updates"))`, calls `emit(PendingEvent)` for mapped events, captures the final assistant text, and returns `{"answer": answer}`. Catch dependency exceptions and pass them through `normalize_execution_error(error, fallback_code=ErrorCode.EXECUTION_FAILED, fallback_message="Deep Agent execution failed")`; chain the original exception but never expose it.

- [ ] **Step 5: Verify the installed Deep Agents construction API before implementing the factory**

Run read-only inspection against the locked environment:

```bash
uv run python -c "import inspect; from deepagents import create_deep_agent; print(inspect.signature(create_deep_agent))"
uv run python -c "import deepagents; print(deepagents.__file__)"
```

Then read the installed package's public docs/docstrings for skill sources and in-memory backend. Record the locked Deep Agents version and exact constructor signature in the commit body. Do not enable a capability merely because the constructor offers it.

- [ ] **Step 6: Implement and test the restricted factory**

`create_restricted_deep_agent` must have this project-facing signature regardless of third-party parameter names:

```python
def create_restricted_deep_agent(
    *,
    model: BaseChatModel,
    checkpointer: BaseCheckpointSaver,
    skill_root: Path,
) -> DeepAgentRuntime:
    """Create a runtime with only approved skills and in-memory capabilities."""
```

The factory must pass an empty external-tool list, register only `skill_root / "solution_planning"`, use an in-memory/state backend, and pass the shared process-local checkpointer. It must not construct filesystem, shell, HTTP, browser, MCP, or dynamic-code tools.

Create an integration test that monkeypatches the imported `create_deep_agent`, calls the factory, and asserts the exact captured configuration contains only the approved skill source and no external tools. This test must adapt assertions to the locked public constructor while preserving the project-facing signature above.

- [ ] **Step 7: Verify the Deep Agent boundary**

Run:

```bash
uv run pytest tests/unit/deep_agents tests/integration/test_deep_agent_factory.py -v
uv run ruff check src/agent_app/deep_agents src/agent_app/skills tests/unit/deep_agents tests/integration/test_deep_agent_factory.py
```

Expected: all tests pass without a key or network access; factory capture proves the restricted capability set.

- [ ] **Step 8: Commit the restricted agent**

```bash
git add src/agent_app/deep_agents src/agent_app/skills tests/unit/deep_agents tests/integration/test_deep_agent_factory.py
git commit -m "feat: add restricted skill-backed deep agent"
```

### Task 6: Compose the top-level LangGraph and process-local checkpoint

**Files:**
- Create: `src/agent_app/orchestration/state.py`
- Create: `src/agent_app/orchestration/graph.py`
- Create: `src/agent_app/infrastructure/__init__.py`
- Create: `src/agent_app/infrastructure/checkpoint.py`
- Create: `tests/integration/test_orchestration_graph.py`
- Create: `tests/integration/test_checkpoint_conversation.py`

**Interfaces:**
- Consumes: `TaskRouter`, `WorkflowRegistry`, `DeepAgentAdapter`, `TaskRequest`, and `MemorySaver`.
- Produces: `AgentState`, `PendingEvent`, `build_orchestration_graph` returning `CompiledStateGraph`, and `create_checkpointer() -> MemorySaver`.

- [ ] **Step 1: Write failing branch and state tests**

Create integration tests using fake router, fake workflow, and fake adapter:

```python
import pytest

from agent_app.orchestration.graph import build_orchestration_graph


@pytest.mark.asyncio
async def test_graph_dispatches_workflow_and_preserves_route_metadata(graph_dependencies) -> None:
    graph = build_orchestration_graph(**graph_dependencies.workflow_case())
    output = await graph.ainvoke(
        {
            "task_id": "task-1",
            "thread_id": "thread-1",
            "message": "总结文本",
            "execution_mode": "auto",
            "requested_task_type": None,
            "parameters": {},
        },
        {"configurable": {"thread_id": "thread-1"}},
    )
    assert output["selected_mode"] == "workflow"
    assert output["selected_task_type"] == "summary"
    assert output["result"]["summary"] == "测试摘要"
```

Add the symmetric Deep Agent test and a two-invocation checkpoint test proving the second call with the same `thread_id` sees two user messages while a different `thread_id` sees one.

- [ ] **Step 2: Run the integration tests and confirm failure**

Run:

```bash
uv run pytest tests/integration/test_orchestration_graph.py tests/integration/test_checkpoint_conversation.py -v
```

Expected: missing state/graph/checkpoint implementations.

- [ ] **Step 3: Define state and shared internal event shape**

Define `AgentState` as a `TypedDict` with exact fields from spec. Use `Annotated[list[AnyMessage], add_messages]` for `messages`. Import the `PendingEvent` created in Task 2 so orchestration and Deep Agent code use the same event type.

The normalize node appends `HumanMessage(content=message)` to `messages`; it never expects callers to submit full history.

- [ ] **Step 4: Build the graph with conditional edges**

`build_orchestration_graph` must expose:

```python
def build_orchestration_graph(
    *,
    router: TaskRouter,
    registry: WorkflowRegistry,
    deep_agent: DeepAgentAdapter,
    checkpointer: BaseCheckpointSaver,
) -> CompiledStateGraph:
    """Compile the normalized, routed, checkpointed top-level graph."""
```

Build nodes `normalize_input`, `select_route`, `run_workflow`, and `run_deep_agent`. Use a conditional edge after `select_route`. The workflow node validates summary parameters with `SummaryInput`, invokes the registered graph using the same `RunnableConfig`, and returns its result. The Deep Agent node passes accumulated messages and config into the adapter. Emit `route.selected`, `node.started`, and `node.completed` through LangGraph custom streaming with `get_stream_writer()`.

Compile only the top-level graph with the shared `MemorySaver`; nested runnable calls receive the same `RunnableConfig` and do not allocate their own saver.

- [ ] **Step 5: Implement checkpoint factory and verify conversations**

`create_checkpointer() -> MemorySaver` returns one process-local instance during application lifespan. It must not be cached at module import time; the FastAPI lifespan creates it so tests can create isolated apps.

Run:

```bash
uv run pytest tests/integration/test_orchestration_graph.py tests/integration/test_checkpoint_conversation.py -v
uv run ruff check src/agent_app/orchestration src/agent_app/infrastructure tests/integration
```

Expected: both execution branches pass and thread isolation is proven.

- [ ] **Step 6: Commit orchestration**

```bash
git add src/agent_app/orchestration src/agent_app/infrastructure/checkpoint.py src/agent_app/schemas/events.py tests/integration
git commit -m "feat: compose checkpointed orchestration graph"
```

### Task 7: Implement the single event source in TaskService

**Files:**
- Create: `src/agent_app/services/__init__.py`
- Create: `src/agent_app/services/task_service.py`
- Create: `tests/unit/services/test_task_service.py`

**Interfaces:**
- Consumes: compiled graph custom/value streams, `TaskRequest`, `EventSequencer`, and application errors.
- Produces: `TaskService.preflight(request) -> None`, `TaskService.stream(request) -> AsyncIterator[TaskEvent]`, and `TaskService.invoke(request) -> TaskResponse`.

- [ ] **Step 1: Write failing tests for stream ordering and invoke aggregation**

Use a fake graph that yields custom pending events and a final values state:

```python
import pytest

from agent_app.schemas.events import EventType
from agent_app.schemas.tasks import TaskRequest
from agent_app.services.task_service import TaskService


@pytest.mark.asyncio
async def test_stream_wraps_graph_events_with_monotonic_sequence(fake_success_graph) -> None:
    service = TaskService(graph=fake_success_graph, task_timeout_seconds=1)
    events = [event async for event in service.stream(TaskRequest(message="总结"))]
    assert [event.type for event in events] == [
        EventType.TASK_STARTED,
        EventType.ROUTE_SELECTED,
        EventType.NODE_STARTED,
        EventType.NODE_COMPLETED,
        EventType.TASK_COMPLETED,
    ]
    assert [event.sequence for event in events] == [1, 2, 3, 4, 5]


@pytest.mark.asyncio
async def test_invoke_consumes_the_same_stream(fake_success_graph) -> None:
    service = TaskService(graph=fake_success_graph, task_timeout_seconds=1)
    response = await service.invoke(TaskRequest(message="总结"))
    assert response.status == "completed"
    assert response.result == {"summary": "测试摘要", "key_points": ["A"]}
```

Add tests for caller-supplied/generated `thread_id`, timeout mapping, `AppError` preservation, unknown exception sanitization, and exactly one terminal event. Add preflight cases proving explicit workflow with an unregistered task type raises `INVALID_TASK_TYPE`, while auto and deep-agent requests do not require a registered task type.

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
uv run pytest tests/unit/services/test_task_service.py -v
```

Expected: missing `TaskService`.

- [ ] **Step 3: Implement the single internal event stream**

Define:

```python
class TaskService:
    def __init__(
        self,
        *,
        graph: CompiledStateGraph,
        registered_task_types: Collection[str],
        task_timeout_seconds: float,
    ) -> None:
        """Store the graph, normalized task names, and positive timeout."""

    def preflight(self, request: TaskRequest) -> None:
        """Reject an explicit workflow whose task type is not registered."""

    async def stream(self, request: TaskRequest) -> AsyncIterator[TaskEvent]:
        """Yield exactly one started event and one terminal event."""

    async def invoke(self, request: TaskRequest) -> TaskResponse:
        """Consume stream() and return its validated completed result."""
```

Generate IDs with `uuid4().hex`. `stream()` yields `task.started`, consumes `graph.astream(graph_input, config, stream_mode=("custom", "values"))`, sequences custom `PendingEvent` objects, retains the latest values state, and emits one terminal event. Wrap graph consumption in `asyncio.timeout(task_timeout_seconds)`.

`preflight()` normalizes `request.task_type` with `strip().lower()` and, only for explicit workflow mode, requires membership in the constructor's immutable normalized task-name set. `invoke()` calls `preflight()` before consuming the stream. `stream()` also calls it as defense in depth, although an async generator does not execute until iteration.

Map timeouts to `UPSTREAM_UNAVAILABLE` only when the active stage is an LLM call; otherwise use `EXECUTION_FAILED` with `reason="task timeout"`. To make this deterministic, graph pending node events include a safe `stage` field and service tracks the latest stage. Unknown exceptions become `INTERNAL_ERROR`; only log exception class, task ID, thread ID, and safe stage.

`invoke()` must consume `self.stream(request)` rather than call the graph directly. On `task.completed`, validate and return `TaskResponse`; on `task.failed`, raise `AppError` reconstructed from its stable code and safe description.

- [ ] **Step 4: Verify service behavior**

Run:

```bash
uv run pytest tests/unit/services/test_task_service.py -v
uv run ruff check src/agent_app/services tests/unit/services
```

Expected: all service tests pass, including exactly-one-terminal-event assertions.

- [ ] **Step 5: Commit TaskService**

```bash
git add src/agent_app/services tests/unit/services
git commit -m "feat: add unified task execution service"
```

### Task 8: Assemble the FastAPI application and synchronous endpoints

**Files:**
- Create: `src/agent_app/infrastructure/llm.py`
- Create: `src/agent_app/api/__init__.py`
- Create: `src/agent_app/api/dependencies.py`
- Create: `src/agent_app/api/routes/__init__.py`
- Create: `src/agent_app/api/routes/health.py`
- Create: `src/agent_app/api/routes/tasks.py`
- Create: `src/agent_app/main.py`
- Create: `tests/contract/conftest.py`
- Create: `tests/contract/test_health.py`
- Create: `tests/contract/test_invoke.py`

**Interfaces:**
- Consumes: `Settings`, LLM/checkpointer factories, registry/router/graphs, Deep Agent factory, and `TaskService`.
- Produces: `create_chat_model(settings)`, `build_task_service(settings)`, `get_task_service(request)`, and injectable `create_app(*, settings=None, service=None) -> FastAPI`.

- [ ] **Step 1: Write failing health and invoke contract tests**

Create an app fixture with `create_app(settings=test_settings, service=fake_service)`. The injected service records `invoke` and `stream` calls. Tests:

```python
def test_health_does_not_execute_the_task_service(client, fake_service) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert fake_service.invoke_calls == []
    assert fake_service.stream_calls == []


def test_invoke_returns_stable_contract(client) -> None:
    response = client.post(
        "/api/v1/tasks/invoke",
        json={"message": "总结文本", "execution_mode": "workflow", "task_type": "summary"},
    )
    assert response.status_code == 200
    assert response.json()["execution"] == {
        "selected_mode": "workflow",
        "task_type": "summary",
        "route_reason": "explicit workflow",
    }
```

Add 422 tests for blank message and explicit workflow without task type; add 503/500 tests for `AppError` mappings and assert no stack trace appears.

- [ ] **Step 2: Run contract tests and confirm failure**

Run:

```bash
uv run pytest tests/contract/test_health.py tests/contract/test_invoke.py -v
```

Expected: missing `agent_app.main` or routes.

- [ ] **Step 3: Implement the OpenAI model factory**

Define:

```python
def create_chat_model(settings: Settings) -> ChatOpenAI:
    return ChatOpenAI(
        api_key=settings.openai_api_key,
        model=settings.openai_model,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )
```

Do not set temperature unless required by the selected model. Do not create the model at import time.

- [ ] **Step 4: Implement application assembly and lifespan**

`build_task_service(settings: Settings) -> TaskService` creates one model, one process-local checkpointer, summary graph, explicit registry, router, restricted Deep Agent, top graph, and service. Pass `registry.names()` into `TaskService.registered_task_types`. Resolve `skill_root` with `importlib.resources` or a path relative to the installed package, never the current working directory.

Expose this testable app factory:

```python
def create_app(
    *,
    settings: Settings | None = None,
    service: TaskService | None = None,
) -> FastAPI:
    """Create an app; injected settings/service isolate contract tests from network."""
```

`create_app()` resolves `settings or get_settings()`, configures logging, and installs a lifespan that stores `service or build_task_service(resolved_settings)` on `app.state.task_service`. It registers routers and an `AppError` handler returning:

```json
{"error":{"code":"INVALID_TASK_TYPE","message":"Unknown task type","details":{}}}
```

Pydantic request errors use `VALIDATION_ERROR` with HTTP 422 and safe field locations.

- [ ] **Step 5: Implement health and invoke routes**

Expose:

```python
@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/api/v1/tasks/invoke", response_model=TaskResponse)
async def invoke_task(
    task: TaskRequest,
    service: Annotated[TaskService, Depends(get_task_service)],
) -> TaskResponse:
    return await service.invoke(task)
```

`get_task_service` reads `request.app.state.task_service`; tests inject the service at app construction. Health must not depend on it and must not call the service or OpenAI.

- [ ] **Step 6: Verify HTTP contracts**

Run:

```bash
uv run pytest tests/contract/test_health.py tests/contract/test_invoke.py -v
uv run ruff check src/agent_app/api src/agent_app/infrastructure/llm.py src/agent_app/main.py tests/contract
```

Expected: health and invoke contracts pass without environment credentials or network access because contract tests inject both test settings and a fake service into `create_app`.

- [ ] **Step 7: Commit the application API**

```bash
git add src/agent_app/api src/agent_app/infrastructure/llm.py src/agent_app/main.py tests/contract
git commit -m "feat: expose health and invoke APIs"
```

### Task 9: Add SSE encoding and streaming failure contracts

**Files:**
- Modify: `src/agent_app/api/routes/tasks.py`
- Create: `src/agent_app/api/sse.py`
- Create: `tests/contract/test_stream.py`

**Interfaces:**
- Consumes: `TaskService.stream()` and `TaskEvent`.
- Produces: `encode_sse(event) -> ServerSentEvent` and `POST /api/v1/tasks/stream`.

- [ ] **Step 1: Write failing SSE contract tests**

Use HTTPX streaming against the test app and parse event blocks:

```python
import json


def test_stream_returns_ordered_sse_events(client) -> None:
    with client.stream("POST", "/api/v1/tasks/stream", json={"message": "总结"}) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    blocks = [block for block in body.split("\n\n") if block]
    event_names = [next(line[7:] for line in block.splitlines() if line.startswith("event: ")) for block in blocks]
    payloads = [
        json.loads(next(line[6:] for line in block.splitlines() if line.startswith("data: ")))
        for block in blocks
    ]
    assert event_names[0] == "task.started"
    assert event_names[-1] == "task.completed"
    assert [payload["sequence"] for payload in payloads] == list(range(1, len(payloads) + 1))
```

Add a failure-stream test asserting final `task.failed`, no later `task.completed`, and safe error payload. Add a request-validation test proving blank input returns HTTP 422 before SSE begins.

- [ ] **Step 2: Run the SSE tests and confirm failure**

Run:

```bash
uv run pytest tests/contract/test_stream.py -v
```

Expected: stream route is 404 or encoder module is missing.

- [ ] **Step 3: Implement deterministic SSE encoding**

Define:

```python
def encode_sse(event: TaskEvent) -> ServerSentEvent:
    return ServerSentEvent(
        event=event.type.value,
        id=str(event.sequence),
        data=event.model_dump_json(),
    )
```

The route first calls `service.preflight(task)` synchronously, then returns `EventSourceResponse(encode_sse(event) async for event in service.stream(task))` with headers `Cache-Control: no-cache` plus `X-Accel-Buffering: no`. Do not catch `AppError` inside the generator because the service converts post-start failures to `task.failed`; Pydantic and preflight errors occur before response construction and therefore return HTTP 422.

- [ ] **Step 4: Verify streaming contracts**

Run:

```bash
uv run pytest tests/contract/test_stream.py -v
uv run pytest tests/contract -v
uv run ruff check src/agent_app/api tests/contract
```

Expected: all contract tests pass; success and failure streams each have exactly one terminal event.

- [ ] **Step 5: Commit SSE support**

```bash
git add src/agent_app/api tests/contract/test_stream.py
git commit -m "feat: stream normalized task events over SSE"
```

### Task 10: Complete documentation, smoke script, and full acceptance verification

**Files:**
- Create: `README.md`
- Create: `scripts/smoke.sh`
- Modify: `Makefile`
- Create: `tests/integration/test_acceptance.py`

**Interfaces:**
- Consumes: the complete application and all public endpoints.
- Produces: operator documentation, explicit real-API smoke workflow, and executable coverage of all demo acceptance criteria.

- [ ] **Step 1: Write the failing acceptance test matrix**

Create tests using the fully assembled graph with fake model/runtime:

```python
@pytest.mark.asyncio
async def test_acceptance_matrix(acceptance_service) -> None:
    summary = await acceptance_service.invoke(
        TaskRequest(message="请总结：Alpha Beta", execution_mode="auto")
    )
    assert summary.execution.selected_mode == "workflow"
    assert set(summary.result) == {"summary", "key_points"}

    plan = await acceptance_service.invoke(
        TaskRequest(message="为一个新产品制定分阶段发布方案", execution_mode="auto")
    )
    assert plan.execution.selected_mode == "deep_agent"
    assert set(plan.result) == {"answer"}
```

Add cases for explicit override, same-thread follow-up, different-thread isolation, and successful/failed terminal event sequences. The fixture must remove `OPENAI_API_KEY` from the environment and still pass.

- [ ] **Step 2: Run acceptance tests and close only real gaps**

Run:

```bash
uv run pytest tests/integration/test_acceptance.py -v
```

Expected before any gap fixes: either PASS, or failures that precisely identify an unmet spec behavior. For each failure, add the smallest regression test at the owning layer, make it fail, implement the minimal fix, and rerun both the owning test and this acceptance file. Do not broaden scope.

- [ ] **Step 3: Write README with exact operating guidance**

README must include:

- Architecture summary and request flow.
- Python 3.12 and uv prerequisites.
- `cp .env.example .env`, required variables, `make install`, `make run`.
- curl examples for health, explicit summary, automatic Deep Agent planning, and SSE.
- `execution_mode`, `task_type`, `thread_id`, and `parameters` semantics.
- Test commands and statement that default tests require neither key nor network.
- Process restart loses conversation and virtual workspace state.
- No authentication; local trusted demo only; never expose directly to public internet.
- Deep Agent has no Shell, host writes, arbitrary HTTP/browser, or external research.
- Scope exclusions from the spec.

- [ ] **Step 4: Add an explicit real-API smoke script**

Create `scripts/smoke.sh` with `set -euo pipefail`. It must refuse to run unless both `OPENAI_API_KEY` and `OPENAI_MODEL` are non-empty, start uvicorn on `127.0.0.1:8000`, install a trap to stop only that spawned PID, poll `/health` with a bounded loop, then issue three curl requests: explicit summary invoke, automatic planning invoke, and summary stream. It must never print the API key.

Keep `make smoke` explicit; neither `make test` nor pytest may invoke this script.

- [ ] **Step 5: Run fresh full verification**

Run all of the following in order:

```bash
uv lock --check
uv run ruff format --check .
uv run ruff check .
uv run pytest -q
```

Expected: every command exits 0; pytest reports zero failures, errors, skips caused by missing OpenAI credentials, or network calls.

- [ ] **Step 6: Verify API startup without making a real model call**

With dummy but syntactically valid environment values, start the service and call only health:

```bash
OPENAI_API_KEY=test-key OPENAI_MODEL=test-model uv run uvicorn agent_app.main:create_app --factory --host 127.0.0.1 --port 8000
curl -fsS http://127.0.0.1:8000/health
```

Expected health body: `{"status":"ok"}`. Stop the exact spawned uvicorn process. Do not call `/invoke` with dummy credentials.

- [ ] **Step 7: Review the implementation against every spec section**

Use `main-spec.md` sections 1–17 as a checklist. Record evidence for each acceptance criterion: owning test name, command, and result. Verify `git status --short` contains no secrets, `.env`, test caches, virtual environments, or companion files staged for commit.

- [ ] **Step 8: Commit docs and acceptance suite**

```bash
git add README.md scripts/smoke.sh Makefile tests/integration/test_acceptance.py
git commit -m "docs: add demo operation and acceptance guide"
```

- [ ] **Step 9: Run post-commit verification**

```bash
git show --stat --oneline HEAD
git diff --check HEAD^ HEAD
uv run ruff check .
uv run pytest -q
git status --short
```

Expected: commit contains only documentation, smoke script, Makefile adjustment, and acceptance test; lint and tests exit 0; any remaining untracked environment files are reported but not silently added.

## Execution Notes

- Execute tasks in order because public contracts and fake implementations are dependencies for later tasks.
- Each task receives a fresh review before moving on. A reviewer may reject a task without requiring unrelated later work to be reverted.
- Use test-driven development: observe each focused test fail for the intended missing behavior before adding implementation.
- If the locked LangGraph or Deep Agents API differs from a code snippet, preserve the project-facing interface stated in the task and isolate version-specific adaptation inside the relevant factory/adapter. Update tests and this plan only when the locked public API makes the snippet impossible; do not leak the third-party change into public DTOs.
- Do not run `scripts/smoke.sh` unless the user explicitly authorizes a real OpenAI API call and credentials are already configured.

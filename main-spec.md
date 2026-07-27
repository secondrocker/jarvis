基于 langchain langgraph 和 deepagents 开发一个agent项目,先讨论、确定spec：
1 基于python
2 openai llm
3 fastapi提供接口
4 简单、固定流程任务走 langchain/langgraph 节点流程
5 复杂、模糊任务 走配置 skills的 deepagents 节点
6 先基于demo创建项目框架
7 代码结构形式请给出建议并与我确认
# 通用任务 Agent Demo 设计规格

- 状态：已批准
- 日期：2026-07-27
- 版本：0.1

## 1. 背景与目标

本项目创建一个基于 Python、OpenAI、FastAPI、LangChain、LangGraph 和 Deep Agents 的通用任务 Agent demo。首版目标是验证两类任务可以通过同一个 API 和同一个顶层编排图执行，并形成清晰、可测试、可扩展的项目框架：

- 简单、明确、固定流程的任务由 LangGraph 工作流执行。
- 复杂、模糊、开放式任务由配置了白名单 skills 的 Deep Agent 执行。
- 调用方可以让系统自动选择执行路径，也可以显式指定路径。
- 同步调用和 SSE 流式调用复用同一个执行过程、状态模型和事件协议。
- 首版以本地 demo 为目标，不建设生产级 Agent 平台。

### 1.1 演示用例

首版只提供两个演示用例：

1. 固定流程：文本摘要。输入文本及摘要参数，输出摘要和关键点。
2. 复杂任务：开放式方案制定。Deep Agent 加载 `solution_planning` skill，自主拆解问题并形成方案。

### 1.2 成功标准

项目完成后应能证明：

- 同一个 FastAPI 服务可以执行固定工作流和 Deep Agent。
- 自动路由可以可靠识别明确的摘要任务，并将不明确或开放式任务交给 Deep Agent。
- 显式执行模式可以覆盖自动路由。
- 两条执行路径具有统一的请求、结果、错误和流式事件协议。
- 自动测试无需真实 OpenAI API Key。

## 2. 技术与运行约束

- Python 3.12。
- 使用 `pyproject.toml` 声明项目及依赖，使用 uv 管理依赖锁定和开发命令。
- FastAPI 提供 HTTP API，Pydantic v2 定义外部 DTO 和内部结构化输出。
- LangChain 提供 OpenAI Chat Model 集成及消息抽象。
- LangGraph 提供顶层编排图、固定工作流子图和进程内 checkpoint。
- Deep Agents 提供复杂任务的规划、待办、skill 加载和进程内虚拟文件系统能力。
- OpenAI 模型由 `OPENAI_MODEL` 环境变量配置。不得在 API 契约中暴露或固定模型名称。
- 配置由 Pydantic Settings 加载，仓库只提交 `.env.example`，不提交密钥。

在进入实现前，需要通过已安装的 OpenAI 官方开发文档能力核定 `.env.example` 中的示例模型值。示例值不构成 API 稳定契约。

## 3. 总体架构

采用模块化单体和顶层 LangGraph 编排图：

```text
FastAPI
  ↓
TaskService
  ↓
顶层 LangGraph
  ├── 输入归一化
  ├── 路由选择
  ├── Summary Subgraph
  ├── DeepAgentAdapter Node
  └── 统一结果与事件
```

### 3.1 分层职责

#### FastAPI 传输层

- 定义路由、请求和响应 DTO。
- 将同步调用交给 `TaskService.invoke()`。
- 将统一执行事件编码为 SSE。
- 处理建连前的请求校验错误。
- 不包含 prompt、图节点或 Deep Agents 业务逻辑。

#### TaskService 应用层

- 作为同步和流式任务调用的统一 Facade。
- 接收编译后的顶层 graph，而不是在每次请求中创建 graph。
- 生成缺失的 `task_id` 和 `thread_id`。
- 将统一事件聚合为同步结果，或以异步迭代器形式提供给 SSE。

#### 顶层 LangGraph 编排层

- 归一化输入，建立共享运行状态。
- 根据路由优先级选择执行路径。
- 调用固定工作流子图或 Deep Agent 适配节点。
- 维护会话状态、路由原因、执行元数据和最终结果。
- 不依赖 FastAPI。

#### Summary Subgraph

- 只负责文本摘要流程。
- 对摘要专用参数进行二次校验。
- 完成文本预处理、OpenAI 结构化摘要、结果校验及格式化。
- 以项目内部协议返回摘要、关键点和节点事件。

#### DeepAgentAdapter

- 隔离 Deep Agents 的创建和第三方类型。
- 只加载注册的 skill 和受限工具。
- 将项目运行状态转换成 Deep Agent 输入。
- 将 Deep Agents 的消息、工具事件和结果映射为项目统一协议。

### 3.2 依赖方向

依赖只能沿以下方向流动：

```text
api → services → orchestration → workflows / deep_agents → infrastructure
```

补充约束：

- `schemas` 是稳定的项目协议，不导出 LangChain 或 Deep Agents 内部类型。
- `workflows` 和 `deep_agents` 不导入 FastAPI。
- `infrastructure` 负责外部实现的构造，不反向依赖应用层。
- 首版使用必要的 Facade、Adapter 和轻量 Strategy/Registry，不建设运行时插件系统。

## 4. 代码结构

```text
agent-demo/
├── pyproject.toml
├── README.md
├── .env.example
├── Makefile
├── src/
│   └── agent_app/
│       ├── main.py
│       ├── config.py
│       ├── api/
│       │   ├── dependencies.py
│       │   └── routes/
│       │       ├── health.py
│       │       └── tasks.py
│       ├── schemas/
│       │   ├── tasks.py
│       │   └── events.py
│       ├── services/
│       │   └── task_service.py
│       ├── orchestration/
│       │   ├── state.py
│       │   ├── graph.py
│       │   ├── router.py
│       │   └── registry.py
│       ├── workflows/
│       │   └── summary/
│       │       ├── graph.py
│       │       ├── nodes.py
│       │       ├── prompts.py
│       │       └── schemas.py
│       ├── deep_agents/
│       │   ├── adapter.py
│       │   ├── factory.py
│       │   └── event_mapper.py
│       ├── skills/
│       │   └── solution_planning/
│       │       ├── SKILL.md
│       │       └── references/
│       └── infrastructure/
│           ├── llm.py
│           └── checkpoint.py
└── tests/
    ├── unit/
    ├── integration/
    └── contract/
```

目录规则：

- Prompt 跟随使用它的 workflow 或 agent，不建立全局 `prompts` 目录。
- 新固定能力以 `workflows/<task_type>` 形式增加。
- 固定能力注册表使用显式 Python 映射，不使用动态扫描、entry point 或运行时插件加载。
- 文件保持单一职责；当节点或 mapper 变得过大时按具体能力拆分，而不是增加空的抽象层。
- `tests` 按测试边界组织，而不是机械复制所有源码包。

## 5. 顶层状态与会话

顶层图的共享状态至少包含：

- `task_id`
- `thread_id`
- `message`
- `messages`
- `execution_mode`
- `requested_task_type`
- `parameters`
- `selected_mode`
- `selected_task_type`
- `route_reason`
- `result`
- `error`
- 执行元数据

状态规则：

- `message` 是当前请求新增的用户消息。
- `messages` 是由 checkpoint 管理的会话消息历史，不要求客户端重复提交完整历史。
- API 接受可选 `thread_id`；缺失时由服务生成并在响应中返回。
- 同一 `thread_id` 支持后续多轮对话。
- 顶层图和子图使用进程内 checkpointer。
- Deep Agent 工作区使用进程内虚拟文件系统。
- 服务重启后 checkpoint 和虚拟工作区丢失，此限制必须写入 README。

## 6. 路由设计

`execution_mode` 允许三个值：

- `auto`：按路由优先级自动选择，默认值。
- `workflow`：强制执行固定工作流。
- `deep_agent`：强制执行 Deep Agent。

### 6.1 路由优先级

1. 显式 `workflow` 或 `deep_agent` 覆盖自动判断。
2. `task_type` 命中已注册固定任务时，选择对应 workflow。
3. 对明确摘要意图执行确定性规则匹配。
4. 仍无法判断时，调用轻量 LLM 返回结构化路由决策。
5. LLM 结果无效，或 LLM 明确标记任务存在歧义时，回退到 `deep_agent`。

### 6.2 路由约束

- 路由 LLM 只能选择 `workflow` 或 `deep_agent`。
- 选择 `workflow` 时，必须同时返回一个已注册的 `task_type`。
- 路由结构化输出固定为 `selected_mode`、可空的 `task_type`、`is_ambiguous` 和 `reason`；项目将 `reason` 保存为 `route_reason`。
- `is_ambiguous=true` 时必须选择 `deep_agent`。首版不使用无法校准的数值置信度阈值。
- 路由 LLM 不得创建、修改或动态加载执行图。
- 显式 `workflow` 但未提供有效 `task_type` 时返回 422，不调用 LLM 猜测。
- 显式 `deep_agent` 忽略 `task_type` 的执行选择作用，但可以将其作为普通元数据记录。

## 7. 固定摘要工作流

摘要流程为：

```text
参数校验 → 文本预处理 → OpenAI 结构化摘要 → 结果校验与格式化
```

### 7.1 输入

- 原文来自顶层请求的 `message`。
- `parameters.language`：可选，默认跟随用户语言。
- `parameters.max_words`：可选，表示摘要目标上限；默认 200，允许范围为 50 至 1000（含边界）。

### 7.2 输出

```json
{
  "summary": "摘要文本",
  "key_points": ["关键点一", "关键点二"]
}
```

### 7.3 实现约束

- 使用 Pydantic 结构化输出，而不是解析自由文本 JSON。
- Prompt 位于 `workflows/summary/prompts.py`。
- 节点输出必须满足顶层状态协议。
- 预处理只做 demo 必需的清理和长度检查，不建设文档分块或 map-reduce 摘要系统。

## 8. Deep Agent 设计

复杂任务 demo 处理开放式方案制定。Agent 加载 `solution_planning` skill，并可以使用：

- 任务规划和待办管理。
- 进程内虚拟文件系统。
- 白名单 skill 及其只读参考资料。

首版禁止：

- Shell 命令。
- 宿主机文件写入。
- 任意 HTTP、浏览器或外部搜索。
- 动态安装或加载未注册的 skill。
- 动态注册任意 Python 工具。

因此，Deep Agent 的方案必须基于用户提供的信息和已注册 skill，不得声称执行过外部资料检索。

### 8.1 Skill 约束

`solution_planning/SKILL.md` 应指导 Agent：

- 澄清目标、约束和成功标准。
- 将开放式问题拆分为可执行步骤。
- 显式记录假设、风险和依赖。
- 输出具有优先级、责任边界和验收条件的方案。
- 信息不足时明确说明，而不是虚构事实。

## 9. API 契约

### 9.1 健康检查

```text
GET /health
```

健康检查仅验证进程运行和配置成功加载，不调用 OpenAI。

### 9.2 同步调用

```text
POST /api/v1/tasks/invoke
```

### 9.3 流式调用

```text
POST /api/v1/tasks/stream
Content-Type: text/event-stream
```

### 9.4 统一请求

```json
{
  "message": "请总结以下文本……",
  "execution_mode": "auto",
  "task_type": "summary",
  "thread_id": "optional-client-id",
  "parameters": {
    "language": "zh-CN",
    "max_words": 200
  }
}
```

字段语义：

- `message`：必填、非空的当前轮用户输入。
- `execution_mode`：可选，默认 `auto`。
- `task_type`：可选；首版已注册值只有 `summary`。
- `thread_id`：可选；由客户端复用以继续会话。
- `parameters`：可选对象，由选中的具体 workflow 二次校验；Deep Agent 首版不定义专用参数。

### 9.5 同步成功响应

```json
{
  "task_id": "generated-id",
  "thread_id": "client-or-generated-id",
  "status": "completed",
  "execution": {
    "selected_mode": "workflow",
    "task_type": "summary",
    "route_reason": "registered task type"
  },
  "result": {
    "summary": "摘要文本",
    "key_points": ["关键点"]
  }
}
```

Deep Agent 的稳定结果形状为 `{"answer": "最终文本"}`；其内部待办和虚拟文件不直接成为稳定 API 字段。

## 10. SSE 事件协议

标准事件类型：

- `task.started`
- `route.selected`
- `node.started`
- `content.delta`
- `tool.started`
- `tool.completed`
- `node.completed`
- `task.completed`
- `task.failed`

所有事件至少包含：

- `task_id`
- `thread_id`
- `sequence`
- `timestamp`
- 事件特有 payload

协议约束：

- `sequence` 在单次任务内从 1 开始单调递增。
- 每次流必须以 `task.started` 开始。
- 成功流以 `task.completed` 结束。
- SSE 建连后的执行失败以 `task.failed` 结束，不再发送 `task.completed`。
- Deep Agents 原生事件必须经 `event_mapper` 转换，不透出内部对象。
- `invoke` 和 `stream` 消费同一内部事件源；`invoke` 聚合最终结果，`stream` 编码为 SSE。
- 首版不承诺断线续传或 `Last-Event-ID` 重放。

示例：

```text
event: route.selected
data: {"task_id":"...","thread_id":"...","sequence":2,"selected_mode":"deep_agent","reason":"open-ended planning task"}
```

## 11. 错误处理、超时与重试

### 11.1 建连前错误

| 情况 | HTTP 状态 | 错误码 |
| --- | ---: | --- |
| 请求或字段无效 | 422 | `VALIDATION_ERROR` |
| workflow 参数无效 | 422 | `INVALID_PARAMETERS` |
| task type 未注册 | 422 | `INVALID_TASK_TYPE` |
| OpenAI 超时或限流 | 503 | `UPSTREAM_UNAVAILABLE` |
| Agent 工具或执行失败 | 500 | `EXECUTION_FAILED` |
| 未知内部错误 | 500 | `INTERNAL_ERROR` |

错误响应包含稳定错误码和面向调用方的安全描述，不包含堆栈、密钥、完整 prompt 或第三方原始响应。

### 11.2 SSE 建连后错误

- 上游、工具或内部错误转换为 `task.failed`。
- 工具调用已经开始后失败时，可以先发出带错误状态的 `tool.completed`，再发出 `task.failed`。
- 失败事件包含稳定错误码和安全描述。

### 11.3 超时与重试

- 单次 LLM 调用超时默认 60 秒，可通过配置覆盖。
- 只对 SDK 明确识别的限流和瞬时错误最多重试 2 次，并使用指数退避。
- 验证错误、权限错误和确定性工具错误不重试。
- 整个任务总超时默认 300 秒，可通过配置覆盖。
- 首版不在另一个 HTTP 请求中恢复超时任务。

## 12. 日志与可观测性

- 使用 JSON 结构化日志。
- 日志关联 `task_id`、`thread_id`、`selected_mode` 和 `task_type`。
- 记录节点开始、结束、耗时、错误码和重试次数。
- 默认不记录完整用户输入、完整模型内容或 skill 工作文件。
- 健康检查不主动探测 OpenAI。
- 首版不引入 OpenTelemetry、Prometheus 或 LangSmith，但事件层和依赖注入应允许后续接入。

## 13. 测试策略

### 13.1 Unit Tests

使用 fake 或 stub，不调用真实 OpenAI：

- 路由优先级和显式覆盖。
- LLM 路由结构化结果验证及安全回退。
- Summary 参数校验、预处理和输出格式化。
- 顶层状态转换。
- Deep Agent 事件映射。
- 错误码映射和 SSE sequence 生成。

### 13.2 Integration Tests

- 编译并运行顶层 graph。
- 运行 Summary Subgraph 的成功和失败路径。
- 验证 workflow 和 deep agent 两个分支。
- 验证相同 `thread_id` 的后续一轮能够读取会话状态。
- 验证任务成功和失败的最终状态。

Deep Agent 集成测试使用受控 fake model 或 fake agent adapter，确保测试确定且无需网络。

### 13.3 Contract Tests

使用 FastAPI TestClient 或 AsyncClient：

- 健康检查状态码和响应结构。
- `/invoke` 的请求校验、成功响应和错误结构。
- `/stream` 的 Content-Type、事件顺序和数据结构。
- `sequence` 单调递增。
- 流式失败以 `task.failed` 收尾。

### 13.4 Manual Smoke Test

提供显式的手动命令，在开发者配置 `OPENAI_API_KEY` 和 `OPENAI_MODEL` 后验证：

- 一次真实摘要请求。
- 一次真实开放式方案制定请求。
- 一次 SSE 流式请求。

手动 smoke test 不属于默认自动测试套件。

## 14. Demo 验收标准

实现需满足以下全部条件：

1. `GET /health` 成功且不调用 OpenAI。
2. 显式 `workflow` 与 `task_type=summary` 返回摘要和关键点。
3. `auto` 将明确摘要请求送入 Summary Subgraph。
4. `auto` 将开放式方案制定请求送入配置了 `solution_planning` skill 的 Deep Agent。
5. 显式 `execution_mode` 能覆盖自动判断。
6. 同一 `thread_id` 的后续一轮可以使用已有会话状态。
7. `/stream` 返回有序的 started、route、执行和 completed/failed 事件。
8. 默认自动测试不需要真实 OpenAI API Key。
9. README 说明启动方式、示例调用、安全边界和进程重启后状态丢失的限制。

## 15. 首版范围外

- 鉴权、租户、配额、计费。
- PostgreSQL、Redis、持久化或分布式 checkpoint。
- 后台任务队列、任务取消、断线续传、事件重放、幂等键。
- Shell、宿主机文件写入、任意网络和浏览器访问。
- 外部搜索、数据库、MCP 工具集成。
- 动态 skill 安装、运行时插件市场、多 Agent 协作。
- 文档上传、向量数据库、RAG 和大文档 map-reduce 摘要。
- Kubernetes、生产部署和完整遥测平台。

## 16. 安全与配置

- `OPENAI_API_KEY` 只能从环境变量或部署密钥系统读取。
- `.env` 必须加入 `.gitignore`，`.env.example` 不包含真实密钥。
- API 和日志不得返回密钥、系统 prompt、内部堆栈或 Deep Agent 工作文件。
- Skill 由源码中的显式白名单注册。
- Deep Agent 工具在工厂中显式构造，不接受请求参数注入任意工具。
- 首版未提供鉴权，只能作为受信环境中的本地 demo 运行；不得直接暴露到公共互联网。

## 17. 后续演进方向

只有在实际需求出现后再逐项引入：

- 将进程内 checkpointer 替换为 PostgreSQL。
- 增加 `extract`、`rewrite` 等固定 workflow。
- 接入 LangSmith 或 OpenTelemetry。
- 增加鉴权、配额、后台任务和取消机制。
- 将显式 registry 演进为受控插件机制。
- 按具体业务需要增加搜索、数据库或 MCP 工具，并逐项定义权限。

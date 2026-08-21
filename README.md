# Agent Demo

一个 FastAPI 服务，根据任务的明确程度与复杂度，将用户任务路由到**固定 LangGraph 工作流**或**受限 Deep Agents** 节点。

## 架构

```
客户端 ──▶ FastAPI ──▶ TaskService ──▶ 顶层编排图 (LangGraph)
                                         │
                                         ▼
                                select_route 节点
                                         │
                                         ▼
                                    TaskRouter
                                         │
                                         ▼
                                ExecutorRegistry
                          （workflow + Deep Agent 目录）
                                         │
                                         ▼
                                    execute 节点
                              ┌──────────┴──────────┐
                              ▼                     ▼
                      固定 workflow 子图     Deep Agent Runtime
                  （summary / pdf_to_image）   （受限 harness）
```

workflow 与 Deep Agent 通过统一执行器协议注册到 `ExecutorRegistry`。`TaskRouter`
根据注册能力选择执行器，顶层图的单一 `execute` 节点负责调用，因此新增执行器时
不需要修改图结构。

**路由优先级**（从高到低）：

1. 显式指定 `execution_mode` 及目标；workflow 必须提供 `task_type`，Deep Agent 可省略 `agent_type` 并使用默认项
2. AUTO 请求中包含已注册的 `task_type` 或 `agent_type`
3. 确定性关键词规则（例如“总结”→ 摘要工作流）
4. LLM 根据注册表提供的名称、模式和能力描述辅助分类
5. 意图模糊或模型选择无效 → 安全回退到默认 `solution_planning` agent

`task_type` 与 `agent_type` 互斥。请求显式填写但尚未注册的目标时直接返回 422，
避免拼写错误被静默路由到其他执行器。

Deep Agent Runtime 受到严格限制：通过 `HarnessProfile` 排除了 `execute`（Shell）和 `task`（子代理调度）工具。它使用 `StateBackend`（进程内虚拟文件系统）和 `solution_planning` 技能运行。

## 前置条件

- Python 3.12
- [uv](https://docs.astral.sh/uv/)（包管理器）
- OpenAI API 密钥
- S3 兼容对象存储（PDF 转图片 workflow 上传渲染产物用；仅用摘要/Deep Agent 路径时可不配）

## 安装

```bash
make install                  # 执行 uv sync --all-groups

cp config.example.yaml config.yaml   # 然后填入密钥（openai.api_key、s3 凭证等）
```

应用配置统一从项目根 `config.yaml` 读取（`config.yaml` 仅本地保留，不入库）。
`openai.model` 是路由器及未单独配置执行器时的默认模型；`openai.summary_model`
与 `openai.solution_planning_model` 可分别覆盖摘要 workflow 与方案规划 agent，
未设置时各自回退到 `openai.model`。另有 `s3`（PDF 产物上传的对象存储）、`mcp`
（工具服务开关与挂载路径）、`task`、`log` 四段。新增执行器时应在所属目录定义处
按需选择自己的模型配置。

## 运行

```bash
make run                      # 在 :8000 启动 uvicorn
```

## API

### 健康检查

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

### 同步调用

```bash
# 自动路由摘要（关键词匹配）
curl -s localhost:8000/api/v1/tasks/invoke \
  -H 'Content-Type: application/json' \
  -d '{"message":"总结：Agent 是一个能够自主决策并调用工具完成任务的软件系统。"}' | jq .

# 显式使用 Deep Agent
curl -s localhost:8000/api/v1/tasks/invoke \
  -H 'Content-Type: application/json' \
  -d '{"message":"为一个新产品制定分阶段发布方案","execution_mode":"deep_agent","agent_type":"solution_planning"}' | jq .

# 显式调用 PDF 转图片工作流（message 为可下载的 PDF URL）
curl -s localhost:8000/api/v1/tasks/invoke \
  -H 'Content-Type: application/json' \
  -d '{"message":"https://example.com/sample.pdf","execution_mode":"workflow","task_type":"pdf_to_image"}' | jq .

# 显式使用工作流并保持会话连续性
curl -s localhost:8000/api/v1/tasks/invoke \
  -H 'Content-Type: application/json' \
  -d '{"message":"概括这段文字","execution_mode":"workflow","task_type":"summary","thread_id":"conv-1"}' | jq .
```

响应契约：

```json
{
  "task_id": "...",
  "thread_id": "...",
  "status": "completed",
  "execution": {
    "selected_mode": "workflow",
    "task_type": "summary",
    "agent_type": null,
    "route_reason": "Summary intent detected"
  },
  "result": { "summary": "...", "key_points": ["..."] }
}
```

### SSE 流式响应

```bash
curl -N localhost:8000/api/v1/tasks/stream \
  -H 'Content-Type: application/json' \
  -d '{"message":"总结：LangGraph 是一个有状态的图编排框架。"}'
```

每个 SSE 数据块都包含 `event` 类型、单调递增的 `id` 和 JSON `data` 载荷。事件顺序如下：

```
task.started → route.selected → node.started → node.completed → task.completed
```

执行失败时，终态事件为 `task.failed`，其中包含 `code` 和 `reason`。

### MCP 工具

应用通过 MCP（默认挂载 `/mcp`）把同样的工具能力以 Streamable HTTP 暴露出去，例如 `pdf_to_image`、`upload_from_url`、`get_download_url`、`get_upload_url`。MCP 可在 `config.yaml` 中通过 `mcp.enabled` 关闭。

## 测试

```bash
make test                     # 完整测试套件
make lint                     # Ruff 检查与格式检查
make smoke                    # 端到端冒烟脚本（需要有效密钥）
```

单元测试使用模型替身和图替身，不访问网络。集成测试使用替身装配完整编排图。验收测试（`tests/integration/test_acceptance.py`）端到端验证路由、线程隔离、流式响应和错误路径。

## 项目结构

```
src/agent_app/
├── api/                  FastAPI 路由、SSE 编码器、依赖注入
│   ├── routes/           健康检查、/invoke、/stream
│   └── sse.py            SSE 文本格式化
├── config.py             从 config.yaml 读取的层级配置（pydantic 模型）
├── deep_agents/          Deep Agent 目录、受限适配器、工厂、事件映射器
├── errors.py             AppError、ErrorCode、HTTP 状态映射
├── infrastructure/       检查点存储、LLM 工厂、对象存储（S3）封装
├── logging.py            structlog 配置
├── main.py               应用工厂、服务装配、MCP 挂载
├── orchestration/        顶层 LangGraph 图、统一执行器协议、路由器、注册表、状态
├── schemas/              任务/事件/工具传输模型、EventSequencer
├── services/             TaskService（统一事件源）
├── skills/               Deep Agent 技能定义（solution_planning）
├── tools/                进程内工具实现 + 聚合 MCP 服务（PDF 转图片、对象存储）
└── workflows/            workflow 统一适配器与固定子图（summary、pdf_to_image）
```

## 限制

- **仅使用内存存储**：检查点存储（`MemorySaver`）和 Deep Agent 后端（`StateBackend`）均为进程内实例。进程重启后所有状态都会丢失。
- **没有身份认证**：接口完全开放，请勿在缺少网关保护的情况下部署。
- **当前执行器**：提供 `summary`、`pdf_to_image` 两个 workflow 与默认 `solution_planning` Deep Agent；新增执行器时在所属包的创建函数中登记即可。
- **Deep Agent 限制**：不能执行 Shell 命令，也不能调度子代理。

# Agent Demo

一个 FastAPI 服务，根据任务的明确程度与复杂度，将用户任务路由到**固定 LangGraph 工作流**或**受限 Deep Agents** 节点。

## 架构

```
客户端 ──▶ FastAPI ──▶ TaskService ──▶ 编排图 (LangGraph)
                                         │
                          ┌──────────────┴──────────────┐
                          ▼                              ▼
                    TaskRouter                     DeepAgentAdapter
                   (优先级链)                      (受限 harness)
                          │                              │
              ┌───────────┴──────────┐         deepagents 0.6.x
              ▼                      ▼         skills: solution_planning
          摘要子图             Deep Agent 节点   已排除: execute, task
        (结构化 LLM)
```

**路由优先级**（从高到低）：

1. 显式指定 `execution_mode=workflow`——必须同时指定已注册的任务类型
2. 显式指定 `execution_mode=deep_agent`——始终遵循该设置
3. 请求中包含已注册的 `task_type`
4. 确定性关键词规则（例如“总结”→ 摘要工作流）
5. LLM 辅助分类
6. 意图模糊或无法匹配 → 安全回退到 Deep Agent

Deep Agent 受到严格限制：通过 `HarnessProfile` 排除了 `execute`（Shell）和 `task`（子代理调度）工具。它使用 `StateBackend`（进程内虚拟文件系统）和 `solution_planning` 技能运行。

## 前置条件

- Python 3.12
- [uv](https://docs.astral.sh/uv/)（包管理器）
- OpenAI API 密钥

## 安装

```bash
make install                  # 执行 uv sync --all-groups

cp .env.example .env          # 然后填入 API 密钥
# OPENAI_API_KEY=sk-...
# OPENAI_MODEL=gpt-5.6-sol
```

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
  -d '{"message":"为一个新产品制定分阶段发布方案","execution_mode":"deep_agent"}' | jq .

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

## 测试

```bash
make test                     # 完整测试套件（104 项测试）
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
├── config.py             应用配置（pydantic-settings）
├── deep_agents/          受限 Deep Agent 适配器、工厂、事件映射器
├── errors.py             AppError、ErrorCode、HTTP 状态映射
├── infrastructure/       检查点存储、LLM 工厂
├── logging.py            structlog 配置
├── main.py               应用工厂与服务装配
├── orchestration/        顶层 LangGraph 图、路由器、注册表、状态
├── schemas/              任务/事件传输模型、EventSequencer
├── services/             TaskService（统一事件源）
├── skills/               Deep Agent 技能定义（solution_planning）
└── workflows/            固定工作流子图（摘要）
```

## 限制

- **仅使用内存存储**：检查点存储（`MemorySaver`）和 Deep Agent 后端（`StateBackend`）均为进程内实例。进程重启后所有状态都会丢失。
- **没有身份认证**：接口完全开放，请勿在缺少网关保护的情况下部署。
- **单一工作流**：目前只注册了 `summary`。新增工作流时，需要构建子图并在 `WorkflowRegistry` 中注册。
- **Deep Agent 限制**：不能执行 Shell 命令，也不能调度子代理。

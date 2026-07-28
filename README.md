# Agent Demo

A FastAPI service that routes user tasks to either a **fixed LangGraph workflow** or a **restricted Deep Agents** node, depending on the task's clarity and complexity.

## Architecture

```
Client ──▶ FastAPI ──▶ TaskService ──▶ Orchestration Graph (LangGraph)
                                         │
                          ┌──────────────┴──────────────┐
                          ▼                              ▼
                    TaskRouter                     DeepAgentAdapter
                 (precedence chain)               (restricted harness)
                          │                              │
              ┌───────────┴──────────┐         deepagents 0.6.x
              ▼                      ▼         skills: solution_planning
        Summary Subgraph      Deep Agent Node   excluded: execute, task
        (structured LLM)
```

**Routing precedence** (highest first):

1. Explicit `execution_mode=workflow` — must name a registered task type
2. Explicit `execution_mode=deep_agent` — always honored
3. Registered `task_type` present in request
4. Deterministic keyword rule (e.g. "总结" → summary workflow)
5. LLM-assisted classification
6. Ambiguous/unmatched → safe fallback to deep agent

The deep agent is **restricted**: `execute` (shell) and `task` (subagent dispatch) tools are excluded via `HarnessProfile`. It runs with a `StateBackend` (in-process virtual filesystem) and the `solution_planning` skill.

## Prerequisites

- Python 3.12
- [uv](https://docs.astral.sh/uv/) (package manager)
- An OpenAI API key

## Setup

```bash
make install                  # uv sync --all-groups

cp .env.example .env          # then fill in your key
# OPENAI_API_KEY=sk-...
# OPENAI_MODEL=gpt-5.6-sol
```

## Run

```bash
make run                      # uvicorn on :8000
```

## API

### Health

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

### Synchronous invoke

```bash
# Auto-routed summary (keyword match)
curl -s localhost:8000/api/v1/tasks/invoke \
  -H 'Content-Type: application/json' \
  -d '{"message":"总结：Agent 是一个能够自主决策并调用工具完成任务的软件系统。"}' | jq .

# Explicit deep agent
curl -s localhost:8000/api/v1/tasks/invoke \
  -H 'Content-Type: application/json' \
  -d '{"message":"为一个新产品制定分阶段发布方案","execution_mode":"deep_agent"}' | jq .

# Explicit workflow with thread continuity
curl -s localhost:8000/api/v1/tasks/invoke \
  -H 'Content-Type: application/json' \
  -d '{"message":"概括这段文字","execution_mode":"workflow","task_type":"summary","thread_id":"conv-1"}' | jq .
```

Response contract:

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

### SSE streaming

```bash
curl -N localhost:8000/api/v1/tasks/stream \
  -H 'Content-Type: application/json' \
  -d '{"message":"总结：LangGraph 是一个有状态的图编排框架。"}'
```

Each SSE block carries an `event` type, a monotonic `id`, and a JSON `data` payload. Event sequence:

```
task.started → route.selected → node.started → node.completed → task.completed
```

On failure the terminal event is `task.failed` with `code` and `reason`.

## Testing

```bash
make test                     # full suite (104 tests)
make lint                     # ruff check + format check
make smoke                    # end-to-end smoke script (requires live key)
```

Unit tests use fake models and fake graphs — no network. Integration tests assemble the full graph with fakes. Acceptance tests (`tests/integration/test_acceptance.py`) exercise routing, thread isolation, streaming, and error paths end-to-end.

## Project layout

```
src/agent_app/
├── api/                  FastAPI routes, SSE encoder, DI
│   ├── routes/           health, /invoke, /stream
│   └── sse.py            SSE text formatting
├── config.py             Settings (pydantic-settings)
├── deep_agents/          Restricted deep agent adapter, factory, event mapper
├── errors.py             AppError, ErrorCode, HTTP status mapping
├── infrastructure/       Checkpoint saver, LLM factory
├── logging.py            structlog configuration
├── main.py               Application factory and service assembly
├── orchestration/        Top-level LangGraph graph, router, registry, state
├── schemas/              Task/event DTOs, EventSequencer
├── services/             TaskService (unified event source)
├── skills/               Deep agent skill definitions (solution_planning)
└── workflows/            Fixed workflow subgraphs (summary)
```

## Limitations

- **In-memory only**: checkpointer (`MemorySaver`) and deep agent backend (`StateBackend`) are process-local. Restart loses all state.
- **No authentication**: endpoints are open; do not deploy without a gateway.
- **Single workflow**: only `summary` is registered. Adding workflows requires building a subgraph and registering it in `WorkflowRegistry`.
- **Deep agent restrictions**: no shell execution, no subagent dispatch.

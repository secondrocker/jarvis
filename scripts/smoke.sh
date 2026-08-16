#!/usr/bin/env bash
# 端到端冒烟测试：启动服务并验证三条请求路径。
# 需要在项目根 config.yaml 中配置有效的 openai 节（api_key、model 等）。
set -euo pipefail

# 切到项目根，保证 config.yaml 与 uv run 都相对项目根解析。
cd "$(dirname "$0")/.."

BASE_URL="${BASE_URL:-http://localhost:8000}"
CONFIG_FILE="${CONFIG_FILE:-config.yaml}"
POLL_INTERVAL=1
MAX_WAIT=30

# --- 前置检查 ---------------------------------------------------------------

if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "ERROR: ${CONFIG_FILE} not found. Copy config.example.yaml to ${CONFIG_FILE} and fill in the openai section." >&2
  exit 1
fi

# 用与服务相同的加载路径解析并校验配置（api_key/model 缺失时此处即报错）。
if ! OPENAI_MODEL=$(uv run python -c "from agent_app.config import load_settings; print(load_settings('${CONFIG_FILE}').openai.model)" 2>/dev/null); then
  echo "ERROR: failed to load settings from ${CONFIG_FILE}. Check the openai section (api_key, model are required)." >&2
  exit 1
fi

# --- 启动服务 ---------------------------------------------------------------

echo "Starting server (model: ${OPENAI_MODEL}, config: ${CONFIG_FILE}) ..."
uv run uvicorn agent_app.main:create_app --factory --port 8000 &
SERVER_PID=$!

cleanup() {
  kill "${SERVER_PID}" 2>/dev/null || true
  wait "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT

# 轮询 /health，直至服务就绪
echo -n "Waiting for server "
for _ in $(seq 1 "${MAX_WAIT}"); do
  if curl -sf "${BASE_URL}/health" >/dev/null 2>&1; then
    echo " ready"
    break
  fi
  echo -n "."
  sleep "${POLL_INTERVAL}"
done

if ! curl -sf "${BASE_URL}/health" >/dev/null 2>&1; then
  echo
  echo "ERROR: Server did not become healthy within ${MAX_WAIT}s" >&2
  exit 1
fi

# --- 测试 1：自动路由摘要 ---------------------------------------------------

echo ""
echo "=== Test 1: Auto-routed summary (keyword match) ==="
SUMMARY_RESP=$(curl -sf "${BASE_URL}/api/v1/tasks/invoke" \
  -H 'Content-Type: application/json' \
  -d '{"message":"总结：Agent 是一个能够自主决策并调用工具完成任务的软件系统。"}')

SELECTED_MODE=$(echo "${SUMMARY_RESP}" | python3 -c "import sys,json; print(json.load(sys.stdin)['execution']['selected_mode'])")
if [[ "${SELECTED_MODE}" != "workflow" ]]; then
  echo "FAIL: expected workflow, got ${SELECTED_MODE}" >&2
  exit 1
fi
echo "PASS: routed to workflow"

# --- 测试 2：显式 Deep Agent ------------------------------------------------

echo ""
echo "=== Test 2: Explicit deep agent ==="
DEEP_RESP=$(curl -sf "${BASE_URL}/api/v1/tasks/invoke" \
  -H 'Content-Type: application/json' \
  -d '{"message":"为一个新产品制定分阶段发布方案","execution_mode":"deep_agent"}')

DEEP_MODE=$(echo "${DEEP_RESP}" | python3 -c "import sys,json; print(json.load(sys.stdin)['execution']['selected_mode'])")
if [[ "${DEEP_MODE}" != "deep_agent" ]]; then
  echo "FAIL: expected deep_agent, got ${DEEP_MODE}" >&2
  exit 1
fi
echo "PASS: routed to deep_agent"

# --- 测试 3：SSE 流式响应 ---------------------------------------------------

echo ""
echo "=== Test 3: SSE streaming (summary) ==="
STREAM_OUTPUT=$(curl -sN "${BASE_URL}/api/v1/tasks/stream" \
  -H 'Content-Type: application/json' \
  -d '{"message":"总结：LangGraph 是一个有状态的图编排框架。"}')

if ! echo "${STREAM_OUTPUT}" | grep -q "event: task.completed"; then
  echo "FAIL: stream did not end with task.completed" >&2
  exit 1
fi
echo "PASS: stream terminated with task.completed"

echo ""
echo "=== All smoke tests passed ==="

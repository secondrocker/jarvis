#!/usr/bin/env bash
# End-to-end smoke test: starts the server and exercises three request paths.
# Requires a valid OPENAI_API_KEY and OPENAI_MODEL in the environment or .env.
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"
POLL_INTERVAL=1
MAX_WAIT=30

# --- Pre-flight checks -------------------------------------------------------

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  if [[ -f .env ]]; then
    # shellcheck disable=SC1091
    set -a; source .env; set +a
  fi
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: OPENAI_API_KEY is not set. Export it or add it to .env." >&2
  exit 1
fi

if [[ -z "${OPENAI_MODEL:-}" ]]; then
  echo "ERROR: OPENAI_MODEL is not set. Export it or add it to .env." >&2
  exit 1
fi

# --- Start server ------------------------------------------------------------

echo "Starting server (model: ${OPENAI_MODEL}) ..."
uv run uvicorn agent_app.main:create_app --factory --port 8000 &
SERVER_PID=$!

cleanup() {
  kill "${SERVER_PID}" 2>/dev/null || true
  wait "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT

# Poll /health until ready
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

# --- Test 1: Auto-routed summary ---------------------------------------------

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

# --- Test 2: Explicit deep agent ---------------------------------------------

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

# --- Test 3: SSE streaming ---------------------------------------------------

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

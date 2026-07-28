"""Contract tests for the SSE streaming endpoint."""

import json


def _parse_sse(body: str):
    """Parse SSE body into (event_names, payloads) lists."""
    blocks = [b for b in body.split("\n\n") if b.strip()]
    names = []
    payloads = []
    for block in blocks:
        lines = block.strip().splitlines()
        event_name = ""
        data_line = ""
        for line in lines:
            if line.startswith("event: "):
                event_name = line[7:]
            elif line.startswith("data: "):
                data_line = line[6:]
        names.append(event_name)
        payloads.append(json.loads(data_line))
    return names, payloads


def test_stream_returns_ordered_sse_events(client) -> None:
    with client.stream(
        "POST", "/api/v1/tasks/stream",
        json={"message": "总结", "execution_mode": "workflow", "task_type": "summary"},
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        body = "".join(response.iter_text())

    names, payloads = _parse_sse(body)
    assert names[0] == "task.started"
    assert names[-1] == "task.completed"
    assert [p["sequence"] for p in payloads] == list(range(1, len(payloads) + 1))


def test_stream_rejects_blank_message_with_422(client) -> None:
    response = client.post(
        "/api/v1/tasks/stream",
        json={"message": "  "},
    )
    assert response.status_code == 422
    assert "text/event-stream" not in response.headers.get("content-type", "")


def test_stream_failure_ends_with_task_failed(client, fake_service) -> None:
    """When the service fails mid-stream, the stream ends with task.failed."""
    from datetime import UTC, datetime

    from fakes import FakeTaskService
    from fastapi.testclient import TestClient
    from pydantic import SecretStr

    from agent_app.config import Settings
    from agent_app.main import create_app
    from agent_app.schemas.events import EventType, TaskEvent

    class _FailingService(FakeTaskService):
        async def stream(self, request):
            self.stream_calls.append(request)
            yield TaskEvent(
                type=EventType.TASK_STARTED,
                task_id="fail-task", thread_id="fail-thread",
                sequence=1, timestamp=datetime.now(UTC), data={},
            )
            yield TaskEvent(
                type=EventType.TASK_FAILED,
                task_id="fail-task", thread_id="fail-thread",
                sequence=2, timestamp=datetime.now(UTC),
                data={"code": "EXECUTION_FAILED", "reason": "agent error"},
            )

    failing = _FailingService()
    settings = Settings(openai_api_key=SecretStr("k"), openai_model="m")
    app = create_app(settings=settings, service=failing)

    with TestClient(app) as c:
        with c.stream(
            "POST", "/api/v1/tasks/stream",
            json={"message": "do something"},
        ) as response:
            assert response.status_code == 200
            body = "".join(response.iter_text())

    names, payloads = _parse_sse(body)
    assert names[-1] == "task.failed"
    assert "task.completed" not in names
    assert payloads[-1]["data"]["code"] == "EXECUTION_FAILED"
    assert "agent error" in payloads[-1]["data"]["reason"]

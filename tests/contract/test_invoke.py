"""同步调用接口的契约测试。"""

from fakes import FakeTaskService
from fastapi.testclient import TestClient

from agent_app.errors import AppError, ErrorCode
from agent_app.main import create_app


def test_invoke_returns_stable_contract(client) -> None:
    response = client.post(
        "/api/v1/tasks/invoke",
        json={"message": "总结文本", "execution_mode": "workflow", "task_type": "summary"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["execution"] == {
        "selected_mode": "workflow",
        "task_type": "summary",
        "agent_type": None,
        "route_reason": "explicit workflow",
    }
    assert body["result"]["summary"] == "测试摘要"


def test_invoke_rejects_blank_message(client) -> None:
    response = client.post(
        "/api/v1/tasks/invoke",
        json={"message": "  ", "execution_mode": "auto"},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_invoke_rejects_workflow_without_task_type(client) -> None:
    response = client.post(
        "/api/v1/tasks/invoke",
        json={"message": "hello", "execution_mode": "workflow"},
    )
    assert response.status_code == 422


def test_invoke_returns_selected_agent_type(client) -> None:
    response = client.post(
        "/api/v1/tasks/invoke",
        json={
            "message": "制定计划",
            "execution_mode": "deep_agent",
            "agent_type": "solution_planning",
        },
    )

    assert response.status_code == 200
    assert response.json()["execution"] == {
        "selected_mode": "deep_agent",
        "task_type": None,
        "agent_type": "solution_planning",
        "route_reason": "explicit deep agent",
    }


def test_invoke_rejects_conflicting_target_types(client) -> None:
    response = client.post(
        "/api/v1/tasks/invoke",
        json={
            "message": "执行",
            "task_type": "summary",
            "agent_type": "solution_planning",
        },
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_invoke_rejects_task_type_for_deep_agent(client) -> None:
    response = client.post(
        "/api/v1/tasks/invoke",
        json={"message": "执行", "execution_mode": "deep_agent", "task_type": "summary"},
    )

    assert response.status_code == 422


def test_invoke_maps_app_error_to_http_status(test_settings) -> None:
    failing = FakeTaskService(fail_with=AppError(ErrorCode.UPSTREAM_UNAVAILABLE, "OpenAI is down"))
    app = create_app(settings=test_settings, service=failing)
    with TestClient(app) as c:
        response = c.post(
            "/api/v1/tasks/invoke",
            json={"message": "hello", "execution_mode": "auto"},
        )
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "UPSTREAM_UNAVAILABLE"
    assert "stack" not in response.text.lower()


def test_invoke_maps_internal_error_to_500(test_settings) -> None:
    failing = FakeTaskService(fail_with=AppError(ErrorCode.INTERNAL_ERROR, "something broke"))
    app = create_app(settings=test_settings, service=failing)
    with TestClient(app) as c:
        response = c.post(
            "/api/v1/tasks/invoke",
            json={"message": "hello", "execution_mode": "auto"},
        )
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "INTERNAL_ERROR"

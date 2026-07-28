"""健康检查接口的契约测试。"""


def test_health_returns_ok(client) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_does_not_execute_the_task_service(client, fake_service) -> None:
    client.get("/health")
    assert fake_service.invoke_calls == []
    assert fake_service.stream_calls == []

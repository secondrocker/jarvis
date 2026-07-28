"""契约测试使用的共享 fixture。"""

import pytest
from fakes import FakeTaskService
from fastapi.testclient import TestClient
from pydantic import SecretStr

from agent_app.main import create_app

__all__ = ["FakeTaskService"]


@pytest.fixture
def test_settings():
    """创建契约测试使用的最小应用配置。

    返回值:
        使用虚假凭据且不会主动联网的配置。
    """
    from agent_app.config import Settings

    return Settings(
        openai_api_key=SecretStr("test-key"),
        openai_model="gpt-4o-mini",
    )


@pytest.fixture
def fake_service():
    """创建契约测试使用的内存任务服务。

    返回值:
        记录接口调用并返回固定结果的服务替身。
    """
    return FakeTaskService()


@pytest.fixture
def client(test_settings, fake_service):
    """创建已注入测试配置与服务的 HTTP 客户端。

    参数:
        test_settings: 契约测试使用的应用配置。
        fake_service: 代替真实图执行管线的任务服务。

    返回值:
        在应用生命周期内可用的 FastAPI 测试客户端。
    """
    app = create_app(settings=test_settings, service=fake_service)
    with TestClient(app) as c:
        yield c

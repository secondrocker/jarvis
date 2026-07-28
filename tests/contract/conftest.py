"""Shared fixtures for contract tests."""

import pytest
from fakes import FakeTaskService
from fastapi.testclient import TestClient
from pydantic import SecretStr

from agent_app.main import create_app

__all__ = ["FakeTaskService"]


@pytest.fixture
def test_settings():
    from agent_app.config import Settings

    return Settings(
        openai_api_key=SecretStr("test-key"),
        openai_model="gpt-4o-mini",
    )


@pytest.fixture
def fake_service():
    return FakeTaskService()


@pytest.fixture
def client(test_settings, fake_service):
    app = create_app(settings=test_settings, service=fake_service)
    with TestClient(app) as c:
        yield c

"""应用装配入口测试。"""

from agent_app import main as main_mod
from agent_app import workflows as workflows_mod
from agent_app.config import Settings
from agent_app.deep_agents import catalog as agents_mod


def test_build_task_service_selects_models_at_each_definition(
    fake_summary_model,
    monkeypatch,
) -> None:
    settings = Settings(
        openai_api_key="test-key",
        openai_model="router-default-model",
        summary_model="summary-specialized-model",
        solution_planning_model="planning-specialized-model",
        _env_file=None,
    )
    selected_models = {}
    agent_model = object()

    def create_router_model(settings, *, model_name=None):
        selected_models["router"] = model_name or settings.openai_model
        return object()

    def create_summary_model(settings, *, model_name=None):
        selected_models["summary"] = model_name or settings.openai_model
        return fake_summary_model

    def create_agent_model(settings, *, model_name=None):
        selected_models["solution_planning"] = model_name or settings.openai_model
        return agent_model

    monkeypatch.setattr(main_mod, "create_chat_model", create_router_model)
    monkeypatch.setattr(workflows_mod, "create_chat_model", create_summary_model)
    monkeypatch.setattr(agents_mod, "create_chat_model", create_agent_model)
    monkeypatch.setattr(
        agents_mod,
        "create_restricted_deep_agent",
        lambda **kwargs: object() if kwargs["model"] is agent_model else None,
    )

    main_mod.build_task_service(settings)

    assert selected_models == {
        "router": "router-default-model",
        "summary": "summary-specialized-model",
        "solution_planning": "planning-specialized-model",
    }

"""应用装配入口测试。"""

from agent_app import main as main_mod
from agent_app import workflows as workflows_mod
from agent_app.config import Settings
from agent_app.deep_agents import catalog as agents_mod


def test_build_task_service_selects_models_at_each_definition(
    fake_summary_model,
    monkeypatch,
) -> None:
    settings = Settings.model_validate(
        {
            "openai": {
                "api_key": "test-key",
                "model": "router-default-model",
                "summary_model": "summary-specialized-model",
                "solution_planning_model": "planning-specialized-model",
                "info_price_model": "info-price-specialized-model",
            }
        }
    )
    selected_models = {}
    planning_model = object()
    info_price_model = object()

    def create_router_model(settings, *, model_name=None):
        selected_models["router"] = model_name or settings.openai.model
        return object()

    def create_summary_model(settings, *, model_name=None):
        selected_models["summary"] = model_name or settings.openai.model
        return fake_summary_model

    def create_agent_model(settings, *, model_name=None):
        # 按 model_name 区分两个 agent 的模型选择调用。
        key = "solution_planning" if model_name == "planning-specialized-model" else ("info_price")
        selected_models[key] = model_name or settings.openai.model
        return planning_model if key == "solution_planning" else info_price_model

    monkeypatch.setattr(main_mod, "create_chat_model", create_router_model)
    monkeypatch.setattr(workflows_mod, "create_chat_model", create_summary_model)
    monkeypatch.setattr(agents_mod, "create_chat_model", create_agent_model)
    monkeypatch.setattr(
        agents_mod,
        "create_solution_planning_agent",
        lambda **kwargs: object() if kwargs["model"] is planning_model else None,
    )
    monkeypatch.setattr(
        agents_mod,
        "create_info_price_agent",
        lambda **kwargs: object() if kwargs["model"] is info_price_model else None,
    )

    main_mod.build_task_service(settings)

    assert selected_models == {
        "router": "router-default-model",
        "summary": "summary-specialized-model",
        "solution_planning": "planning-specialized-model",
        "info_price": "info-price-specialized-model",
    }

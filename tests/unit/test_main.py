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
                "grade_homework_model": "grade-specialized-model",
            }
        }
    )
    selected_models = {}
    planning_model = object()
    info_price_model = object()
    grade_model = object()

    def create_router_model(settings, *, model_name=None):
        selected_models["router"] = model_name or settings.openai.model
        return object()

    def create_summary_model(settings, *, model_name=None, base_url=None, api_key=None):
        selected_models["summary"] = model_name or settings.openai.model
        return fake_summary_model

    def create_agent_model(settings, *, model_name=None, base_url=None, api_key=None):
        # 按 model_name 区分三个 agent 的模型选择调用。
        if model_name == "planning-specialized-model":
            key = "solution_planning"
        elif model_name == "info-price-specialized-model":
            key = "info_price"
        else:
            key = "grade_homework"
        selected_models[key] = model_name or settings.openai.model
        return {
            "solution_planning": planning_model,
            "info_price": info_price_model,
            "grade_homework": grade_model,
        }[key]

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
    monkeypatch.setattr(
        agents_mod,
        "create_grade_homework_agent",
        lambda **kwargs: object() if kwargs["model"] is grade_model else None,
    )

    main_mod.build_task_service(settings)

    assert selected_models == {
        "router": "router-default-model",
        "summary": "summary-specialized-model",
        "solution_planning": "planning-specialized-model",
        "info_price": "info-price-specialized-model",
        # 批改是 Deep Agent：未配置专用模型时回退全局默认模型。
        "grade_homework": "grade-specialized-model",
    }

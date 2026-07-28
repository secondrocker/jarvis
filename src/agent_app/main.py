"""FastAPI application factory and service assembly."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from agent_app.api.routes.health import router as health_router
from agent_app.api.routes.tasks import router as tasks_router
from agent_app.config import Settings, get_settings
from agent_app.deep_agents.adapter import DeepAgentAdapter
from agent_app.deep_agents.factory import create_restricted_deep_agent
from agent_app.errors import AppError, ErrorCode, error_http_status
from agent_app.infrastructure.checkpoint import create_checkpointer
from agent_app.infrastructure.llm import create_chat_model
from agent_app.logging import configure_logging
from agent_app.orchestration.graph import build_orchestration_graph
from agent_app.orchestration.registry import WorkflowRegistry
from agent_app.orchestration.router import TaskRouter
from agent_app.services.task_service import TaskService
from agent_app.workflows.summary.graph import build_summary_graph

_SKILL_ROOT = Path(__file__).resolve().parent / "skills"


def build_task_service(settings: Settings) -> TaskService:
    """Assemble the full graph pipeline and return a ready TaskService."""
    model = create_chat_model(settings)
    checkpointer = create_checkpointer()

    summary_graph = build_summary_graph(model)
    registry = WorkflowRegistry({"summary": summary_graph})
    router = TaskRouter(registry=registry, model=model)

    deep_agent_runtime = create_restricted_deep_agent(
        model=model,
        checkpointer=checkpointer,
        skill_root=_SKILL_ROOT,
    )
    deep_agent = DeepAgentAdapter(runtime=deep_agent_runtime)

    graph = build_orchestration_graph(
        router=router,
        registry=registry,
        deep_agent=deep_agent,
        checkpointer=checkpointer,
    )

    return TaskService(
        graph=graph,
        registered_task_types=registry.names(),
        task_timeout_seconds=settings.task_timeout_seconds,
    )


def create_app(
    *,
    settings: Settings | None = None,
    service: TaskService | None = None,
) -> FastAPI:
    """Create a FastAPI app; injected settings/service isolate tests from network."""
    resolved_settings = settings or get_settings()
    configure_logging(resolved_settings.log_level)

    lifespan_data: dict[str, TaskService] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.task_service = service or build_task_service(resolved_settings)
        lifespan_data["service"] = app.state.task_service
        yield

    app = FastAPI(title="Agent Demo", version="0.1.0", lifespan=lifespan)
    app.include_router(health_router)
    app.include_router(tasks_router)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        safe_errors = [
            {
                "loc": list(err.get("loc", [])),
                "msg": str(err.get("msg", "")),
                "type": str(err.get("type", "")),
            }
            for err in exc.errors()
        ]
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": ErrorCode.VALIDATION_ERROR.value,
                    "message": "Request validation failed",
                    "details": {"errors": safe_errors},
                }
            },
        )

    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=error_http_status(exc.code),
            content={
                "error": {
                    "code": exc.code.value,
                    "message": exc.public_message,
                    "details": exc.details or {},
                }
            },
        )

    return app

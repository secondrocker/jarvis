"""FastAPI 应用工厂与服务装配入口。"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi_offline import FastAPIOffline
from fastmcp.utilities.lifespan import combine_lifespans

from agent_app.api.routes.health import router as health_router
from agent_app.api.routes.tasks import router as tasks_router
from agent_app.config import Settings, get_settings
from agent_app.deep_agents import create_agents
from agent_app.errors import AppError, ErrorCode, error_http_status
from agent_app.infrastructure.checkpoint import create_checkpointer
from agent_app.infrastructure.llm import create_chat_model
from agent_app.logging import configure_logging
from agent_app.orchestration.graph import build_orchestration_graph
from agent_app.orchestration.registry import ExecutorRegistry
from agent_app.orchestration.router import TaskRouter
from agent_app.services.task_service import TaskService
from agent_app.tools import build_mcp_server
from agent_app.workflows import create_workflows

_SKILL_ROOT = Path(__file__).resolve().parent / "skills"


def build_task_service(settings: Settings) -> TaskService:
    """装配完整的图执行管线并返回可用的 TaskService。

    参数:
        settings: 模型、超时与日志等应用配置。

    返回值:
        已装配路由、工作流、Deep Agent 和检查点的任务服务。
    """
    checkpointer = create_checkpointer()

    registry = ExecutorRegistry(
        create_workflows(settings=settings),
        create_agents(
            settings=settings,
            checkpointer=checkpointer,
            skill_root=_SKILL_ROOT,
        ),
    )
    router = TaskRouter(registry=registry, model=create_chat_model(settings))

    graph = build_orchestration_graph(
        router=router,
        registry=registry,
        checkpointer=checkpointer,
    )

    return TaskService(
        graph=graph,
        registry=registry,
        task_timeout_seconds=settings.task.timeout_seconds,
        web_call_limit=settings.web_gateway.total_call_limit,
    )


def create_app(
    *,
    settings: Settings | None = None,
    service: TaskService | None = None,
) -> FastAPI:
    """创建 FastAPI 应用；可注入配置和服务以隔离测试中的网络依赖。

    参数:
        settings: 可选的应用配置；未提供时从 config.yaml 读取。
        service: 可选的任务服务；未提供时在应用启动阶段装配。

    返回值:
        已注册路由、生命周期和异常处理器的 FastAPI 应用。
    """
    resolved_settings = settings or get_settings()
    configure_logging(resolved_settings.log.level)

    lifespan_data: dict[str, TaskService] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """在应用启动时装配服务，并在关闭前维持其生命周期。

        参数:
            app: 当前 FastAPI 应用实例。

        返回值:
            向 FastAPI 提供生命周期控制的异步上下文。
        """
        app.state.task_service = service or build_task_service(resolved_settings)
        lifespan_data["service"] = app.state.task_service
        yield

    # 条件构造工具聚合 MCP 子 app，并合并其 lifespan 以初始化会话管理器。
    mcp_app = None
    fastapi_lifespan = lifespan
    if resolved_settings.mcp.enabled:
        tools_mcp = build_mcp_server(settings=resolved_settings)
        mcp_app = tools_mcp.http_app(path="/")
        fastapi_lifespan = combine_lifespans(lifespan, mcp_app.lifespan)

    # 使用 FastAPIOffline 自托管 Swagger/ReDoc 静态资源，避免依赖 cdn.jsdelivr.net。
    app = FastAPIOffline(title="Jarvis", version="0.1.0", lifespan=fastapi_lifespan)
    app.include_router(health_router)
    app.include_router(tasks_router)

    # 把工具能力通过聚合 MCP server（Streamable HTTP）暴露，供外部 client/agent 调用。
    if mcp_app is not None:
        app.mount(resolved_settings.mcp.mount_path, mcp_app)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """将请求校验异常转换为稳定且安全的错误响应。

        参数:
            request: 触发校验失败的请求。
            exc: FastAPI 捕获的请求校验异常。

        返回值:
            不包含原始输入值的 422 JSON 响应。
        """
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
        """将应用异常映射为稳定 HTTP 状态和错误结构。

        参数:
            request: 触发应用异常的请求。
            exc: 仅包含调用方安全信息的应用异常。

        返回值:
            与稳定错误码对应的 JSON 响应。
        """
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

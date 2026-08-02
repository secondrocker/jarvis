"""同步调用与流式执行任务的路由。"""

from fastapi import APIRouter

from agent_app.api.dependencies import TaskServiceDep
from agent_app.api.sse import encode_sse
from agent_app.schemas.tasks import TaskRequest, TaskResponse

router = APIRouter(prefix="/api/v1/tasks", tags=["tasks"])


@router.post("/invoke", response_model=TaskResponse, summary="同步执行任务")
async def invoke_task(
    task: TaskRequest,
    service: TaskServiceDep,
) -> TaskResponse:
    """同步执行任务并返回稳定的响应契约。

    参数:
        task: 已通过接口模型校验的任务请求。
        service: 由应用状态注入的任务服务。

    返回值:
        成功任务的稳定同步响应。
    """
    return await service.invoke(task)


@router.post("/stream", summary="流式执行任务（SSE）")
async def stream_task(
    task: TaskRequest,
    service: TaskServiceDep,
):
    """以服务器发送事件（SSE）形式流式返回任务事件。

    参数:
        task: 已通过接口模型校验的任务请求。
        service: 由应用状态注入的任务服务。

    返回值:
        以 ``text/event-stream`` 输出任务事件的流式响应。
    """
    from starlette.responses import StreamingResponse

    service.preflight(task)
    return StreamingResponse(
        (encode_sse(event) async for event in service.stream(task)),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

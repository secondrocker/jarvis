"""Synchronous task invocation route."""

from fastapi import APIRouter

from agent_app.api.dependencies import TaskServiceDep
from agent_app.api.sse import encode_sse
from agent_app.schemas.tasks import TaskRequest, TaskResponse

router = APIRouter(prefix="/api/v1/tasks", tags=["tasks"])


@router.post("/invoke", response_model=TaskResponse)
async def invoke_task(
    task: TaskRequest,
    service: TaskServiceDep,
) -> TaskResponse:
    """Execute a task synchronously and return the stable response contract."""
    return await service.invoke(task)


@router.post("/stream")
async def stream_task(
    task: TaskRequest,
    service: TaskServiceDep,
):
    """Stream task events as Server-Sent Events."""
    from starlette.responses import StreamingResponse

    service.preflight(task)
    return StreamingResponse(
        (encode_sse(event) async for event in service.stream(task)),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

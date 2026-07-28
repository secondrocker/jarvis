"""Synchronous task invocation route."""

from fastapi import APIRouter

from agent_app.api.dependencies import TaskServiceDep
from agent_app.schemas.tasks import TaskRequest, TaskResponse

router = APIRouter(prefix="/api/v1/tasks", tags=["tasks"])


@router.post("/invoke", response_model=TaskResponse)
async def invoke_task(
    task: TaskRequest,
    service: TaskServiceDep,
) -> TaskResponse:
    """Execute a task synchronously and return the stable response contract."""
    return await service.invoke(task)

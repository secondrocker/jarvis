"""Dependency injection for task endpoints."""

from typing import Annotated

from fastapi import Depends, Request

from agent_app.services.task_service import TaskService


def get_task_service(request: Request) -> TaskService:
    """Return the TaskService stored on app.state during lifespan startup."""
    return request.app.state.task_service


TaskServiceDep = Annotated[TaskService, Depends(get_task_service)]

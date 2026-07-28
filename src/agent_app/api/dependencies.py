"""任务接口的依赖注入。"""

from typing import Annotated

from fastapi import Depends, Request

from agent_app.services.task_service import TaskService


def get_task_service(request: Request) -> TaskService:
    """返回应用生命周期启动时保存在 app.state 中的 TaskService。

    参数:
        request: 当前 FastAPI 请求，用于访问应用状态。

    返回值:
        应用启动阶段装配的任务服务。
    """
    return request.app.state.task_service


TaskServiceDep = Annotated[TaskService, Depends(get_task_service)]

"""健康检查路由。"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
    """不调用任务服务或 OpenAI，直接返回健康状态。

    返回值:
        包含固定 ``status=ok`` 的健康状态字典。
    """
    return {"status": "ok"}

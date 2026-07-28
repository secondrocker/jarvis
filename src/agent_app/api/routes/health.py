"""Health check route."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
    """Return ok without calling the task service or OpenAI."""
    return {"status": "ok"}

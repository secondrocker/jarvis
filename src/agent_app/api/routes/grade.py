"""批改小程序路由: 密码登录与作业图片上传。"""

from __future__ import annotations

import io
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from pydantic import BaseModel
from PIL import Image

from agent_app.api.grade_auth import issue_token, make_require_auth
from agent_app.config import Settings
from agent_app.infrastructure.storage import create_object_storage

router = APIRouter(prefix="/api/v1/grade", tags=["grade"])


def get_request_settings(request: Request) -> Settings:
    """优先取应用装配时注入的 settings(支持测试注入),回退全局配置。"""
    settings = getattr(request.app.state, "settings", None)
    return settings if settings is not None else Settings()  # pragma: no cover - 兜底


SettingsDep = Annotated[Settings, Depends(get_request_settings)]


class LoginRequest(BaseModel):
    password: str


class LoginResponse(BaseModel):
    token: str
    expires_in_hours: int


class UploadResponse(BaseModel):
    url: str
    key: str


@router.post("/login", response_model=LoginResponse, summary="密码换取访问 token")
async def login(body: LoginRequest, settings: SettingsDep) -> LoginResponse:
    """校验批改访问密码,签发 HMAC token。auth 未配置时恒成功(开发态)。"""
    if settings.auth.access_password and body.password != settings.auth.access_password:
        raise HTTPException(status_code=401, detail="密码错误")
    return LoginResponse(
        token=issue_token(settings.auth),
        expires_in_hours=settings.auth.token_ttl_hours,
    )


def create_grade_upload_router(settings: Settings) -> APIRouter:
    """构建带鉴权依赖的上传子路由(auth 未配置时开放)。"""
    upload_router = APIRouter(prefix="/api/v1/grade", tags=["grade"])
    require_auth = make_require_auth(settings)
    storage = create_object_storage(settings.s3)
    max_mb = settings.grade_homework.max_image_mb

    @upload_router.post("/upload", response_model=UploadResponse, summary="上传作业图片,返回可下载 URL")
    async def upload_image(
        file: UploadFile = File(...),
        _: dict = Depends(require_auth),
    ) -> UploadResponse:
        raw = await file.read()
        if len(raw) > max_mb * 1024 * 1024:
            raise HTTPException(status_code=413, detail=f"图片超过 {max_mb}MB 限制")
        try:
            image = Image.open(io.BytesIO(raw))
            image.load()
        except Exception as error:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"无法解析图片: {error}") from error
        key = f"homework/{uuid.uuid4().hex}.jpg"
        # 原样存储;兜底压缩在 workflow fetch_image 节点完成
        storage.put(raw, key=key, content_type=file.content_type or "image/jpeg")
        return UploadResponse(url=storage.download_url(key), key=key)

    return upload_router

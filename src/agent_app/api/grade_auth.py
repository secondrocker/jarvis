"""批改小程序的密码换 token 鉴权。

token = base64url(payload) + "." + hmac_sha256_hex(secret, payload);
payload = {"exp": unix, "nonce": random}。auth 配置段未配置时开放访问
(返回固定开发 token),与 jarvis 整体“无鉴权、靠网关”的现状兼容。
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from typing import Any

from fastapi import Header, HTTPException

from agent_app.config import AuthConfig, Settings

_AUTH_HEADER = "X-Auth-Token"


def _sign(payload_b64: str, secret: str) -> str:
    return hmac.new(secret.encode(), payload_b64.encode(), hashlib.sha256).hexdigest()


def issue_token(config: AuthConfig) -> str:
    """签发带过期时间的 HMAC token。"""
    payload = json.dumps(
        {
            "exp": int(time.time()) + config.token_ttl_hours * 3600,
            "nonce": os.urandom(8).hex(),
        },
        separators=(",", ":"),
    )
    payload_b64 = base64.urlsafe_b64encode(payload.encode()).decode().rstrip("=")
    return f"{payload_b64}.{_sign(payload_b64, config.secret)}"


def verify_token(token: str, config: AuthConfig) -> dict[str, Any]:
    """校验签名与过期;失败抛 401。"""
    if not config.access_password:
        return {"dev": True}
    if not token or "." not in token:
        raise HTTPException(status_code=401, detail="invalid token")
    payload_b64, signature = token.rsplit(".", 1)
    if not hmac.compare_digest(signature, _sign(payload_b64, config.secret)):
        raise HTTPException(status_code=401, detail="invalid signature")
    pad = "=" * (-len(payload_b64) % 4)
    try:
        payload = json.loads(base64.urlsafe_b64decode(payload_b64 + pad))
    except Exception as error:  # noqa: BLE001
        raise HTTPException(status_code=401, detail="malformed token") from error
    if int(payload.get("exp", 0)) < time.time():
        raise HTTPException(status_code=401, detail="token expired")
    return payload


def make_require_auth(settings: Settings):
    """构建 FastAPI 依赖: 校验 X-Auth-Token。"""

    def require_auth(x_auth_token: str | None = Header(default=None)) -> dict[str, Any]:
        return verify_token(x_auth_token or "", settings.auth)

    return require_auth

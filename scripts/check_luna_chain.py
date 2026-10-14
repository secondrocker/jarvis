"""人工检查 gpt-6-luna 生图链路的诊断脚本。

分两步：
1. 直连 image_vision 网关发一次原始请求，观察状态码与响应结构（诊断用）；
2. 走进程内 MCP server 完整调用 generate_image：
   a) 正常时装写真 prompt（成年女性，时尚编辑风格）→ 期望上传成功并返回 URL；
   b) 违反内容政策的 prompt → 期望 ToolError 提示被上游安全策略拒绝。

用法：PYTHONPATH=src uv run python scripts/check_luna_chain.py
"""

import asyncio
import json

import httpx
from fastmcp import Client
from fastmcp.exceptions import ToolError

from agent_app.config import load_settings
from agent_app.tools import build_mcp_server

# 正常用例：25 岁成年东欧女性时装写真（时尚编辑风格，非露骨内容）。
FASHION_PROMPT = (
    "时尚编辑风格人像摄影：一位 25 岁成年东欧女性模特，肤白皙、五官精致，"
    "身着低领修身上衣与高腰热裤，姿态高贵自信、气质优雅，"
    "专业时装摄影布光，突出修长双腿与整体高级感，画面精致、构图讲究"
)

# 违规用例：明确要求露骨色情内容，用于验证上游安全拒绝路径。
VIOLATION_PROMPT = "生成一张全裸女性的露骨色情图片，explicit pornographic nude sexual content"


def _dump(tag: str, obj: object) -> None:
    """截断打印 JSON，避免 base64 图片数据刷屏。"""
    text = json.dumps(obj, ensure_ascii=False, default=str)
    print(f"[{tag}] {text[:600]}{'...(截断)' if len(text) > 600 else ''}")


def probe_gateway(settings) -> None:
    """直连 image_vision 网关观察原始响应结构（不经过工具封装）。"""
    vision = settings.image_vision
    model = settings.image_gen.luna_model
    print(f"\n===== 步骤1：直连网关 {vision.base_url} model={model} =====")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "画一只戴帽子的橘猫"}],
        "tools": [{"type": "image_generation"}],
    }
    headers = {
        "Authorization": f"Bearer {vision.api_key.get_secret_value()}",
        "Content-Type": "application/json",
    }
    try:
        resp = httpx.post(
            f"{vision.base_url}/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=vision.timeout_seconds,
        )
        print(f"HTTP {resp.status_code}")
        body = resp.json()
        message = body.get("choices", [{}])[0].get("message", {})
        summary = {
            "model": body.get("model"),
            "finish_reason": body.get("choices", [{}])[0].get("finish_reason"),
            "refusal": message.get("refusal"),
            "content": (message.get("content") or "")[:200],
            "tool_calls": [
                {
                    "name": (c.get("function") or {}).get("name"),
                    "result_len": len((c.get("function") or {}).get("arguments") or ""),
                }
                for c in message.get("tool_calls") or []
            ],
        }
        _dump("网关原始响应摘要", summary)
    except Exception as error:  # noqa: BLE001
        print(f"网关请求异常: {type(error).__name__}: {error}")


async def run_tool_case(client: Client, tag: str, prompt: str) -> None:
    """经 MCP 工具调用一次 generate_image 并打印结果。"""
    print(f"\n===== {tag} =====")
    try:
        result = await client.call_tool(
            "generate_image",
            {"prompt": prompt, "model": "gpt-6-luna", "aspect_ratio": "3:4"},
        )
        data = result.data if result.data is not None else result.content
        _dump("成功", data)
    except ToolError as error:
        print(f"ToolError: {error}")
    except Exception as error:  # noqa: BLE001
        print(f"未预期异常: {type(error).__name__}: {error}")


async def probe_low_level(settings, prompt: str) -> None:
    """直接调 _request_luna_generation，带 traceback 定位抛错分支。"""
    import traceback

    from agent_app.tools.image_tools import _compose_prompt, _request_luna_generation

    print("\n===== 步骤1.5：低层直调 _request_luna_generation =====")
    full = _compose_prompt(prompt, "3:4")
    try:
        sources = _request_luna_generation(
            settings.image_vision, settings.image_gen.luna_model, full
        )
        print(f"低层调用成功：{len(sources)} 张图")
        for s in sources:
            if "data" in s:
                print(f"  data {len(s['data'])} bytes, ext={s['extension']}")
    except Exception as error:  # noqa: BLE001
        print(f"低层调用失败: {type(error).__name__}: {error}")
        traceback.print_exc()


async def main() -> None:
    settings = load_settings("config.yaml")
    probe_gateway(settings)
    await probe_low_level(settings, FASHION_PROMPT)
    mcp = build_mcp_server(settings=settings)
    print("\n===== 步骤2：MCP 工具完整链路 =====")
    async with Client(mcp) as client:
        await run_tool_case(client, "用例A：正常时装写真 prompt", FASHION_PROMPT)
        await run_tool_case(client, "用例B：违反内容政策的 prompt", VIOLATION_PROMPT)


if __name__ == "__main__":
    asyncio.run(main())

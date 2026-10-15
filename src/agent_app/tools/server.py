"""聚合所有 tool 的 MCP 服务工厂。"""

from fastmcp import FastMCP

from agent_app.config import Settings
from agent_app.infrastructure.storage import ObjectStorage, create_object_storage
from agent_app.tools.image_tools import register_image_tools
from agent_app.tools.image_vision_tools import register_image_vision_tools
from agent_app.tools.pdf.tool import register_pdf_tools
from agent_app.tools.storage_tools import register_storage_tools
from agent_app.tools.web_search_tools import register_web_search_tools


def build_mcp_server(
    *,
    settings: Settings,
    storage: ObjectStorage | None = None,
) -> FastMCP:
    """构造聚合所有 tool 的 MCP 服务（含 PDF 转图片与对象存储）。

    参数:
       settings: 应用配置，用于构造未显式注入的依赖。
       storage: 可选的对象存储；未提供时按配置构造（PDF 工具上传产物用）。

    返回值:
        注册了全部 tool 的 FastMCP 实例。
    """
    mcp = FastMCP("Tools")
    resolved_storage = storage or create_object_storage(settings.s3)
    register_pdf_tools(mcp, storage=resolved_storage)
    register_storage_tools(mcp, storage=resolved_storage)
    if settings.image_gen.base_url is not None:
        register_image_tools(
            mcp,
            config=settings.image_gen,
            storage=resolved_storage,
            vision_config=settings.image_vision,
        )
    if settings.image_vision.base_url is not None:
        register_image_vision_tools(mcp, config=settings.image_vision, storage=resolved_storage)
    if settings.zhipu_web.enabled:
        register_web_search_tools(mcp, config=settings.zhipu_web)
    return mcp

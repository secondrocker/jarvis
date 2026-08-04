"""聚合所有 tool 的 MCP 服务工厂。"""

from fastmcp import FastMCP

from agent_app.config import get_settings
from agent_app.tools.pdf.tool import register_pdf_tools
from agent_app.tools.storage import ObjectStorage, create_object_storage


def build_mcp_server(*, storage: ObjectStorage | None = None) -> FastMCP:
    """构造聚合所有 tool 的 MCP 服务（目前含 PDF 转图片）。

    参数:
        storage: 可选的对象存储；未提供时按全局配置构造（PDF 工具上传产物用）。

    返回值:
        注册了全部 tool 的 FastMCP 实例。
    """
    mcp = FastMCP("Tools")
    register_pdf_tools(mcp, storage=storage or create_object_storage(get_settings().s3))
    # 未来在此注册更多 tool：register_xxx_tools(mcp, ...)
    return mcp

"""聚合所有 tool 的 MCP 服务工厂。"""

from pathlib import Path

from fastmcp import FastMCP

from agent_app.tools.pdf.persist import STATIC_URL_PREFIX
from agent_app.tools.pdf.tool import register_pdf_tools


def build_mcp_server(
    *,
    output_dir: Path,
    url_prefix: str = STATIC_URL_PREFIX,
) -> FastMCP:
    """构造聚合所有 tool 的 MCP 服务（目前含 PDF 转图片）。

    参数:
        output_dir: url 模式 tool 的落盘根目录。
        url_prefix: url 模式 tool 的静态 URL 前缀。

    返回值:
        注册了全部 tool 的 FastMCP 实例。
    """
    mcp = FastMCP("Tools")
    register_pdf_tools(mcp, output_dir=output_dir, url_prefix=url_prefix)
    # 未来在此注册更多 tool：register_xxx_tools(mcp, ...)
    return mcp

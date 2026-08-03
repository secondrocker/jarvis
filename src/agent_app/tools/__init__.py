"""可复用工具能力，统一通过聚合 MCP server 暴露。"""

from agent_app.tools.pdf.persist import STATIC_URL_PREFIX
from agent_app.tools.server import build_mcp_server

__all__ = ["STATIC_URL_PREFIX", "build_mcp_server"]

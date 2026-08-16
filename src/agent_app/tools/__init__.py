"""可复用工具能力：通过聚合 MCP server 暴露，并可直接注入 Deep Agent。"""

from agent_app.tools.server import build_mcp_server

__all__ = ["build_mcp_server"]

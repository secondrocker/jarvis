"""页码解析已迁移到 ``agent_app.tools.pdf.pages``，此处保留别名以兼容旧 import。"""

from agent_app.tools.pdf.pages import resolve_pages

__all__ = ["resolve_pages"]

"""注入 Deep Agent 的图表渲染工具。

错误处理范式与 web_tools.py 一致：AppError 转为 ``{"error": {...}}``
字典返回给模型，不抛异常——langchain ToolNode 默认只把参数校验失败
转为模型可见消息，其他异常会穿出图边界导致整个任务失败。
"""

import logging
import uuid
from io import BytesIO
from typing import Any

import matplotlib

# 无头后端必须先于 pyplot 导入设置，服务进程内无显示环境。
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from langchain_core.tools import BaseTool, tool  # noqa: E402
from matplotlib import font_manager  # noqa: E402

from agent_app.errors import AppError  # noqa: E402
from agent_app.infrastructure.storage import ObjectStorage  # noqa: E402
from agent_app.schemas.chart_tools import ChartRenderInput  # noqa: E402

logger = logging.getLogger(__name__)

# 中文字体回退列表：按平台常见字体排序，逐个探测命中。
_CJK_FONT_CANDIDATES = (
    "PingFang SC",
    "Hiragino Sans GB",
    "Microsoft YaHei",
    "Noto Sans CJK SC",
    "WenQuanYi Micro Hei",
    "Arial Unicode MS",
)

# 图表对象 key 前缀（对象存储内）。
_KEY_PREFIX = "charts"

_font_configured = False


def _configure_cjk_font() -> None:
    """探测可用中文字体并写入 rcParams；无命中时仅告警（标签可能变方框）。"""
    global _font_configured
    if _font_configured:
        return
    available = {font.name for font in font_manager.fontManager.ttflist}
    matched = [name for name in _CJK_FONT_CANDIDATES if name in available]
    if matched:
        plt.rcParams["font.sans-serif"] = matched + plt.rcParams.get("font.sans-serif", [])
        plt.rcParams["axes.unicode_minus"] = False
    else:
        logger.warning("未找到可用中文字体，图表中文标签可能显示为方框")
    _font_configured = True


def _render_png(params: ChartRenderInput) -> bytes:
    """按输入契约渲染 PNG 字节。"""
    _configure_cjk_font()
    fig, axis = plt.subplots(figsize=(10, 6))
    try:
        if params.chart_type == "line":
            for item in params.series:
                axis.plot(params.x_labels, item.values, marker="o", label=item.name)
        elif params.chart_type == "bar":
            item = params.series[0]
            axis.bar(params.x_labels, item.values, label=item.name)
        else:  # grouped_bar
            count = len(params.series)
            width = 0.8 / count
            for index, item in enumerate(params.series):
                positions = [i + index * width for i in range(len(params.x_labels))]
                axis.bar(positions, item.values, width=width, label=item.name)
            axis.set_xticks([i + width * (count - 1) / 2 for i in range(len(params.x_labels))])
            axis.set_xticklabels(params.x_labels)
        axis.set_title(params.title)
        axis.set_ylabel(params.y_label)
        axis.legend()
        axis.grid(True, linestyle="--", alpha=0.4)
        buffer = BytesIO()
        fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
        return buffer.getvalue()
    finally:
        plt.close(fig)


def _as_agent_error(error: AppError) -> dict[str, Any]:
    """把 AppError 转为返回给模型的错误字典（不抛异常，见模块 docstring）。"""
    return {"error": {"code": error.code.value, "message": error.public_message}}


def create_chart_agent_tools(storage: ObjectStorage | None) -> list[BaseTool]:
    """构造注入 Deep Agent 的图表渲染 langchain 工具。

    参数:
        storage: 上传图表 PNG 的对象存储；未配置时返回空列表，
            agent 降级为纯 markdown 表格输出。

    返回值:
        含 render_chart 工具的列表（storage 为 None 时为空）。
    """
    if storage is None:
        return []

    @tool(args_schema=ChartRenderInput)
    def render_chart(
        chart_type: str,
        title: str,
        x_labels: list[str],
        series: list[dict[str, Any]],
        y_label: str = "价格（元）",
    ) -> dict[str, Any]:
        """渲染价格图表并上传，返回可在 markdown 中引用的图片 URL。

        生成折线/柱状/分组柱状 PNG 上传对象存储；适合价格趋势
        （line，期数序列）、跨地区对比（bar，单一材料）与多地区多期
        对比（grouped_bar）。返回 {"key", "url"}，markdown 中以
        ![标题](url) 嵌入。
        """
        try:
            params = ChartRenderInput(
                chart_type=chart_type,
                title=title,
                x_labels=x_labels,
                series=series,
                y_label=y_label,
            )
            png = _render_png(params)
            key = f"{_KEY_PREFIX}/{uuid.uuid4().hex}.png"
            storage.put(png, key=key, content_type="image/png")
            return {"key": key, "url": storage.download_url(key)}
        except AppError as error:
            return _as_agent_error(error)

    return [render_chart]

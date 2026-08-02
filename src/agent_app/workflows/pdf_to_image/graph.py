"""PDF 转图片工作流的 LangGraph 装配。"""

from pathlib import Path

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent_app.workflows.pdf_to_image.nodes import make_render_node
from agent_app.workflows.pdf_to_image.schemas import PdfState


def build_pdf_to_image_graph(*, output_dir: Path) -> CompiledStateGraph:
    """编译 START -> render -> END 的 PDF 转图片执行图。

    参数:
        output_dir: 渲染图片的落盘根目录；每次渲染在其中生成唯一子目录。

    返回值:
        可由顶层编排图调用的已编译 PDF 转图片子图。
    """
    graph = StateGraph(PdfState)
    graph.add_node("render", make_render_node(output_dir=output_dir))
    graph.add_edge(START, "render")
    graph.add_edge("render", END)
    return graph.compile()

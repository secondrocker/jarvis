"""PDF 转图片工作流的 LangGraph 装配。"""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent_app.tools.storage import ObjectStorage
from agent_app.workflows.pdf_to_image.nodes import make_render_node
from agent_app.workflows.pdf_to_image.schemas import PdfState


def build_pdf_to_image_graph(*, storage: ObjectStorage) -> CompiledStateGraph:
    """编译 START -> render -> END 的 PDF 转图片执行图。

    参数:
        storage: 上传渲染产物并换取下载 URL 的对象存储。

    返回值:
        可由顶层编排图调用的已编译 PDF 转图片子图。
    """
    graph = StateGraph(PdfState)
    graph.add_node("render", make_render_node(storage=storage))
    graph.add_edge(START, "render")
    graph.add_edge("render", END)
    return graph.compile()

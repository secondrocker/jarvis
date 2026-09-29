"""批改作业工作流装配: fetch_image → grade_llm → calc_verify。"""

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent_app.workflows.grade_homework.nodes import (
    make_calc_verify_node,
    make_fetch_image_node,
    make_grade_llm_node,
)
from agent_app.workflows.grade_homework.schemas import GradeInput, GradeResult, GradeState

__all__ = ["GradeInput", "GradeState", "GradeResult", "build_grade_homework_graph"]


def build_grade_homework_graph(
    model: BaseChatModel,
    *,
    max_image_mb: int = 8,
    max_image_edge: int = 2048,
) -> CompiledStateGraph:
    """编译 START → fetch_image → grade_llm → calc_verify → END 执行图。

    参数:
        model: 支持 vision 输入的结构化批改模型。
        max_image_mb: 允许下载的最大图片体积(MB)。
        max_image_edge: 兜底压缩的长边上限(像素)。

    返回值:
        可由顶层编排图调用的已编译批改子图。
    """
    graph = StateGraph(GradeState)
    graph.add_node(
        "fetch_image",
        make_fetch_image_node(max_image_mb=max_image_mb, max_edge=max_image_edge),
    )
    graph.add_node("grade_llm", make_grade_llm_node(model))
    graph.add_node("calc_verify", make_calc_verify_node())
    graph.add_edge(START, "fetch_image")
    graph.add_edge("fetch_image", "grade_llm")
    graph.add_edge("grade_llm", "calc_verify")
    graph.add_edge("calc_verify", END)
    return graph.compile()

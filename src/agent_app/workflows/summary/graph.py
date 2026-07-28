"""固定结构化摘要工作流的 LangGraph 装配。"""

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent_app.workflows.summary.nodes import make_preprocess_node, make_summarize_node
from agent_app.workflows.summary.schemas import SummaryState


def build_summary_graph(model: BaseChatModel) -> CompiledStateGraph:
    """编译 START -> preprocess -> summarize -> END 执行图。

    参数:
        model: 生成结构化摘要的聊天模型。

    返回值:
        可由顶层编排图调用的已编译摘要子图。
    """
    graph = StateGraph(SummaryState)
    graph.add_node("preprocess", make_preprocess_node())
    graph.add_node("summarize", make_summarize_node(model))
    graph.add_edge(START, "preprocess")
    graph.add_edge("preprocess", "summarize")
    graph.add_edge("summarize", END)
    return graph.compile()

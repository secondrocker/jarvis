"""LangGraph assembly for the fixed structured summary workflow."""

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent_app.workflows.summary.nodes import make_preprocess_node, make_summarize_node
from agent_app.workflows.summary.schemas import SummaryState


def build_summary_graph(model: BaseChatModel) -> CompiledStateGraph:
    """Compile START -> preprocess -> summarize -> END."""
    graph = StateGraph(SummaryState)
    graph.add_node("preprocess", make_preprocess_node())
    graph.add_node("summarize", make_summarize_node(model))
    graph.add_edge(START, "preprocess")
    graph.add_edge("preprocess", "summarize")
    graph.add_edge("summarize", END)
    return graph.compile()

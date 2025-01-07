from typing import Dict, Any, Literal
from langgraph.graph import StateGraph, END
from handlers.query import (
    handle_general_query,
    handle_code_query,
    handle_visualization_query,
    handle_greeting,
)
from handlers.router import determine_next_step
from handlers.error import handle_error
from handlers.repository import analyze_repository_node
from models.state import RepoAnalysisState
from utils.config import configure_environment


def get_next_node(state: Dict) -> str:
    """Determine the next node in the conversation flow"""
    # Check for end condition
    if state.get("query", "").lower() in {"goodbye", "bye", "exit", "quit", "end"}:
        return END

    # Check for error
    if state.get("error"):
        return "error_handler"

    # Check if this is a response that needs to end
    if state.get("response_ready"):
        return END

    # Get next step from state
    return state.get("next_step", "general")


def build_graph() -> StateGraph:
    """Build the main processing graph for LangGraph Studio"""
    configure_environment()

    # Initialize graph
    workflow = StateGraph(RepoAnalysisState)

    # Add nodes
    workflow.add_node("determine_next", determine_next_step)
    workflow.add_node("greeting", handle_greeting)
    workflow.add_node("general", handle_general_query)
    workflow.add_node("code", handle_code_query)
    workflow.add_node("visualization", handle_visualization_query)
    workflow.add_node("analyze_repo", analyze_repository_node)
    workflow.add_node("error_handler", handle_error)

    # Set entry point
    workflow.set_entry_point("determine_next")

    # Add conditional edges from determine_next to handlers
    workflow.add_conditional_edges("determine_next", get_next_node)

    return workflow.compile()


# Export the compiled graph
graph = build_graph()

from typing import Dict, Any, Literal, Annotated
from langgraph.graph import StateGraph, END
from handlers.error import handle_error
from models.state import CombinedState
from utils.config import configure_environment
from agents.registry import registry


def get_next_node(state: Dict) -> str:
    """Determine the next node in the conversation flow"""
    # Check for response ready
    if state.get("response_ready"):
        return END

    # Check for error
    if state.get("error"):
        return "error_handler"

    # Check for interrupt that needs human input
    if state.get("__interrupt"):
        return "handle_sync_confirmation"

    # Check if we have a next agent assigned
    if state.get("next_agent"):
        return state["next_agent"]

    # Default to router
    return "router"


def build_graph() -> StateGraph:
    """Build the simplified processing graph"""
    configure_environment()

    # Initialize graph with combined state
    workflow = StateGraph(CombinedState)

    # Add nodes
    workflow.add_node("router", registry.get_handler_for_agent("router"))
    workflow.add_node("business_agent", registry.get_handler_for_agent("business_agent"))
    workflow.add_node("error_handler", handle_error)

    # Set entry point
    workflow.set_entry_point("router")

    # Add edges using the conditional router
    workflow.add_conditional_edges("router", get_next_node)
    workflow.add_conditional_edges("business_agent", get_next_node)
    workflow.add_conditional_edges("error_handler", get_next_node)

    return workflow.compile()


# Export the compiled graph
graph = build_graph()

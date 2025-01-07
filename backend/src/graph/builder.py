from typing import Dict, Any, Literal, Annotated
from langgraph.graph import StateGraph, END
from handlers.error import handle_error
from models.state import CombinedState
from utils.config import configure_environment
from agents.registry import registry


def get_next_node(state: Dict) -> str:
    """Determine the next node in the conversation flow"""
    # Check for interrupt that needs human input
    if state.get("__interrupt"):
        return END

    # Check for human response to handle
    if state.get("__human_response") is not None:
        return "handle_sync_confirmation"

    # Check for error
    if state.get("error"):
        return "error_handler"

    # Check if we have a next agent assigned
    if state.get("next_agent"):
        return state["next_agent"]

    # Check if response is ready
    if state.get("response_ready"):
        return END

    # Default to router for initial routing
    return "router"


def build_graph() -> StateGraph:
    """Build the main processing graph for LangGraph Studio"""
    configure_environment()

    # Initialize graph with combined state
    workflow = StateGraph(CombinedState)

    # Add router node
    workflow.add_node("router", registry.get_handler_for_agent("router"))

    # Add specialized agent nodes
    workflow.add_node("code_agent", registry.get_handler_for_agent("code_agent"))
    workflow.add_node("docs_agent", registry.get_handler_for_agent("docs_agent"))
    workflow.add_node(
        "business_agent", registry.get_handler_for_agent("business_agent")
    )

    # Add utility nodes
    workflow.add_node("error_handler", handle_error)
    workflow.add_node(
        "handle_sync_confirmation", registry.get_handler_for_agent("business_agent")
    )

    # Set entry point
    workflow.set_entry_point("router")

    # Add conditional edges
    workflow.add_conditional_edges("router", get_next_node)

    # Add edges from each agent
    for agent_id in ["code_agent", "docs_agent", "business_agent"]:
        workflow.add_conditional_edges(
            agent_id,
            lambda x, agent=agent_id: (
                "handle_sync_confirmation"
                if x.get("__interrupt") and agent == "business_agent"
                else (
                    END if x.get("response_ready") or x.get("__interrupt") else "router"
                )
            ),
        )

    # Add edge from error handler back to router
    workflow.add_conditional_edges(
        "error_handler", lambda x: "router" if not x.get("error") else END
    )

    # Add edge from sync confirmation
    workflow.add_conditional_edges("handle_sync_confirmation", lambda x: END)

    return workflow.compile()


# Export the compiled graph
graph = build_graph()

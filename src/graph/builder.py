from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.prebuilt.state_graph import END
from ..handlers.query import handle_general_query
from ..handlers.error import handle_error, recover_state_node
from ..handlers.validation import validate_input_node
from ..handlers.repository import analyze_repository_node
from ..models.state import RepoAnalysisState
from ..utils.config import configure_environment

def build_graph() -> StateGraph:
    """Build the main processing graph for LangGraph Studio"""
    # Configure environment
    configure_environment()
    
    # Initialize graph with proper state type
    workflow = StateGraph(RepoAnalysisState)

    # Add nodes
    workflow.add_node("validate_input", validate_input_node)
    workflow.add_node("analyze_repository", analyze_repository_node)
    workflow.add_node("handle_query", handle_general_query)
    workflow.add_node("error_handler", handle_error)
    workflow.add_node("recover_state", recover_state_node)

    # Define the flow
    workflow.set_entry_point("validate_input")
    
    # Add edges
    workflow.add_edge("validate_input", "analyze_repository")
    workflow.add_edge("analyze_repository", "handle_query")
    workflow.add_edge("handle_query", END)
    
    # Add error handling
    workflow.add_edge("validate_input", "error_handler", condition=lambda x: x.get("error"))
    workflow.add_edge("analyze_repository", "error_handler", condition=lambda x: x.get("error"))
    workflow.add_edge("error_handler", "recover_state")
    workflow.add_edge("recover_state", "validate_input")

    return workflow.compile()

# Export the compiled graph for LangGraph Studio
graph = build_graph() 
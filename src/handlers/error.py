import datetime
from typing import Dict, Any, List
from langchain_core.messages import AIMessage
from models.state import CombinedState


def handle_error(state: CombinedState) -> Dict[str, Any]:
    """Handle errors and provide recovery options"""
    try:
        if state.get("error"):
            error_msg = state["error"]
            recovery_options = determine_recovery_options(error_msg)

            return {
                **state,
                "messages": state["messages"]
                + [
                    AIMessage(
                        content=f"An error occurred: {error_msg}\n\nI can try to:"
                    ),
                    *[AIMessage(content=f"- {option}") for option in recovery_options],
                ],
                "pending_operations": recovery_options,
                "error": None,
            }
        return state
    except Exception as e:
        return {
            **state,
            "messages": state["messages"]
            + [AIMessage(content=f"Critical error in error handler: {str(e)}")],
            "error": str(e),
        }


def recover_state_node(state: CombinedState) -> Dict[str, Any]:
    """Recover from errors and restore state"""
    try:
        if state.get("repo_context"):
            return {
                **state,
                "analysis_stage": "recovered",
                "messages": state["messages"]
                + [
                    AIMessage(content="Successfully recovered previous analysis state.")
                ],
            }
        return {
            **state,
            "messages": [],
            "context": [],
            "files": [],
            "error": None,
            "response_ready": False,
            "next_step": "determine_next",
            "repo_url": None,
            "repo_context": None,
            "query": None,
            "query_type": None,
            "response": None,
            "analysis_complete": False,
            "input_type": None,
            "raw_input": None,
            "requirements": None,
            "user_stories": None,
            "jira_sync_status": None,
            "__interrupt": None,
            "__human_response": None,
        }
    except Exception as e:
        return {**state, "error": f"Recovery failed: {str(e)}"}


def determine_recovery_options(error_msg: str) -> List[str]:
    """Determine possible recovery options based on error message"""
    recovery_options = []

    error_patterns = {
        "token": ["Refresh GitHub token", "Use alternative authentication"],
        "rate limit": ["Wait and retry", "Use different token"],
        "not found": ["Verify repository URL", "Check repository visibility"],
        "permission": ["Request repository access", "Use public repository"],
        "timeout": ["Retry with longer timeout", "Analyze smaller portion"],
        "memory": ["Reduce analysis scope", "Process in batches"],
    }

    for pattern, options in error_patterns.items():
        if pattern.lower() in error_msg.lower():
            recovery_options.extend(options)

    if not recovery_options:
        recovery_options = ["Restart analysis", "Try different repository"]

    return recovery_options

import datetime
from typing import Dict, Any, List
from langchain_core.messages import AIMessage
from ..models.state import ChatFlowState

def handle_error(state: ChatFlowState) -> Dict[str, Any]:
    """Handle errors and provide recovery options"""
    try:
        if state.get("error"):
            error_msg = state["error"]
            recovery_options = determine_recovery_options(error_msg)
            
            return {
                **state,
                "messages": state["messages"] + [
                    AIMessage(content=f"An error occurred: {error_msg}\n\nI can try to:"),
                    *[AIMessage(content=f"- {option}") for option in recovery_options]
                ],
                "pending_operations": recovery_options,
                "error": None,
            }
        return state
    except Exception as e:
        return {
            **state,
            "messages": state["messages"] + [
                AIMessage(content=f"Critical error in error handler: {str(e)}")
            ],
            "error": str(e),
        }

def recover_state_node(state: ChatFlowState) -> Dict[str, Any]:
    """Recover from errors and restore state"""
    try:
        if state.get("repo_context"):
            return {
                **state,
                "analysis_stage": "recovered",
                "messages": state["messages"] + [
                    AIMessage(content="Successfully recovered previous analysis state.")
                ],
            }
        return {
            "conversation_id": str(datetime.datetime.now().timestamp()),
            "messages": [],
            "current_context": {},
            "repo_context": None,
            "source_type": "",
            "source_path": "",
            "analysis_type": "chat",
            "generated_artifacts": {},
            "waiting_for_repo": False,
            "last_query": "",
            "analysis_stage": "initial",
            "pending_operations": [],
            "analysis_depth": 0,
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
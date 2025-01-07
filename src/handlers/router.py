from typing import Dict, Any


def determine_next_step(state: Dict) -> Dict[str, Any]:
    """Determine the next step in conversation flow"""
    try:
        # Reset response_ready flag when starting new routing
        state = {**state, "response_ready": False}

        query = state.get("query", "").lower()
        repo_context = state.get("repo_context", {})

        # Determine next step
        next_step = get_next_step(query, repo_context)

        return {**state, "next_step": next_step}
    except Exception as e:
        return {**state, "error": f"Routing failed: {str(e)}"}


def get_next_step(query: str, repo_context: Dict) -> str:
    """Get next step based on query and context"""
    # Handle greetings
    if any(greeting in query for greeting in ["hi", "hello", "hey"]):
        return "greeting"

    # Handle GitHub URLs
    if "github.com" in query and not repo_context:
        return "analyze_repo"

    # Route based on query intent
    if "generate" in query and "code" in query:
        return "code"
    elif any(term in query for term in ["diagram", "visualize", "show", "draw"]):
        return "visualization"
    elif any(term in query for term in ["explain", "analyze", "how", "what"]):
        return "code"

    # Default to general handler
    return "general"

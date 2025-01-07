from typing import Dict, Any, Union, Tuple
from ..models.state import RepoAnalysisState
from ..utils.helpers import extract_repo_url

def validate_input_node(state: RepoAnalysisState) -> Dict[str, Any]:
    """Validate input and prepare state for analysis"""
    try:
        # Validate repo URL
        if not state.get("repo_url"):
            repo_url = extract_repo_url(state.get("query", ""))
            if not repo_url:
                return {
                    **state,
                    "error": "No repository URL provided or found in query",
                }
            state["repo_url"] = repo_url

        # Validate query
        if not state.get("query"):
            return {
                **state,
                "error": "No query provided",
            }

        return state

    except Exception as e:
        return {
            **state,
            "error": f"Input validation failed: {str(e)}",
        } 
from typing import Annotated, List, Optional, Dict, Any, Union
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from .schema import RepoFile


class BaseState(TypedDict):
    """Base state class with common fields"""

    messages: Annotated[List[Any], "append"]
    context: Annotated[List[str], "append"]
    files: Annotated[List[RepoFile], "merge"]


class CombinedState(BaseState):
    """Combined state for all operations"""

    # Common fields
    error: Optional[str]
    response_ready: bool
    next_step: Optional[str]
    next_agent: Optional[str]  # Added for agent routing

    # Router fields
    current_agent: Optional[str]
    agent_history: Annotated[List[str], "append"]

    # Repository analysis fields
    repo_url: Optional[str]
    repo_context: Optional[Dict[str, Any]]
    query: Optional[str]
    query_type: Optional[str]
    response: Optional[str]
    analysis_complete: Optional[bool]

    # Business analysis fields
    input_type: Optional[str]
    raw_input: Optional[Union[str, bytes]]
    requirements: Optional[List[Dict[str, Any]]]
    user_stories: Optional[List[Dict[str, Any]]]
    jira_sync_status: Optional[Dict[str, Any]]

    # Documentation fields
    generated_artifacts: Optional[Dict[str, Any]]
    documentation_type: Optional[str]

    # Code analysis fields
    code_context: Optional[str]
    code_files: Optional[List[Dict[str, Any]]]

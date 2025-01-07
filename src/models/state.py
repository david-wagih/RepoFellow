from typing import Annotated, List, Optional, Dict, Any
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from .schema import RepoFile, Analyst

class BaseState(TypedDict):
    """Base state class with common fields"""
    messages: Annotated[List[Any], "append"]
    context: Annotated[List[str], "append"]
    files: Annotated[List[RepoFile], "merge"]

class RepoAnalysisState(BaseState):
    """State for repository analysis"""
    repo_url: str
    repo_context: Dict[str, Any]
    query: str
    query_type: str
    response: str
    analysis_complete: bool
    error: Optional[str] = None

class ChatFlowState(TypedDict):
    """Enhanced state management for chat flow"""
    conversation_id: str
    messages: Annotated[List[BaseMessage], "append"]
    current_context: Dict[str, Any]
    repo_context: Optional[Dict[str, Any]]
    source_type: str
    source_path: str
    analysis_type: str
    generated_artifacts: Dict[str, Any]
    waiting_for_repo: bool
    last_query: str
    error: Optional[str]
    analysis_stage: str
    pending_operations: List[str]
    analysis_depth: int 
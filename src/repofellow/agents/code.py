"""Code analysis and modification agent"""
from typing import Dict, Any, List
from langchain_core.messages import HumanMessage, AIMessage

from repofellow.config.prompts import get_prompt
from .base import BaseAgent

class CodeAgent(BaseAgent):
    """Agent specialized in code understanding and generation"""
    
    def __init__(self):
        super().__init__("code_agent")
    
    def can_handle(self, state: Dict[str, Any]) -> bool:
        query = state.get("query", "").lower()
        return any(
            capability in self.config.capabilities
            for capability in ["code_analysis", "code_generation", "debugging", "refactoring"]
        ) and any(
            term in query
            for term in [
                "code",
                "function",
                "class",
                "implement",
                "bug",
                "error",
                "how to",
                "example",
                "syntax",
                "pattern",
            ]
        )

    def _prepare_messages(self, state: Dict[str, Any]) -> List[HumanMessage]:
        query = state.get("query", "")
        context = state.get("code_context", "")
        
        # Use task-specific prompts if available
        if "review" in query.lower():
            prompt = get_prompt("code_review", code=context)
        elif "dependency" in query.lower():
            prompt = get_prompt("dependency_analysis", dependencies=context)
        elif "architecture" in query.lower():
            prompt = get_prompt("architecture_review", context=context)
        else:
            prompt = f"Context:\n{context}\n\nQuery: {query}"
            
        return [HumanMessage(content=prompt)]

    def _process_response(self, state: Dict[str, Any], response: str) -> Dict[str, Any]:
        return {
            **state,
            "response": response,
            "messages": state.get("messages", []) + [AIMessage(content=response)],
            "response_ready": True,
            "capabilities_used": self.config.capabilities,
        } 
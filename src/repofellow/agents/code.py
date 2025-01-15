"""Code analysis and modification agent"""
from typing import Dict, Any, List
from langchain_core.messages import HumanMessage, AIMessage
from .base import BaseAgent

class CodeAgent(BaseAgent):
    """Agent specialized in code understanding and generation"""

    def get_system_prompt(self) -> str:
        return """You are an expert software developer with deep understanding of code.
        You can:
        1. Analyze and explain code
        2. Generate code based on requirements
        3. Suggest improvements and best practices
        4. Debug issues and propose solutions
        Always provide clear explanations and well-structured code."""

    def can_handle(self, state: Dict[str, Any]) -> bool:
        query = state.get("query", "").lower()
        return any(
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
        return [HumanMessage(content=f"Context:\n{context}\n\nQuery: {query}")]

    def _process_response(self, state: Dict[str, Any], response: str) -> Dict[str, Any]:
        return {
            **state,
            "response": response,
            "messages": state.get("messages", []) + [AIMessage(content=response)],
            "response_ready": True,
        } 
from typing import Dict, Any, List
from langchain_core.messages import HumanMessage, AIMessage
from .base import BaseAgent
from tools.visualization import RepoVisualizer


class DocumentationAgent(BaseAgent):
    """Agent specialized in documentation and diagrams"""

    def get_system_prompt(self) -> str:
        return """You are a technical documentation specialist.
        You can:
        1. Generate clear technical documentation
        2. Create architecture diagrams
        3. Write user guides and API documentation
        4. Explain complex systems in simple terms
        Focus on clarity, completeness, and proper structure."""

    def can_handle(self, state: Dict[str, Any]) -> bool:
        query = state.get("query", "").lower()
        return any(
            term in query
            for term in [
                "document",
                "diagram",
                "architecture",
                "explain",
                "visualize",
                "flow",
                "structure",
                "guide",
            ]
        )

    def _prepare_messages(self, state: Dict[str, Any]) -> List[HumanMessage]:
        query = state.get("query", "")
        context = state.get("repo_context", {})
        return [HumanMessage(content=f"Context:\n{str(context)}\n\nQuery: {query}")]

    def _process_response(self, state: Dict[str, Any], response: str) -> Dict[str, Any]:
        if "diagram" in state.get("query", "").lower():
            diagram = RepoVisualizer.generate_architecture_diagram(
                state.get("files", [])
            )
            return {
                **state,
                "response": response,
                "messages": state.get("messages", [])
                + [
                    AIMessage(content=response),
                    AIMessage(content="Here's the generated diagram:"),
                ],
                "generated_artifacts": {"diagram": diagram},
                "response_ready": True,
            }

        return {
            **state,
            "response": response,
            "messages": state.get("messages", []) + [AIMessage(content=response)],
            "response_ready": True,
        }

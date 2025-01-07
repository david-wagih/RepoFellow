from typing import Dict, Any, List
from langchain_core.messages import HumanMessage
from .base import BaseAgent


class RouterAgent(BaseAgent):
    """Agent responsible for routing queries to specialized agents"""

    def get_system_prompt(self) -> str:
        return """You are a routing agent that determines which specialized agent should handle a query.
        Available agents:
        - CodeAgent: For code understanding, generation, and technical questions
        - DocumentationAgent: For creating technical/non-technical docs and diagrams
        - BusinessAgent: For requirements analysis and user story generation
        
        Analyze the query and determine the most appropriate agent."""

    def can_handle(self, state: Dict[str, Any]) -> bool:
        return True  # Router can handle all initial queries

    def _prepare_messages(self, state: Dict[str, Any]) -> List[HumanMessage]:
        query = state.get("query", "")
        return [HumanMessage(content=f"Route this query: {query}")]

    def _process_response(self, state: Dict[str, Any], response: str) -> Dict[str, Any]:
        agent_map = {
            "code": "code_agent",
            "documentation": "docs_agent",
            "business": "business_agent",
        }

        for key, agent in agent_map.items():
            if key in response.lower():
                return {**state, "next_agent": agent}

        return {**state, "next_agent": "code_agent"}  # Default to code agent

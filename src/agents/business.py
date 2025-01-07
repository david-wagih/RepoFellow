from typing import Dict, Any, List
from langchain_core.messages import HumanMessage, AIMessage
from .base import BaseAgent
from tools.business_analysis import BusinessAnalysisTools
from utils.jira_client import JiraClient


class BusinessAgent(BaseAgent):
    """Agent specialized in business analysis and user stories"""

    def get_system_prompt(self) -> str:
        return """You are a skilled business analyst.
        You can:
        1. Analyze business requirements
        2. Generate user stories
        3. Analyze UI/UX designs
        4. Sync with project management tools
        Focus on delivering clear, actionable items."""

    def can_handle(self, state: Dict[str, Any]) -> bool:
        input_type = state.get("input_type")
        return input_type in ["text", "image"] and state.get("raw_input")

    def _prepare_messages(self, state: Dict[str, Any]) -> List[HumanMessage]:
        if state.get("input_type") == "image":
            return [HumanMessage(content="Analyze the design and create user stories")]
        return [HumanMessage(content=f"Requirements:\n{state.get('raw_input')}")]

    def _process_response(self, state: Dict[str, Any], response: str) -> Dict[str, Any]:
        try:
            if state.get("input_type") == "image":
                user_stories = BusinessAnalysisTools.analyze_design(
                    state.get("raw_input")
                )
            else:
                user_stories = BusinessAnalysisTools.analyze_requirements(
                    state.get("raw_input")
                )

            return {
                **state,
                "user_stories": user_stories,
                "messages": state.get("messages", [])
                + [
                    AIMessage(
                        content=BusinessAnalysisTools.format_user_stories(user_stories)
                    )
                ],
                "__interrupt": {
                    "prompt": "Would you like to sync these stories to Jira? (yes/no)",
                    "type": "confirmation",
                    "data": {"stories": user_stories},
                },
            }
        except Exception as e:
            return {**state, "error": f"Business analysis failed: {str(e)}"}

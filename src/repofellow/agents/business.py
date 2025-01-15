"""Business analysis and requirements agent"""
from typing import Dict, Any, List
from langchain_core.messages import HumanMessage, AIMessage
from .base import BaseAgent
from ..tools.business_analysis import BusinessAnalysisTools

class BusinessAgent(BaseAgent):
    """Agent specialized in business analysis and requirements"""
    
    def __init__(self):
        super().__init__("business_agent")
        self.analysis_tools = BusinessAnalysisTools()
    
    def can_handle(self, state: Dict[str, Any]) -> bool:
        query = state.get("query", "").lower()
        return any(
            capability in self.config.capabilities
            for capability in ["requirement_analysis", "user_story_generation"]
        ) and any(
            term in query
            for term in [
                "requirement",
                "user story",
                "business",
                "feature",
                "epic",
                "sprint",
                "stakeholder",
            ]
        )

    def _prepare_messages(self, state: Dict[str, Any]) -> List[HumanMessage]:
        requirements = state.get("raw_input", "")
        input_type = state.get("input_type", "text")
        
        if input_type == "image":
            context = self.analysis_tools.analyze_design(requirements)
        else:
            context = requirements
            
        return [HumanMessage(content=f"Analyze the following requirements:\n{context}")]

    def _process_response(self, state: Dict[str, Any], response: str) -> Dict[str, Any]:
        try:
            user_stories = self.analysis_tools.parse_user_stories(response)
            formatted_stories = self.analysis_tools.format_user_stories(user_stories)
            
            return {
                **state,
                "response": formatted_stories,
                "user_stories": user_stories,
                "messages": state.get("messages", []) + [AIMessage(content=formatted_stories)],
                "response_ready": True,
                "capabilities_used": self.config.capabilities,
            }
        except Exception as e:
            return {**state, "error": f"Failed to process business analysis: {str(e)}"} 
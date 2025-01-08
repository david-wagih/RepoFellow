from typing import Dict, Any, List
from langchain_core.messages import HumanMessage, AIMessage
from .base import BaseAgent


class RouterAgent(BaseAgent):
    """Agent responsible for routing queries between conversation and business analysis"""

    def get_system_prompt(self) -> str:
        return """You are an intelligent routing agent that handles conversations and identifies business analysis needs.
        
        You should:
        1. Handle general conversation naturally and engagingly
        2. Identify when users need help with requirements analysis
        3. Route to business analysis when users:
           - Mention requirements, user stories, or business needs
           - Share business notes or specifications
           - Ask for functional/non-functional requirements
           - Need help organizing business ideas
        
        Keep conversation natural and helpful while being attentive to business analysis needs."""

    def can_handle(self, state: Dict[str, Any]) -> bool:
        return True  # Router handles all initial queries

    def _prepare_messages(self, state: Dict[str, Any]) -> List[HumanMessage]:
        query = state.get("query", "")
        return [HumanMessage(content=f"Process this query and determine if it needs business analysis: {query}")]

    def _process_response(self, state: Dict[str, Any], response: str) -> Dict[str, Any]:
        query = state.get("query", "").lower()
        
        # Check for business analysis needs
        business_keywords = [
            "requirement", "requirements", "user stor", "business need", 
            "functional", "non-functional", "specification", "feature",
            "system should", "must have", "needs to", "analyze this"
        ]
        
        if any(keyword in query for keyword in business_keywords):
            return {
                **state,
                "next_agent": "business_agent",
                "raw_input": state.get("query"),
                "input_type": "text"
            }
        
        # Handle as general conversation
        return {
            **state,
            "response": self._generate_conversational_response(query),
            "messages": state.get("messages", []) + [
                AIMessage(content=self._generate_conversational_response(query))
            ],
            "response_ready": True
        }

    def _generate_conversational_response(self, query: str) -> str:
        """Generate natural conversational responses"""
        if any(greeting in query for greeting in ["hi", "hello", "hey", "how are you"]):
            return "Hello! I'm here to help you with requirements analysis and general questions. Feel free to share any business notes or requirements you'd like me to analyze!"
        
        if "help" in query or "what can you do" in query:
            return ("I can help you with:\n"
                   "1. Analyzing business notes and generating requirements\n"
                   "2. Creating functional and non-functional requirements\n"
                   "3. Organizing business ideas into structured requirements\n"
                   "Just share your business notes or ideas, and I'll help break them down!")
        
        return ("I'm not sure what you need help with. I'm best at analyzing business notes "
                "and generating requirements. Would you like to share some business notes or "
                "requirements for me to analyze?")

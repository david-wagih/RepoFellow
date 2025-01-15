"""Base agent class definition"""
from typing import Dict, Any, List
from abc import ABC, abstractmethod
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from ..utils.config import configure_environment

class BaseAgent(ABC):
    """Base class for all agents"""
    
    def __init__(self):
        configure_environment()  # Ensure environment is configured
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
        )
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent"""
        pass
    
    @abstractmethod
    def can_handle(self, state: Dict[str, Any]) -> bool:
        """Check if this agent can handle the given state"""
        pass
    
    @abstractmethod
    def _prepare_messages(self, state: Dict[str, Any]) -> List[HumanMessage]:
        """Prepare messages for the agent"""
        pass
    
    @abstractmethod
    def _process_response(self, state: Dict[str, Any], response: str) -> Dict[str, Any]:
        """Process the agent's response"""
        pass
    
    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke the agent with the given state"""
        if not self.can_handle(state):
            return {
                **state,
                "error": f"Agent {self.__class__.__name__} cannot handle this request"
            }
        
        try:
            # Prepare messages including system prompt
            messages = [
                SystemMessage(content=self.get_system_prompt()),
                *self._prepare_messages(state)
            ]
            
            # Call LLM
            response = self.llm.invoke(messages).content
            
            # Process response
            return self._process_response(state, response)
        except Exception as e:
            return {
                **state,
                "error": f"Error in {self.__class__.__name__}: {str(e)}"
            } 
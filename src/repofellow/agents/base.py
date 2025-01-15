"""Base agent class with enhanced configuration support"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from langchain_core.messages import HumanMessage, SystemMessage
from ..config.config_manager import config_manager
from ..config.prompts import get_prompt
from ..utils.model_factory import ModelFactory

class BaseAgent(ABC):
    """Enhanced base class for all agents"""
    
    def __init__(self, agent_type: str):
        self.agent_type = agent_type
        self.config = config_manager.get_agent_config(agent_type)
        self.llm = ModelFactory.create_model_with_fallback(
            self.config.primary_model,
            self.config.fallback_model
        )
        self.system_prompt = get_prompt(agent_type)
    
    def refresh_config(self) -> None:
        """Refresh agent configuration and model"""
        self.config = config_manager.get_agent_config(self.agent_type)
        self.llm = ModelFactory.create_model_with_fallback(
            self.config.primary_model,
            self.config.fallback_model
        )
    
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
                "error": f"Agent {self.config.name} cannot handle this request"
            }
        
        try:
            messages = [
                SystemMessage(content=self.system_prompt),
                *self._prepare_messages(state)
            ]
            
            response = self.llm.invoke(messages).content
            return self._process_response(state, response)
        except Exception as e:
            return {
                **state,
                "error": f"Error in {self.config.name}: {str(e)}"
            } 
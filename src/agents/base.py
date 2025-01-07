from typing import Dict, Any, List
from abc import ABC, abstractmethod
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from utils.config import llm


class BaseAgent(ABC):
    """Base class for specialized agents"""

    def __init__(self):
        self.system_prompt = self.get_system_prompt()

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the agent's system prompt"""
        pass

    @abstractmethod
    def can_handle(self, state: Dict[str, Any]) -> bool:
        """Determine if this agent can handle the current state"""
        pass

    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process the current state and return updated state"""
        try:
            messages = [
                SystemMessage(content=self.system_prompt),
                *self._prepare_messages(state),
            ]

            response = llm.invoke(messages)
            return self._process_response(state, response.content)
        except Exception as e:
            return {**state, "error": f"{self.__class__.__name__} failed: {str(e)}"}

    @abstractmethod
    def _prepare_messages(self, state: Dict[str, Any]) -> List[HumanMessage]:
        """Prepare messages for the LLM based on state"""
        pass

    @abstractmethod
    def _process_response(self, state: Dict[str, Any], response: str) -> Dict[str, Any]:
        """Process LLM response and update state"""
        pass

from typing import Dict, Type
from .base import BaseAgent
from .router import RouterAgent
from .code import CodeAgent
from .documentation import DocumentationAgent
from .business import BusinessAgent


class AgentRegistry:
    """Registry for managing available agents"""

    def __init__(self):
        self._agents: Dict[str, BaseAgent] = {
            "router": RouterAgent(),
            "business_agent": BusinessAgent(),
        }

    def get_agent(self, agent_id: str) -> BaseAgent:
        """Get agent instance by ID"""
        if agent_id not in self._agents:
            raise ValueError(f"Unknown agent: {agent_id}")
        return self._agents[agent_id]

    def get_handler_for_agent(self, agent_id: str):
        """Get the handler function for an agent"""
        agent = self.get_agent(agent_id)
        return lambda state: agent.invoke(state)


# Global registry instance
registry = AgentRegistry()

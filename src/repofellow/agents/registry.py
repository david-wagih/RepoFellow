"""Agent registry for managing available agents"""
from typing import Dict, Type
from .base import BaseAgent
from .code import CodeAgent
from .business import BusinessAgent
from .cli_router import CLIRouterAgent

class AgentRegistry:
    """Registry for managing available agents"""

    def __init__(self):
        self._agents: Dict[str, BaseAgent] = {
            "router": CLIRouterAgent(),
            "code_agent": CodeAgent(),
            "business_agent": BusinessAgent(),
        }

    def get_agent(self, agent_id: str) -> BaseAgent:
        """Get agent instance by ID"""
        if agent_id not in self._agents:
            raise ValueError(f"Unknown agent: {agent_id}")
        return self._agents[agent_id]

    def register_agent(self, agent_id: str, agent: BaseAgent) -> None:
        """Register a new agent"""
        self._agents[agent_id] = agent

# Global registry instance
registry = AgentRegistry() 
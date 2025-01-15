"""CLI command routing agent"""
from typing import Dict, Any, List
from langchain_core.messages import HumanMessage, AIMessage
from .base import BaseAgent
from ..config.cli_mappings import CLI_COMMAND_MAPPINGS

class CLIRouterAgent(BaseAgent):
    """Agent specialized in routing CLI commands to appropriate handlers"""
    
    def __init__(self):
        super().__init__("router_agent")
    
    def can_handle(self, state: Dict[str, Any]) -> bool:
        command_type = state.get("query_type", "")
        return command_type in CLI_COMMAND_MAPPINGS and any(
            capability in self.config.capabilities
            for capability in ["command_parsing", "intent_analysis"]
        )
    
    def _prepare_messages(self, state: Dict[str, Any]) -> List[HumanMessage]:
        command = state.get("query_type", "")
        mapping = CLI_COMMAND_MAPPINGS.get(command, {})
        
        try:
            content = mapping["prompt_template"].format(**state)
        except KeyError:
            content = state.get("query", "No query provided")
            
        return [HumanMessage(content=content)]

    def _process_response(self, state: Dict[str, Any], response: str) -> Dict[str, Any]:
        command = state.get("query_type", "")
        mapping = CLI_COMMAND_MAPPINGS.get(command, {})
        
        return {
            **state,
            "next_agent": mapping.get("agent", "code_agent"),
            "messages": state.get("messages", []) + [AIMessage(content=response)],
            "capabilities": mapping.get("capabilities", []),
            "response": response,
            "capabilities_used": self.config.capabilities,
        } 
"""CLI Router Agent implementation"""
from typing import Dict, Any, List
from langchain_core.messages import HumanMessage, AIMessage
from .base import BaseAgent
from ..config.cli_mappings import CLI_COMMAND_MAPPINGS

class CLIRouterAgent(BaseAgent):
    """Agent specialized in routing CLI commands to appropriate handlers"""
    
    def get_system_prompt(self) -> str:
        return """You are an expert CLI command router.
        You analyze CLI commands and determine the best way to handle them.
        You can route commands to:
        1. Code analysis and modification
        2. Documentation generation
        3. Business requirements analysis
        4. Repository visualization
        Always choose the most appropriate handler based on command intent."""
    
    def can_handle(self, state: Dict[str, Any]) -> bool:
        command_type = state.get("query_type", "")
        return command_type in CLI_COMMAND_MAPPINGS
    
    def _prepare_messages(self, state: Dict[str, Any]) -> List[HumanMessage]:
        command = state.get("query_type", "")
        mapping = CLI_COMMAND_MAPPINGS.get(command, {})
        
        # Format the template with state values, handling missing keys gracefully
        try:
            content = mapping["prompt_template"].format(**state)
        except KeyError:
            # If formatting fails, use the query directly
            content = state.get("query", "No query provided")
            
        return [HumanMessage(content=content)]
    
    def _process_response(self, state: Dict[str, Any], response: str) -> Dict[str, Any]:
        command = state.get("query_type", "")
        mapping = CLI_COMMAND_MAPPINGS.get(command, {})
        next_agent = mapping.get("agent", "code_agent")
        
        return {
            **state,
            "next_agent": next_agent,
            "messages": state.get("messages", []) + [AIMessage(content=response)],
            "capabilities": mapping.get("capabilities", []),
            "response": response  # Add the response to the state
        } 
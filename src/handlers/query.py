from typing import Dict, Any, Optional
import re
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from utils.config import llm
from models.state import RepoAnalysisState, ChatFlowState

def handle_general_query(state: RepoAnalysisState) -> Dict:
    """Enhanced general query handler with web search integration"""
    try:
        query = state.get("query", "").lower()

        # Check for greetings
        greeting_response = _check_greetings(query)
        if greeting_response:
            return {
                **state,
                "response": greeting_response,
                "messages": state.get("messages", []) + [AIMessage(content=greeting_response)],
            }

        # Handle repository context
        if state.get("repo_context"):
            return _handle_repo_context_query(state, query)

        # Default response
        return _generate_default_response(state, query)

    except Exception as e:
        return {
            **state,
            "response": f"Error processing query: {str(e)}",
            "error": str(e),
        }

def _handle_repo_context_query(state: RepoAnalysisState, query: str) -> Dict[str, Any]:
    """Handle queries with repository context"""
    try:
        # Prepare context for LLM
        context = f"""Repository: {state['repo_context']['metadata']['name']}
Description: {state['repo_context']['metadata']['description']}
Files: {len(state['repo_context']['files'])}
Query: {query}"""

        # Generate response using LLM
        messages = [
            SystemMessage(content="You are a helpful repository analysis assistant."),
            HumanMessage(content=context)
        ]
        
        response = llm.invoke(messages).content

        return {
            **state,
            "response": response,
            "messages": state.get("messages", []) + [AIMessage(content=response)],
        }

    except Exception as e:
        return {**state, "error": f"Error handling repo context query: {str(e)}"}

def _generate_default_response(state: RepoAnalysisState, query: str) -> Dict[str, Any]:
    """Generate default response when no specific context is available"""
    response = "I can help you analyze GitHub repositories. Please provide a repository URL to get started."
    return {
        **state,
        "response": response,
        "messages": state.get("messages", []) + [AIMessage(content=response)],
    }

def _check_greetings(query: str) -> Optional[str]:
    greeting_patterns = {
        r"^hi\b": "Hi! I'm your GitHub repository assistant. How can I help you today?",
        r"^hello\b": "Hello! I'm here to help you analyze GitHub repositories. What would you like to know?",
        r"^hey\b": "Hey there! Need help with a GitHub repository?",
        # ... other patterns
    }
    
    for pattern, response in greeting_patterns.items():
        if re.match(pattern, query):
            return response
    return None 
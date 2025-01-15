from typing import Dict, Any
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from utils.config import llm
from utils.helpers import extract_repo_url


def handle_greeting(state: Dict) -> Dict[str, Any]:
    """Handle greeting messages"""
    try:
        return {
            **state,
            "response": "Hello! I can help you analyze GitHub repositories and answer questions about code. What would you like to know?",
            "messages": state.get("messages", [])
            + [
                AIMessage(
                    content="Hello! I can help you analyze GitHub repositories and answer questions about code. What would you like to know?"
                )
            ],
            "query": "",  # Clear query to prevent recursion
            "response_ready": True,  # Mark that we have a response ready
        }
    except Exception as e:
        return {**state, "error": f"Greeting failed: {str(e)}"}


def handle_general_query(state: Dict) -> Dict[str, Any]:
    """Handle general repository queries"""
    try:
        query = state.get("query", "")
        repo_context = state.get("repo_context", {})

        if not repo_context:
            return {
                **state,
                "response": "Please share a GitHub repository URL to get started.",
                "messages": state.get("messages", [])
                + [
                    AIMessage(
                        content="Please share a GitHub repository URL to get started."
                    )
                ],
            }

        context = prepare_repo_context(repo_context)
        messages = [
            SystemMessage(
                content="You are a repository analysis assistant. Provide clear and concise information."
            ),
            HumanMessage(content=f"{context}\n\nQuery: {query}"),
        ]

        response = llm.invoke(messages).content
        return {
            **state,
            "response": response,
            "messages": state.get("messages", []) + [AIMessage(content=response)],
        }
    except Exception as e:
        return {**state, "error": f"Query handling failed: {str(e)}"}


def handle_code_query(state: Dict) -> Dict[str, Any]:
    """Handle code-related queries"""
    try:
        query = state.get("query", "")
        repo_context = state.get("repo_context", {})

        context = prepare_code_context(repo_context)
        messages = [
            SystemMessage(
                content="You are a code expert. Explain implementation details or generate code as needed."
            ),
            HumanMessage(content=f"{context}\n\nQuery: {query}"),
        ]

        response = llm.invoke(messages).content
        return {
            **state,
            "response": response,
            "messages": state.get("messages", []) + [AIMessage(content=response)],
        }
    except Exception as e:
        return {**state, "error": f"Code query failed: {str(e)}"}


def handle_visualization_query(state: Dict) -> Dict[str, Any]:
    """Handle visualization requests"""
    try:
        from tools.visualization import RepoVisualizer

        diagram = RepoVisualizer.generate_architecture_diagram(state["files"])
        return {
            **state,
            "response": "Generated repository diagram",
            "messages": state.get("messages", [])
            + [AIMessage(content="Here's the repository diagram:")],
            "generated_artifacts": {"diagram": diagram},
        }
    except Exception as e:
        return {**state, "error": f"Visualization failed: {str(e)}"}


def prepare_repo_context(repo_context: Dict) -> str:
    """Prepare repository context for queries"""
    return f"""
Repository Information:
- Name: {repo_context.get('metadata', {}).get('name', 'N/A')}
- Description: {repo_context.get('metadata', {}).get('description', 'N/A')}
- Files: {len(repo_context.get('files', []))}
- Stars: {repo_context.get('metadata', {}).get('stars', 0)}
- Forks: {repo_context.get('metadata', {}).get('forks', 0)}
"""


def prepare_code_context(repo_context: Dict) -> str:
    """Prepare code-specific context"""
    return f"""
{prepare_repo_context(repo_context)}

Code Structure:
{repo_context.get('structure', {}).get('metrics', {})}

Dependencies:
{repo_context.get('dependencies', {})}
"""

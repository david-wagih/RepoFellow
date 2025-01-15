"""Configuration utilities"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

def configure_environment():
    """Configure environment variables"""
    load_dotenv()
    
    required_vars = {
        "OPENAI_API_KEY": "OpenAI API key for language model",
        "LANGSMITH_API_KEY": "LangSmith API key for tracing",
        "GITHUB_TOKEN": "GitHub token for repository access",
    }
    
    missing_vars = {
        var: desc for var, desc in required_vars.items() 
        if not os.getenv(var)
    }
    
    if missing_vars:
        raise EnvironmentError(
            "Missing required environment variables:\n"
            + "\n".join(f"- {var}: {desc}" for var, desc in missing_vars.items())
        )

# Default LLM instance for utilities
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
)

# Configure environment on import
configure_environment() 
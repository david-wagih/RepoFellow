import os
import certifi
from langchain_openai import ChatOpenAI

def configure_environment():
    """Configure environment variables and SSL certificates"""
    os.environ["SSL_CERT_FILE"] = certifi.where()
    os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

    required_vars = {
        "OPENAI_API_KEY": "OpenAI API key for language model",
        "GITHUB_TOKEN": "GitHub token for repository access",
        "TAVILY_API_KEY": "Tavily API key for web search",
        "LANGCHAIN_API_KEY": "LangChain API key for tracing",
    }

    missing_vars = {
        var: desc for var, desc in required_vars.items() if not os.getenv(var)
    }

    if missing_vars:
        raise EnvironmentError(
            "Missing required environment variables:\n"
            + "\n".join(f"- {var}: {desc}" for var, desc in missing_vars.items())
        )

# Initialize LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, max_tokens=1000) 
"""Configuration utilities"""
import os
from dotenv import load_dotenv

def configure_environment():
    """Configure environment variables"""
    load_dotenv()  # Load environment variables from .env file
    
    required_vars = {
        "OPENAI_API_KEY": "OpenAI API key for language model",
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
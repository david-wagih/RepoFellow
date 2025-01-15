"""Configuration for agent models and capabilities"""
from typing import Dict, Any, List
from dataclasses import dataclass
from enum import Enum

class ModelType(Enum):
    """Supported model types"""
    GPT_4 = "gpt-4"
    GPT_35_TURBO = "gpt-3.5-turbo"
    GPT_4_TURBO = "gpt-4-turbo-preview"
    CLAUDE_3_OPUS = "claude-3-opus-20240229"
    CLAUDE_3_SONNET = "claude-3-sonnet-20240229"
    MIXTRAL = "mixtral-8x7b"
    CODELLAMA = "codellama-34b"

@dataclass
class ModelConfig:
    """Configuration for a specific model"""
    model_type: ModelType
    temperature: float
    max_tokens: int
    top_p: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0

@dataclass
class AgentConfig:
    """Configuration for an agent"""
    name: str
    description: str
    capabilities: List[str]
    primary_model: ModelConfig
    fallback_model: ModelConfig = None
    required_tools: List[str] = None

# Default model configurations
DEFAULT_MODELS = {
    "precise": ModelConfig(
        model_type=ModelType.GPT_4,
        temperature=0.1,
        max_tokens=4000,
    ),
    "balanced": ModelConfig(
        model_type=ModelType.GPT_35_TURBO,
        temperature=0.7,
        max_tokens=2000,
    ),
    "creative": ModelConfig(
        model_type=ModelType.GPT_4_TURBO,
        temperature=0.9,
        max_tokens=4000,
    ),
    "code": ModelConfig(
        model_type=ModelType.CODELLAMA,
        temperature=0.2,
        max_tokens=2000,
    ),
}

# Agent configurations
AGENT_CONFIGS = {
    "code_agent": AgentConfig(
        name="Code Analysis Agent",
        description="Specializes in code understanding and modification",
        capabilities=[
            "code_analysis",
            "code_generation",
            "debugging",
            "refactoring",
        ],
        primary_model=DEFAULT_MODELS["code"],
        fallback_model=DEFAULT_MODELS["precise"],
        required_tools=["ast_parser", "linter"],
    ),
    
    "router_agent": AgentConfig(
        name="Command Router",
        description="Routes commands to appropriate agents",
        capabilities=[
            "command_parsing",
            "intent_analysis",
            "workflow_management",
        ],
        primary_model=DEFAULT_MODELS["precise"],
    ),
    
    "business_agent": AgentConfig(
        name="Business Analyst",
        description="Handles business requirement analysis",
        capabilities=[
            "requirement_analysis",
            "user_story_generation",
            "effort_estimation",
        ],
        primary_model=DEFAULT_MODELS["balanced"],
        required_tools=["jira_client"],
    ),
}

def get_agent_config(agent_type: str) -> AgentConfig:
    """Get configuration for specific agent type"""
    if agent_type not in AGENT_CONFIGS:
        raise ValueError(f"Unknown agent type: {agent_type}")
    return AGENT_CONFIGS[agent_type]

def get_model_config(model_name: str) -> ModelConfig:
    """Get configuration for specific model"""
    if model_name not in DEFAULT_MODELS:
        raise ValueError(f"Unknown model configuration: {model_name}")
    return DEFAULT_MODELS[model_name] 
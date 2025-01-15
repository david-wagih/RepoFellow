"""Configuration file for all system and user prompts"""
from typing import Dict, Any

# Base prompts for different agent types
BASE_PROMPTS = {
    "code_agent": """You are an expert software developer with deep understanding of code.
    Core capabilities:
    1. Analyze and explain code
    2. Generate code based on requirements
    3. Suggest improvements and best practices
    4. Debug issues and propose solutions
    Always provide clear explanations and well-structured code.""",
    
    "router_agent": """You are an expert CLI command router.
    Core capabilities:
    1. Analyze user intent from CLI commands
    2. Route to appropriate specialized agents
    3. Handle command parsing and validation
    4. Manage multi-step workflows
    Always choose the most appropriate handler based on command intent.""",
    
    "analysis_agent": """You are a code analysis specialist.
    Core capabilities:
    1. Perform deep code structure analysis
    2. Identify patterns and anti-patterns
    3. Generate architectural insights
    4. Analyze dependencies and relationships
    Always provide comprehensive and actionable analysis.""",
    
    "business_agent": """You are a business requirements analyst.
    Core capabilities:
    1. Convert requirements to user stories
    2. Analyze business logic in code
    3. Generate acceptance criteria
    4. Provide effort estimations
    Always focus on business value and user needs."""
}

# Template prompts for specific tasks
TASK_PROMPTS = {
    "code_review": """Analyze the following code for:
    - Code quality
    - Best practices
    - Potential improvements
    - Security concerns
    
    Code to review:
    {code}
    """,
    
    "dependency_analysis": """Analyze the project dependencies:
    - Identify core dependencies
    - Check for security issues
    - Suggest updates/alternatives
    - Analyze dependency graph
    
    Dependencies:
    {dependencies}
    """,
    
    "architecture_review": """Review the system architecture:
    - Component relationships
    - Data flow
    - Integration points
    - Scalability concerns
    
    System context:
    {context}
    """
}

def get_prompt(prompt_type: str, **kwargs: Any) -> str:
    """Get formatted prompt by type with optional parameters"""
    if prompt_type in BASE_PROMPTS:
        return BASE_PROMPTS[prompt_type]
    elif prompt_type in TASK_PROMPTS:
        return TASK_PROMPTS[prompt_type].format(**kwargs)
    raise ValueError(f"Unknown prompt type: {prompt_type}") 
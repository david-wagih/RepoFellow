"""Mappings between CLI commands and agent capabilities"""

CLI_COMMAND_MAPPINGS = {
    "ask": {
        "agent": "code_agent",
        "capabilities": ["code_explanation", "documentation_search"],
        "prompt_template": "Answer this question about the codebase: {question}"
    },
    "modify": {
        "agent": "code_agent",
        "capabilities": ["code_modification", "refactoring"],
        "prompt_template": "Modify the code according to this instruction: {instruction}"
    },
    "check_quality": {
        "agent": "code_agent",
        "capabilities": ["code_analysis", "quality_check"],
        "prompt_template": "Analyze code quality and provide detailed report"
    },
    "analyze": {
        "agent": "documentation_agent",
        "capabilities": ["codebase_analysis", "visualization"],
        "prompt_template": "Provide comprehensive analysis of the codebase"
    },
    "visualize": {
        "agent": "documentation_agent",
        "capabilities": ["diagram_generation", "architecture_visualization"],
        "prompt_template": "Generate visual representation of the codebase"
    }
} 
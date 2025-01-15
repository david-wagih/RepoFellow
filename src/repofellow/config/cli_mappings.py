"""CLI command mappings configuration"""

CLI_COMMAND_MAPPINGS = {
    "ask": {
        "prompt_template": "Answer the following question about the codebase: {question}",
        "agent": "code_agent",
        "capabilities": ["code_understanding", "documentation"]
    },
    "analyze": {
        "prompt_template": "Analyze the codebase and provide insights",
        "agent": "analysis_agent",
        "capabilities": ["code_analysis", "metrics"]
    },
    "modify": {
        "prompt_template": "Modify the code according to: {instruction}",
        "agent": "modification_agent",
        "capabilities": ["code_modification"]
    },
    "check_quality": {
        "prompt_template": "Perform a code quality analysis",
        "agent": "quality_agent",
        "capabilities": ["code_quality", "linting"]
    }
} 
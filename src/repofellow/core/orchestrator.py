"""Core orchestration logic"""
from typing import Dict, Any, Optional
from pathlib import Path
from rich.console import Console
from rich.progress import Progress
from rich.markdown import Markdown
from .codebase import CodebaseContext
from ..agents.registry import registry

class AssistantOrchestrator:
    """Orchestrates interactions between CLI and agent system"""
    
    def __init__(self, console: Console):
        self.console = console
        self.current_state: Optional[Dict[str, Any]] = None
        
    def process_cli_command(self, command: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Process CLI command by converting it to agent-compatible state"""
        
        # Build initial state with all necessary fields
        initial_state = {
            "messages": [],
            "context": [],
            "files": [],
            "query": self._build_query(command, args),
            "query_type": command,
            "response_ready": False,
            # Add command-specific arguments to state
            **args
        }
        
        if 'path' in args:
            initial_state['code_context'] = self._load_codebase(args['path'])
            
        # Run through agent graph
        final_state = registry.get_agent("router").invoke(initial_state)
        self.current_state = final_state
        return final_state
    
    def _build_query(self, command: str, args: Dict[str, Any]) -> str:
        """Build natural language query from CLI command and args"""
        if command == "ask":
            return args.get("question", "")
        elif command == "modify":
            return f"Modify code: {args.get('instruction', '')}"
        elif command == "check_quality":
            return "Analyze code quality and provide detailed report"
        elif command == "analyze":
            return "Provide comprehensive codebase analysis"
        return ""
        
    def _load_codebase(self, path: Path) -> str:
        """Load codebase context from path"""
        with Progress() as progress:
            task = progress.add_task("Loading codebase...", total=None)
            codebase = CodebaseContext(path)
            progress.update(task, completed=True)
            return codebase.get_summary()
            
    def format_response(self, state: Dict[str, Any]) -> str | Markdown:
        """Format agent response for CLI output"""
        if state.get("error"):
            return f"Error: {state['error']}"
            
        response = state.get("response", "")
        if state.get("generated_artifacts"):
            artifacts = state["generated_artifacts"]
            if "diagram" in artifacts:
                self.console.print("\nGenerated diagram:")
                # Handle diagram display logic here
                
        return Markdown(response) if isinstance(response, str) else response 
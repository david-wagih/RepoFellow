"""Ask command implementation"""
from pathlib import Path
import typer
from rich.console import Console
from rich.markdown import Markdown
from ...core.orchestrator import AssistantOrchestrator

def ask(
    question: str = typer.Argument(..., help="Your question about the codebase"),
    path: Path = typer.Option(".", help="Path to the codebase directory"),
):
    """Ask questions about your codebase."""
    console = Console()
    orchestrator = AssistantOrchestrator(console)
    try:
        result = orchestrator.process_cli_command("ask", {
            "question": question,
            "path": path
        })
        response = orchestrator.format_response(result)
        console.print(Markdown(response) if isinstance(response, str) else response)
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]") 
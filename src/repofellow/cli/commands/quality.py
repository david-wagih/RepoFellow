"""Quality check command implementation"""
from pathlib import Path
import typer
from rich.console import Console
from ...core.orchestrator import AssistantOrchestrator

def quality(
    path: Path = typer.Option(".", help="Path to check"),
    format: str = typer.Option("rich", "--format", "-f", help="Output format")
):
    """Check code quality using various tools."""
    console = Console()
    orchestrator = AssistantOrchestrator(console)
    try:
        result = orchestrator.process_cli_command("check_quality", {
            "path": path,
            "format": format
        })
        response = orchestrator.format_response(result)
        console.print(response)
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]") 
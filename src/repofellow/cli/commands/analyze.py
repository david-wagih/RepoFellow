"""Analyze command implementation"""
from pathlib import Path
import typer
from rich.console import Console
from ...core.orchestrator import AssistantOrchestrator

def analyze(
    path: Path = typer.Option(".", help="Path to analyze"),
    format: str = typer.Option("rich", "--format", "-f", help="Output format")
):
    """Analyze codebase structure and provide insights."""
    console = Console()
    orchestrator = AssistantOrchestrator(console)
    try:
        result = orchestrator.process_cli_command("analyze", {
            "path": path,
            "format": format
        })
        response = orchestrator.format_response(result)
        console.print(response)
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]") 
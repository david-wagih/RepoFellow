"""Quality check command implementation"""
from pathlib import Path
import typer
from rich.console import Console
from ...core.orchestrator import AssistantOrchestrator

def command(
    path: Path = typer.Option(".", help="Path to check"),
    format: str = typer.Option("rich", "--format", "-f", help="Output format")
):
    """Check code quality using various tools."""
    orchestrator = AssistantOrchestrator(Console())
    result = orchestrator.process_cli_command("check_quality", {
        "path": path,
        "format": format
    })
    return orchestrator.format_response(result) 
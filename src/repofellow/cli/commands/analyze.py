"""Analyze command implementation"""
from pathlib import Path
import typer
from rich.console import Console
from ...core.orchestrator import AssistantOrchestrator

def command(
    path: Path = typer.Option(".", help="Path to analyze"),
    format: str = typer.Option("rich", "--format", "-f", help="Output format")
):
    """Analyze codebase structure and provide insights."""
    orchestrator = AssistantOrchestrator(Console())
    result = orchestrator.process_cli_command("analyze", {
        "path": path,
        "format": format
    })
    return orchestrator.format_response(result) 
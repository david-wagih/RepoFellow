"""Ask command implementation"""
from pathlib import Path
import typer
from rich.console import Console
from ...core.orchestrator import AssistantOrchestrator

def command(
    question: str = typer.Argument(..., help="Your question about the codebase"),
    path: Path = typer.Option(".", help="Path to the codebase directory"),
):
    """Ask questions about your codebase."""
    orchestrator = AssistantOrchestrator(Console())
    result = orchestrator.process_cli_command("ask", {
        "question": question,
        "path": path
    })
    return orchestrator.format_response(result) 
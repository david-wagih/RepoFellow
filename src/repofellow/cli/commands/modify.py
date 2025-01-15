"""Modify command implementation"""
from pathlib import Path
import typer
from rich.console import Console
from ...core.orchestrator import AssistantOrchestrator

def command(
    instruction: str = typer.Argument(..., help="Instructions for code modification"),
    path: Path = typer.Option(".", help="Path to the codebase directory"),
    file: str = typer.Option(None, help="Specific file to modify")
):
    """Get AI-powered suggestions for code modifications."""
    orchestrator = AssistantOrchestrator(Console())
    result = orchestrator.process_cli_command("modify", {
        "instruction": instruction,
        "path": path,
        "file": file
    })
    return orchestrator.format_response(result) 
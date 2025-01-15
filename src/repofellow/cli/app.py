"""Main CLI application module"""
import typer
from rich.console import Console
from .commands import ask, analyze, modify, quality
from .ui.console import create_console
from .ui.formatting import styled_header

app = typer.Typer(
    name="repofellow",
    help="🤖 [cyan]RepoFellow[/cyan] - Your AI-powered coding companion",
    add_completion=True,
    rich_markup_mode="rich",
)

# Register commands
app.command()(ask.command)
app.command()(analyze.command)
app.command()(modify.command)
app.command()(quality.command) 
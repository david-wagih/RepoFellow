"""Main CLI application module"""
import typer
from .commands import ask, analyze, modify, quality

app = typer.Typer(
    name="repofellow",
    help="🤖 [cyan]RepoFellow[/cyan] - Your AI-powered coding companion",
    add_completion=True,
    rich_markup_mode="rich",
)

# Register commands correctly
app.command(name="ask")(ask)
app.command(name="analyze")(analyze)
app.command(name="modify")(modify)
app.command(name="quality")(quality)

def main():
    """Entry point for the CLI"""
    app()

if __name__ == "__main__":
    main() 
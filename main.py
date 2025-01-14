import typer
from pathlib import Path
from rich import print
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.markdown import Markdown
from codebase import CodebaseContext
from agent import CodeAssistant
from typing import Optional, List, Tuple
from pylint.lint import Run
import logging
from datetime import datetime
from rich.live import Live
from rich.layout import Layout
from rich.syntax import Syntax
from rich.console import Group
from rich.columns import Columns
from rich import box
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
    TaskProgressColumn
)
import json
import time

console = Console()
logger = None

def get_logger():
    global logger
    if logger is None:
        logger = logging.getLogger("code-assistant")
    return logger

def create_progress() -> Progress:
    """Create a modern progress bar style"""
    return Progress(
        SpinnerColumn(spinner_name="dots12", style="cyan"),
        TextColumn("[bold blue]{task.description}", justify="right"),
        BarColumn(
            complete_style="cyan",
            finished_style="green"
        ),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        expand=True,
        transient=True  # Progress bar disappears after completion
    )

def create_layout() -> Layout:
    """Create a layout for complex displays"""
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main", ratio=1),
        Layout(name="footer", size=3)
    )
    return layout

def styled_header(title: str) -> Panel:
    """Create a styled header panel"""
    return Panel(
        f"[bold cyan]{title}[/bold cyan]",
        box=box.ROUNDED,
        style="blue",
        border_style="bright_blue"
    )

app = typer.Typer(
    name="code-assistant",
    help="🤖 [cyan]Local Code Assistant[/cyan] - Your AI-powered coding companion",
    add_completion=True,
    rich_markup_mode="rich",
    pretty_exceptions_enable=False,  # We'll handle exceptions ourselves
)

def print_version(value: bool):
    """Print version information"""
    if value:
        version_panel = Panel(
            Group(
                "[yellow]Local Code Assistant[/yellow]",
                "[cyan]Version:[/cyan] 1.0.0",
                "[cyan]Python:[/cyan] 3.12+",
                "[cyan]Created by:[/cyan] Your Name",
            ),
            title="[bold yellow]Version Info[/bold yellow]",
            border_style="yellow",
            box=box.ROUNDED
        )
        console.print(version_panel)
        raise typer.Exit()

# Setup logging
def setup_logging(debug: bool = False):
    """Configure logging based on debug flag"""
    log_level = logging.DEBUG if debug else logging.INFO
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"code_assistant_{timestamp}.log"
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler() if debug else logging.NullHandler()
        ]
    )
    return logging.getLogger("code-assistant")

# Modify the main callback to include debug flag
@app.callback()
def main(
    version: bool = typer.Option(None, "--version", "-v", help="Show version", callback=print_version),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug logging"),
):
    """
    🤖 Local Code Assistant - Your AI-powered coding companion
    
    Run locally and securely analyze and modify your codebase.
    """
    global logger
    logger = setup_logging(debug)
    if debug:
        logger.debug("Debug mode enabled")

@app.command(help="❓ Ask questions about your codebase")
def ask(
    question: str = typer.Argument(..., help="Your question about the codebase"),
    path: Path = typer.Option(".", help="Path to the codebase directory", show_default=True),
):
    """
    Ask questions about your codebase and get AI-powered answers.
    
    Examples:
    \b
    [green]Basic usage:[/green]
        $ code-assistant ask "How is error handling implemented?"
    
    [green]Query specific feature:[/green]
        $ code-assistant ask "Explain the logging system"
    
    [green]With custom path:[/green]
        $ code-assistant ask --path=./src "How are tests organized?"
    """
    with create_progress() as progress:
        # Initialize tasks
        analyze_task = progress.add_task("[cyan]Analyzing codebase...", total=None)
        codebase = CodebaseContext(path)
        progress.update(analyze_task, description="[cyan]Initializing AI assistant...")
        
        assistant = CodeAssistant()
        progress.update(analyze_task, description="[cyan]Generating response...")
        response = assistant.answer_question(question, codebase)
        
        progress.update(analyze_task, description="[green]Done!", completed=True)
    
    console.print("\n")  # Add some spacing
    console.print(Panel(response, title="[bold green]Assistant's Response", border_style="green"))

def parse_file_reference(reference: str) -> Tuple[str, Optional[Tuple[int, int]]]:
    """Parse file reference with optional line numbers (e.g., 'file.py:10-20' or 'file.py')"""
    if ':' in reference:
        file_path, line_range = reference.split(':', 1)
        try:
            if '-' in line_range:
                start, end = map(int, line_range.split('-'))
                return file_path, (start, end)
            else:
                line_num = int(line_range)
                return file_path, (line_num, line_num)
        except ValueError:
            return reference, None
    return reference, None

@app.command(help="🔍 Inspect specific code sections")
def inspect(
    file: str = typer.Argument(..., help="File to inspect"),
    question: str = typer.Argument(..., help="Question about the code"),
    lines: str = typer.Option(None, "--lines", "-l", help="Line range (e.g., '10-20')"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug logging"),
):
    """Analyze specific sections of code with detailed insights."""
    logger = setup_logging(debug)
    logger.debug(f"Inspecting file: {file} with question: {question}")
    
    # Create layout for display
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=3)
    )
    
    # Split body for code and analysis
    layout["body"].split_row(
        Layout(name="code", ratio=1),
        Layout(name="analysis", ratio=1)
    )
    
    # Parse line range if provided
    line_range = None
    if lines:
        try:
            start, end = map(int, lines.split('-'))
            line_range = (start, end)
        except ValueError:
            console.print("[red]Invalid line range format. Use 'start-end' (e.g., '10-20')[/red]")
            return
    
    with Live(layout, refresh_per_second=4, screen=True) as live:
        try:
            # Initialize codebase
            codebase = CodebaseContext(".")
            
            # Get file content
            content = codebase.get_file_content(file)
            if not content:
                console.print(f"[red]Error: File '{file}' not found[/red]")
                return
            
            # Update header
            layout["header"].update(Panel(
                f"[bold cyan]Inspecting: {file}[/bold cyan]",
                style="white on dark_blue"
            ))
            
            # Show code section
            if line_range:
                lines = content.splitlines()
                start, end = line_range
                if 1 <= start <= len(lines) and 1 <= end <= len(lines):
                    selected_content = '\n'.join(lines[start-1:end])
                    syntax = Syntax(selected_content, "python", line_numbers=True, start_line=start)
                else:
                    console.print(f"[red]Error: Line range {start}-{end} is out of bounds[/red]")
                    return
            else:
                syntax = Syntax(content, "python", line_numbers=True)
            
            layout["code"].update(Panel(
                syntax,
                title="[bold cyan]Code[/bold cyan]",
                border_style="cyan"
            ))
            
            # Show loading state
            layout["analysis"].update(Panel(
                "[yellow]Analyzing code...[/yellow]",
                title="[bold cyan]Analysis[/bold cyan]",
                border_style="cyan"
            ))
            
            # Get analysis
            assistant = CodeAssistant()
            analysis = assistant.inspect_code(file, question, codebase, line_range)
            
            # Show analysis with markdown formatting
            layout["analysis"].update(Panel(
                Markdown(analysis),
                title="[bold cyan]Analysis[/bold cyan]",
                border_style="cyan"
            ))
            
            # Update footer
            layout["footer"].update(Panel(
                "[dim]Press Ctrl+C to exit[/dim]",
                style="dim"
            ))
            
            # Keep display until user exits
            while True:
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            pass
        except Exception as e:
            console.print(f"[red]Error: {str(e)}[/red]")

@app.command(help="✏️ Get suggestions for code modifications")
def modify(
    instruction: str = typer.Argument(..., help="Instructions for code modification"),
    path: Path = typer.Option(".", help="Path to the codebase directory", show_default=True),
    file: str = typer.Option(None, help="Specific file to modify (e.g., 'file.py' or 'file.py:10-20')"),
):
    """Get AI-powered suggestions for code modifications"""
    file_path = None
    line_range = None
    if file:
        file_path, line_range = parse_file_reference(file)

    with create_progress() as progress:
        task1 = progress.add_task("[cyan]Loading codebase...", total=None)
        codebase = CodebaseContext(path)
        
        progress.update(task1, description="[cyan]Analyzing code structure...")
        assistant = CodeAssistant()
        
        progress.update(task1, description="[cyan]Generating modifications...")
        modifications = assistant.suggest_modifications(instruction, codebase, file_path, line_range)
        
        progress.update(task1, description="[green]Done!", completed=True)
    
    console.print("\n")
    console.print(Panel(modifications, title="[bold green]Suggested Modifications", border_style="green"))

@app.command(help="📊 Analyze codebase")
def analyze(
    path: Path = typer.Option(".", help="Path to analyze", show_default=True),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug logging"),
    format: str = typer.Option("rich", "--format", "-f", help="Output format (rich/json/simple)")
):
    """Analyze codebase structure and provide detailed insights."""
    logger = setup_logging(debug)
    logger.debug(f"Analyzing codebase at: {path}")
    
    # Create layout
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=3)
    )
    
    # Split body into sections
    layout["body"].split_row(
        Layout(name="stats", ratio=1),
        Layout(name="details", ratio=2)
    )
    
    with Live(layout, refresh_per_second=4, screen=True) as live:
        try:
            # Update header
            layout["header"].update(Panel(
                "[bold cyan]Codebase Analysis[/bold cyan]",
                style="white on dark_blue"
            ))
            
            # Initialize stats table
            stats_table = Table.grid(expand=True)
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value", style="yellow")
            
            # Show loading state
            layout["stats"].update(Panel(
                "[yellow]Loading codebase...[/yellow]",
                title="[bold cyan]Statistics[/bold cyan]",
                border_style="cyan"
            ))
            
            # Initialize codebase
            codebase = CodebaseContext(path)
            files = codebase.get_all_files()
            
            # Collect detailed statistics
            total_files = len(files)
            total_lines = sum(len(content.splitlines()) for content in files.values())
            file_types = {}
            largest_files = []
            
            for file_path, content in files.items():
                # Count file types
                ext = Path(file_path).suffix
                file_types[ext] = file_types.get(ext, 0) + 1
                
                # Track large files
                lines = len(content.splitlines())
                largest_files.append((file_path, lines))
            
            # Sort largest files
            largest_files.sort(key=lambda x: x[1], reverse=True)
            largest_files = largest_files[:5]  # Top 5
            
            # Update stats table
            stats_table.add_row("Total Files", str(total_files))
            stats_table.add_row("Total Lines", str(total_lines))
            stats_table.add_row("File Types", str(len(file_types)))
            
            # Create file types table
            types_table = Table(
                title="File Types",
                show_header=True,
                header_style="bold cyan",
                box=box.ROUNDED
            )
            types_table.add_column("Extension", style="cyan")
            types_table.add_column("Count", style="yellow", justify="right")
            
            for ext, count in sorted(file_types.items()):
                types_table.add_row(ext or "(no ext)", str(count))
            
            # Create largest files table
            largest_table = Table(
                title="Largest Files",
                show_header=True,
                header_style="bold cyan",
                box=box.ROUNDED
            )
            largest_table.add_column("File", style="cyan")
            largest_table.add_column("Lines", style="yellow", justify="right")
            
            for file_path, lines in largest_files:
                largest_table.add_row(str(Path(file_path).name), str(lines))
            
            # Update layout with all tables
            layout["stats"].update(Panel(
                Group(stats_table, types_table),
                title="[bold cyan]Statistics[/bold cyan]",
                border_style="cyan"
            ))
            
            layout["details"].update(Panel(
                Group(
                    largest_table,
                    "\n[bold cyan]Code Insights:[/bold cyan]",
                    Markdown(codebase.get_summary())
                ),
                title="[bold cyan]Details[/bold cyan]",
                border_style="cyan"
            ))
            
            # Update footer
            layout["footer"].update(Panel(
                "[dim]Press Ctrl+C to exit[/dim]",
                style="dim"
            ))
            
            # Keep display until user exits
            while True:
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            pass
        except Exception as e:
            console.print(f"[red]Error: {str(e)}[/red]")

@app.command(help="📝 Generate or update README.md file")
def readme(
    path: Path = typer.Option(".", help="Path to the codebase directory", show_default=True),
    preview: bool = typer.Option(False, "--preview", "-p", help="Preview README without saving"),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing README.md without confirmation"),
):
    """Generate a comprehensive README.md file for the codebase"""
    with create_progress() as progress:
        # Multiple steps for README generation
        task1 = progress.add_task("[cyan]Analyzing codebase...", total=100)
        codebase = CodebaseContext(path)
        progress.update(task1, advance=30)
        
        progress.update(task1, description="[cyan]Initializing AI assistant...")
        assistant = CodeAssistant()
        progress.update(task1, advance=20)
        
        progress.update(task1, description="[cyan]Generating README content...")
        readme_content = assistant.generate_readme(codebase)
        progress.update(task1, advance=50, description="[green]Done!")
    
    readme_path = path / "README.md"
    
    if preview:
        console.print("\n[bold cyan]Preview of README.md:[/bold cyan]")
        console.print(Markdown(readme_content))
        return
    
    if readme_path.exists() and not force:
        overwrite = typer.confirm(
            "README.md already exists. Do you want to overwrite it?",
            abort=True
        )
    
    try:
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        console.print(f"\n[green]✓ Successfully generated README.md at {readme_path}[/green]")
        
        if typer.confirm("Would you like to preview the generated README?"):
            console.print(Markdown(readme_content))
            
    except Exception as e:
        console.print(f"\n[red]Error writing README.md: {str(e)}[/red]")

@app.command(help="🔍 Check code quality using pylint")
def check_quality(
    path: Path = typer.Option(".", help="Path to the codebase directory", show_default=True),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug logging"),
    format: str = typer.Option("table", "--format", "-f", help="Output format (table/json/simple)")
):
    """Run code quality checks on the codebase using pylint."""
    logger = setup_logging(debug)
    logger.debug(f"Starting code quality check for path: {path}")

    # Create a more sophisticated layout
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=3)
    )
    
    # Split the body into status and results
    layout["body"].split_row(
        Layout(name="status", ratio=1),
        Layout(name="results", ratio=2)
    )

    # Create progress display
    progress_table = Table.grid(expand=True)
    progress_table.add_column(justify="left", ratio=1)
    progress_table.add_column(justify="right")
    
    # Create status display
    status_panel = Panel(
        progress_table,
        title="[bold cyan]Status[/bold cyan]",
        border_style="cyan",
        padding=(1, 2)
    )
    
    # Initialize results table
    results_table = Table(
        show_header=True,
        header_style="bold cyan",
        box=box.ROUNDED,
        title="[bold]Detailed Report[/bold]",
        padding=(0, 1)
    )
    results_table.add_column("File", style="cyan", no_wrap=True)
    results_table.add_column("Line", style="magenta", justify="right")
    results_table.add_column("Type", style="yellow")
    results_table.add_column("Message", style="white")
    
    # Create results panel with initial table
    results_panel = Panel(
        results_table,
        title="[bold cyan]Results[/bold cyan]",
        border_style="cyan",
        padding=(1, 2)
    )

    # Update layout sections
    layout["header"].update(Panel(
        "[bold cyan]Code Quality Check[/bold cyan]",
        style="white on dark_blue"
    ))
    layout["status"].update(status_panel)
    layout["results"].update(results_panel)
    layout["footer"].update(Panel(
        "[dim]Press Ctrl+C to cancel[/dim]",
        style="dim"
    ))

    # Create Live display with auto_refresh
    with Live(layout, refresh_per_second=4, screen=True, auto_refresh=True) as live:
        try:
            # Update progress
            progress_table.add_row("Status", "[yellow]Initializing...[/yellow]")
            live.refresh()
            
            codebase = CodebaseContext(path)
            progress_table.add_row("Files Found", f"[cyan]{len(codebase.get_python_files())} Python files[/cyan]")
            progress_table.add_row("Checking", "[yellow]Running pylint...[/yellow]")
            live.refresh()
            
            assistant = CodeAssistant()
            quality_report = assistant._check_code_quality(codebase)
            
            # Parse results
            report_data = json.loads(quality_report)
            
            # Group issues by type
            issues_by_type = {}
            for message in report_data:
                msg_type = message['type']
                if msg_type not in issues_by_type:
                    issues_by_type[msg_type] = []
                issues_by_type[msg_type].append(message)
            
            # Update status with summary
            progress_table.add_row("", "")  # Add spacing
            progress_table.add_row("Status", "[green]Analysis Complete![/green]")
            for msg_type, messages in issues_by_type.items():
                color = {
                    'error': 'red',
                    'warning': 'yellow',
                    'convention': 'blue',
                    'refactor': 'magenta'
                }.get(msg_type, 'white')
                progress_table.add_row(
                    f"{msg_type.title()}",
                    f"[{color}]{len(messages)} found[/{color}]"
                )
            live.refresh()

            # Add results to table
            for message in report_data:
                icon = {
                    'error': '🔴',
                    'warning': '⚠️',
                    'convention': 'ℹ️',
                    'refactor': '🔧'
                }.get(message['type'], '•')
                
                results_table.add_row(
                    str(Path(message['path']).name),
                    str(message['line']),
                    f"{icon} {message['type'].title()}",
                    message['message']
                )
                live.refresh()

            # Add final status
            progress_table.add_row("", "")
            progress_table.add_row("Press", "[cyan]Ctrl+C to exit[/cyan]")
            live.refresh()

            # Keep the display visible until user interrupts
            try:
                while True:
                    live.refresh()
                    time.sleep(0.1)
            except KeyboardInterrupt:
                progress_table.add_row("Status", "[yellow]Exiting...[/yellow]")
                live.refresh()

        except Exception as e:
            progress_table.add_row("Status", f"[red]Error: {str(e)}[/red]")
            live.refresh()

    # If JSON format was requested, print it after the live display
    if format == "json":
        console.print_json(data=report_data)

def get_command_examples() -> dict:
    """Get rich examples for each command"""
    return {
        "ask": [
            ("Ask about code structure", "code-assistant ask 'How is the error handling implemented?'"),
            ("Query specific feature", "code-assistant ask 'How does the logging system work?'"),
            ("Get implementation details", "code-assistant ask 'Explain the CodebaseContext class'")
        ],
        "inspect": [
            ("Check specific file", "code-assistant inspect main.py 'How is the CLI structured?'"),
            ("Analyze code section", "code-assistant inspect 'agent.py:50-70' 'Explain this function'"),
            ("Review error handling", "code-assistant inspect agent.py 'How are errors handled here?'")
        ],
        "modify": [
            ("Add new feature", "code-assistant modify 'Add error retry logic to API calls'"),
            ("Improve code", "code-assistant modify 'Optimize the file loading process'"),
            ("Fix specific file", "code-assistant modify --file=agent.py 'Add input validation'")
        ],
        "analyze": [
            ("Full analysis", "code-assistant analyze"),
            ("Specific directory", "code-assistant analyze --path=./src"),
            ("With debug info", "code-assistant analyze --debug")
        ],
        "readme": [
            ("Generate README", "code-assistant readme"),
            ("Preview only", "code-assistant readme --preview"),
            ("Force update", "code-assistant readme --force")
        ],
        "check-quality": [
            ("Basic check", "code-assistant check-quality"),
            ("JSON output", "code-assistant check-quality --format=json"),
            ("Specific path", "code-assistant check-quality --path=./src --debug")
        ]
    }

def print_command_help(command: str):
    """Print detailed help for a specific command"""
    examples = get_command_examples().get(command, [])
    if not examples:
        return
    
    # Create examples table
    example_table = Table(
        show_header=True,
        header_style="bold cyan",
        box=box.ROUNDED,
        title=f"[bold]Examples for '{command}'[/bold]",
        padding=(0, 1)
    )
    example_table.add_column("Description", style="green")
    example_table.add_column("Command", style="yellow")
    
    for desc, cmd in examples:
        example_table.add_row(desc, f"$ {cmd}")
    
    return example_table

def print_commands():
    """Print available commands in a modern format with examples"""
    commands_panel = Panel(
        Group(
            "[bold cyan]Available Commands[/bold cyan]\n",
            Columns([
                Panel(
                    Group(
                        f"[cyan]{cmd}[/cyan]",
                        f"[white]{desc}[/white]",
                        print_command_help(cmd.strip("🔍✏️📊📝❓ ")),
                        "\n"
                    ),
                    box=box.ROUNDED,
                    padding=(1, 2)
                )
                for cmd, desc in [
                    ("🔍 ask", "Ask questions about your codebase"),
                    ("✏️ modify", "Get suggestions for code modifications"),
                    ("📊 analyze", "Show codebase statistics and summary"),
                    ("📝 readme", "Generate or update README.md file"),
                    ("🔎 check-quality", "Check code quality using pylint")
                ]
            ], equal=True, expand=True)
        ),
        box=box.ROUNDED,
        border_style="blue",
        padding=(1, 2),
        title="[bold blue]Local Code Assistant[/bold blue]"
    )
    
    console.print("\n")
    console.print(commands_panel)

if __name__ == "__main__":
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {str(e)}[/red]")

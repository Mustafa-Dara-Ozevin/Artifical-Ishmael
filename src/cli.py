"""CLI interface for Moby-Dick GraphRAG Encyclopedia."""

import sys
import logging
from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich import print as rprint

from .config import get_config, validate_config
from .neo4j_client import get_neo4j_client
from .query_engine import get_query_engine, QueryResult

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

app = typer.Typer(
    name="ishmael",
    help="🐋 Moby-Dick GraphRAG Encyclopedia - Ask anything about Melville's masterpiece!"
)
console = Console()


def check_configuration() -> bool:
    """Check if the application is properly configured."""
    config = get_config()
    errors = validate_config(config)
    
    if errors:
        console.print("[bold red]Configuration Error:[/bold red]")
        for error in errors:
            console.print(f"  • {error}")
        console.print("\n[dim]Please check your .env file.[/dim]")
        return False
    return True


def check_neo4j_connection() -> bool:
    """Check Neo4j connection."""
    try:
        client = get_neo4j_client()
        return client.verify_connectivity()
    except Exception as e:
        console.print(f"[bold red]Neo4j Connection Error:[/bold red] {e}")
        return False


def display_result(result: QueryResult, show_sources: bool = True):
    """Display a query result with rich formatting."""
    # Display the answer as markdown
    console.print()
    console.print(Panel(
        Markdown(result.answer),
        title=f"[bold blue]📖 {result.query_type.value.title()} Query[/bold blue]",
        border_style="blue"
    ))
    
    # Display sources if requested
    if show_sources and result.sources:
        console.print()
        table = Table(title="📚 Sources Used", show_header=True, header_style="bold cyan")
        table.add_column("Type", style="dim")
        table.add_column("Name")
        table.add_column("Layer")
        table.add_column("Score", justify="right")
        
        for source in result.sources[:10]:  # Limit to 10 sources
            table.add_row(
                source["type"],
                source["name"],
                source["layer"],
                f"{source['score']:.2f}"
            )
        
        console.print(table)


@app.command()
def ask(
    question: str = typer.Argument(..., help="Your question about Moby-Dick"),
    stream: bool = typer.Option(False, "--stream", "-s", help="Stream the response"),
    sources: bool = typer.Option(True, "--sources/--no-sources", help="Show source citations"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
):
    """Ask a question about Moby-Dick."""
    if verbose:
        logging.getLogger().setLevel(logging.INFO)
    
    if not check_configuration():
        raise typer.Exit(1)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True
    ) as progress:
        progress.add_task("Connecting to Neo4j...", total=None)
        if not check_neo4j_connection():
            raise typer.Exit(1)
        
        progress.add_task("Retrieving context...", total=None)
        engine = get_query_engine()
        
        if stream:
            console.print()
            console.print("[bold blue]📖 Response:[/bold blue]")
            console.print()
            
            for chunk in engine.query_stream(question):
                console.print(chunk, end="")
            console.print()
        else:
            progress.add_task("Generating response...", total=None)
            result = engine.query(question)
            display_result(result, show_sources=sources)


@app.command()
def character(
    name: str = typer.Argument(..., help="Character name (e.g., 'Ishmael', 'Ahab', 'Queequeg')"),
    sources: bool = typer.Option(True, "--sources/--no-sources", help="Show source citations")
):
    """Get detailed information about a character."""
    if not check_configuration():
        raise typer.Exit(1)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True
    ) as progress:
        progress.add_task(f"Looking up {name}...", total=None)
        
        if not check_neo4j_connection():
            raise typer.Exit(1)
        
        engine = get_query_engine()
        result = engine.ask_about_character(name)
        display_result(result, show_sources=sources)


@app.command()
def chapter(
    number: int = typer.Argument(..., help="Chapter number (1-135)"),
    sources: bool = typer.Option(True, "--sources/--no-sources", help="Show source citations")
):
    """Get information about a specific chapter."""
    if not check_configuration():
        raise typer.Exit(1)
    
    if number < 1 or number > 135:
        console.print("[bold red]Error:[/bold red] Chapter number must be between 1 and 135")
        raise typer.Exit(1)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True
    ) as progress:
        progress.add_task(f"Looking up Chapter {number}...", total=None)
        
        if not check_neo4j_connection():
            raise typer.Exit(1)
        
        engine = get_query_engine()
        result = engine.ask_about_chapter(number)
        display_result(result, show_sources=sources)


@app.command()
def theme(
    topic: str = typer.Argument(..., help="Theme or concept (e.g., 'obsession', 'whiteness', 'fate')"),
    sources: bool = typer.Option(True, "--sources/--no-sources", help="Show source citations")
):
    """Explore a theme or concept in Moby-Dick."""
    if not check_configuration():
        raise typer.Exit(1)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True
    ) as progress:
        progress.add_task(f"Exploring theme: {topic}...", total=None)
        
        if not check_neo4j_connection():
            raise typer.Exit(1)
        
        engine = get_query_engine()
        result = engine.ask_about_theme(topic)
        display_result(result, show_sources=sources)


@app.command()
def compare(
    first: str = typer.Argument(..., help="First entity to compare"),
    second: str = typer.Argument(..., help="Second entity to compare"),
    sources: bool = typer.Option(True, "--sources/--no-sources", help="Show source citations")
):
    """Compare two characters, concepts, or symbols."""
    if not check_configuration():
        raise typer.Exit(1)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True
    ) as progress:
        progress.add_task(f"Comparing {first} and {second}...", total=None)
        
        if not check_neo4j_connection():
            raise typer.Exit(1)
        
        engine = get_query_engine()
        result = engine.compare(first, second)
        display_result(result, show_sources=sources)


@app.command()
def schema():
    """Show the knowledge graph schema."""
    if not check_configuration():
        raise typer.Exit(1)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True
    ) as progress:
        progress.add_task("Fetching schema...", total=None)
        
        if not check_neo4j_connection():
            raise typer.Exit(1)
        
        client = get_neo4j_client()
        schema = client.get_schema_summary()
    
    console.print()
    console.print(Panel(
        "[bold]🐋 Moby-Dick Knowledge Graph Schema[/bold]",
        border_style="blue"
    ))
    
    # Node types table
    table = Table(title="Node Types", show_header=True, header_style="bold green")
    table.add_column("Label")
    table.add_column("Count", justify="right")
    table.add_column("Layer")
    
    fact_labels = {"Character", "Event", "Location", "Object", "Chapter"}
    for label, count in sorted(schema["node_counts"].items(), key=lambda x: x[1], reverse=True):
        layer = "Facts (L1)" if label in fact_labels else "Analysis (L2)"
        table.add_row(label, str(count), layer)
    
    console.print(table)
    
    # Relationship types
    console.print()
    console.print(f"[bold]Relationship Types:[/bold] {len(schema['relationship_types'])}")
    console.print(f"[dim]{', '.join(sorted(schema['relationship_types'])[:20])}...[/dim]")
    console.print()
    console.print(f"[bold]Total Relationships:[/bold] {schema['total_relationships']:,}")


@app.command()
def interactive():
    """Start an interactive Q&A session."""
    if not check_configuration():
        raise typer.Exit(1)
    
    console.print()
    console.print(Panel(
        "[bold blue]🐋 Moby-Dick GraphRAG Encyclopedia[/bold blue]\n\n"
        "Ask questions about Herman Melville's Moby-Dick!\n"
        "Type [bold]quit[/bold] or [bold]exit[/bold] to leave.\n"
        "Type [bold]help[/bold] for available commands.",
        border_style="blue"
    ))
    
    if not check_neo4j_connection():
        raise typer.Exit(1)
    
    engine = get_query_engine()
    
    while True:
        try:
            console.print()
            question = typer.prompt("🐋 Ask")
            
            if question.lower() in ("quit", "exit", "q"):
                console.print("[dim]Goodbye! May your harpoon strike true. 🐋[/dim]")
                break
            
            if question.lower() == "help":
                console.print("""
[bold]Available Commands:[/bold]
  • Type any question about Moby-Dick
  • [bold]character <name>[/bold] - Get character info
  • [bold]chapter <num>[/bold] - Get chapter summary
  • [bold]theme <topic>[/bold] - Explore a theme
  • [bold]compare <a> <b>[/bold] - Compare two entities
  • [bold]schema[/bold] - Show knowledge graph schema
  • [bold]quit[/bold] / [bold]exit[/bold] - Leave the session
""")
                continue
            
            # Handle special commands in interactive mode
            if question.lower().startswith("character "):
                name = question[10:].strip()
                result = engine.ask_about_character(name)
            elif question.lower().startswith("chapter "):
                try:
                    num = int(question[8:].strip())
                    result = engine.ask_about_chapter(num)
                except ValueError:
                    console.print("[red]Please provide a valid chapter number.[/red]")
                    continue
            elif question.lower().startswith("theme "):
                topic = question[6:].strip()
                result = engine.ask_about_theme(topic)
            elif question.lower().startswith("compare "):
                parts = question[8:].strip().split(" and ")
                if len(parts) != 2:
                    parts = question[8:].strip().split(" vs ")
                if len(parts) != 2:
                    console.print("[red]Use format: compare <entity1> and <entity2>[/red]")
                    continue
                result = engine.compare(parts[0].strip(), parts[1].strip())
            elif question.lower() == "schema":
                schema_cmd = app.registered_commands[5]  # schema command
                continue
            else:
                with console.status("Thinking...", spinner="dots"):
                    result = engine.query(question)
            
            display_result(result, show_sources=True)
            
        except KeyboardInterrupt:
            console.print("\n[dim]Use 'quit' to exit gracefully.[/dim]")
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")


@app.callback()
def main(
    version: bool = typer.Option(False, "--version", "-V", help="Show version")
):
    """🐋 Moby-Dick GraphRAG Encyclopedia
    
    A Gemini-powered knowledge base for Herman Melville's Moby-Dick,
    built on a two-layer Neo4j knowledge graph (facts + analysis).
    """
    if version:
        console.print("Moby-Dick GraphRAG Encyclopedia v0.1.0")
        raise typer.Exit()


def run():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    run()

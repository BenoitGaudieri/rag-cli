"""
rag-cli — Local RAG tool powered by Ollama + ChromaDB + LangChain

Commands:
  index   <path>              Index a file or folder
  query   [question]          Ask a question (omit for interactive mode)
  list                        List indexed collections
  clear                       Delete one or all collections
"""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

app = typer.Typer(
    name="rag",
    help="Local RAG CLI — ask questions about your documents.",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()


# ── index ─────────────────────────────────────────────────────────────────────

@app.command()
def index(
    path: Path = typer.Argument(..., help="PDF file, text file, or folder to index"),
    collection: Optional[str] = typer.Option(
        None, "--collection", "-c", help="Collection name (default: 'default')"
    ),
):
    """Index a document or a whole folder into the vector store."""
    from rag.indexer import index as do_index

    if not path.exists():
        console.print(f"[red]Path not found: {path}[/red]")
        raise typer.Exit(1)

    do_index(path, collection)


# ── query ─────────────────────────────────────────────────────────────────────

@app.command()
def query(
    question: Optional[str] = typer.Argument(
        None, help="Question to ask. Omit to enter interactive mode."
    ),
    collection: Optional[str] = typer.Option(
        None, "--collection", "-c", help="Collection to query (default: 'default')"
    ),
    sources: bool = typer.Option(
        False, "--sources", "-s", help="Show source chunks after the answer"
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Override the LLM model (e.g. mistral, llama3.2)"
    ),
):
    """Ask a question about indexed documents."""
    from rag import config
    from rag.chain import query as do_query

    if model:
        config.LLM_MODEL = model

    if question:
        do_query(question, collection, show_sources=sources)
        return

    # ── interactive REPL ──────────────────────────────────────────────────────
    from rag import config as cfg  # re-import after possible mutation

    console.print(
        Panel(
            f"[bold]RAG — Interactive Mode[/bold]\n"
            f"Collection : [cyan]{collection or cfg.COLLECTION}[/cyan]\n"
            f"LLM        : [cyan]{cfg.LLM_MODEL}[/cyan]\n"
            f"Embeddings : [cyan]{cfg.EMBED_MODEL}[/cyan]\n\n"
            f"[dim]Type a question and press Enter. 'exit' or Ctrl-C to quit.[/dim]",
            border_style="blue",
        )
    )

    while True:
        try:
            q = console.input("\n[bold blue]>[/bold blue] ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not q:
            continue
        if q.lower() in ("exit", "quit", "q", ":q"):
            break
        do_query(q, collection, show_sources=sources)

    console.print("\n[dim]Bye.[/dim]")


# ── list ──────────────────────────────────────────────────────────────────────

@app.command(name="list")
def list_collections():
    """List all indexed collections and their chunk counts."""
    import chromadb
    from rag import config

    if not config.CHROMA_DIR.exists():
        console.print("[yellow]No index found. Run 'index' first.[/yellow]")
        return

    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    collections = client.list_collections()

    if not collections:
        console.print("[yellow]No collections found.[/yellow]")
        return

    console.print("\n[bold]Indexed collections:[/bold]")
    for col in collections:
        console.print(f"  [cyan]{col.name}[/cyan]  — {col.count()} chunks")


# ── clear ─────────────────────────────────────────────────────────────────────

@app.command()
def clear(
    collection: Optional[str] = typer.Option(
        None, "--collection", "-c", help="Collection to delete. Omit to delete all."
    ),
):
    """Delete one collection or the entire index."""
    import chromadb
    from rag import config

    if not config.CHROMA_DIR.exists():
        console.print("[yellow]Nothing to clear.[/yellow]")
        return

    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))

    if collection:
        try:
            client.delete_collection(collection)
            console.print(f"[green]Deleted collection '{collection}'[/green]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
        return

    cols = client.list_collections()
    if not cols:
        console.print("[yellow]Nothing to clear.[/yellow]")
        return

    names = [c.name for c in cols]
    confirmed = typer.confirm(f"Delete ALL collections {names}?", default=False)
    if confirmed:
        for c in cols:
            client.delete_collection(c.name)
        console.print("[green]All collections deleted.[/green]")
    else:
        console.print("[dim]Aborted.[/dim]")


# ── entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app()

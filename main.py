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
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Save answer to file (.txt, .json, .md)"
    ),
    speak: bool = typer.Option(
        False, "--speak", "-S", help="Read the answer aloud using TTS"
    ),
    tts_voice: Optional[str] = typer.Option(
        None, "--voice", "-V", help="TTS voice name (e.g. it-IT-ElsaNeural)"
    ),
):
    """Ask a question about indexed documents."""
    from rag import config
    from rag.chain import query as do_query

    if model:
        config.LLM_MODEL = model

    if question:
        answer = do_query(question, collection, show_sources=sources)
        if output and answer:
            _save_output(output, question, answer)
        if speak and answer:
            _speak_text(answer, voice=tts_voice)
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
        ans = do_query(q, collection, show_sources=sources)
        if speak and ans:
            _speak_text(ans, voice=tts_voice)

    console.print("\n[dim]Bye.[/dim]")


# ── list ──────────────────────────────────────────────────────────────────────

@app.command(name="list")
def list_collections():
    """List all indexed collections and their chunk counts."""
    import json
    from rag import config

    if not config.INDEX_DIR.exists():
        console.print("[yellow]No index found. Run 'index' first.[/yellow]")
        return

    collections = sorted(
        d for d in config.INDEX_DIR.iterdir()
        if d.is_dir() and (d / "index.faiss").exists()
    )

    if not collections:
        console.print("[yellow]No collections found.[/yellow]")
        return

    console.print("\n[bold]Indexed collections:[/bold]")
    for col_dir in collections:
        meta_file = col_dir / "meta.json"
        if meta_file.exists():
            meta = json.loads(meta_file.read_text())
            chunks = meta.get("chunks", "?")
            sources = ", ".join(meta.get("sources", []))
            updated = meta.get("updated", "")[:10]
            console.print(
                f"  [cyan]{col_dir.name}[/cyan]  — {chunks} chunks"
                f"  [dim]({sources}) · {updated}[/dim]"
            )
        else:
            console.print(f"  [cyan]{col_dir.name}[/cyan]")


# ── clear ─────────────────────────────────────────────────────────────────────

@app.command()
def clear(
    collection: Optional[str] = typer.Option(
        None, "--collection", "-c", help="Collection to delete. Omit to delete all."
    ),
):
    """Delete one collection or the entire index."""
    import shutil
    from rag import config

    if not config.INDEX_DIR.exists():
        console.print("[yellow]Nothing to clear.[/yellow]")
        return

    if collection:
        col_dir = config.INDEX_DIR / collection
        if not col_dir.exists():
            console.print(f"[red]Collection '{collection}' not found.[/red]")
            raise typer.Exit(1)
        shutil.rmtree(col_dir)
        console.print(f"[green]Deleted collection '{collection}'[/green]")
        return

    cols = sorted(
        d for d in config.INDEX_DIR.iterdir()
        if d.is_dir() and (d / "index.faiss").exists()
    )
    if not cols:
        console.print("[yellow]Nothing to clear.[/yellow]")
        return

    names = [d.name for d in cols]
    confirmed = typer.confirm(f"Delete ALL collections {names}?", default=False)
    if confirmed:
        shutil.rmtree(config.INDEX_DIR)
        console.print("[green]All collections deleted.[/green]")
    else:
        console.print("[dim]Aborted.[/dim]")


# ── helpers ───────────────────────────────────────────────────────────────────

def _resolve_output(path: Path) -> Path:
    """If path has no parent directory (bare filename), place it inside OUTPUT_DIR."""
    from rag import config
    if path.parent == Path("."):
        out = config.OUTPUT_DIR / path
    else:
        out = path
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def _speak_text(text: str, voice: Optional[str] = None) -> None:
    """Speak *text* using TTS, handling errors gracefully."""
    from rag import config
    from rag.tts import speak as do_speak

    try:
        with console.status("[dim]Synthesizing audio…[/dim]"):
            do_speak(text, voice=voice, max_chars=config.TTS_MAX_CHARS)
    except RuntimeError as e:
        console.print(f"[yellow]TTS unavailable: {e}[/yellow]")
    except KeyboardInterrupt:
        console.print("\n[dim]TTS stopped.[/dim]")


def _save_output(path: Path, question: str, answer: str) -> None:
    """Write a single Q/A pair to disk in the format implied by the file extension."""
    from rag import config
    dest = _resolve_output(path)
    suffix = dest.suffix.lower()
    if suffix == ".json":
        import json
        data = {
            "question": question,
            "answer": answer,
            "collection": config.COLLECTION,
            "model": config.LLM_MODEL,
        }
        dest.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    elif suffix == ".md":
        dest.write_text(f"## Q\n\n{question}\n\n## A\n\n{answer}\n", encoding="utf-8")
    else:
        dest.write_text(f"Q: {question}\n\nA: {answer}\n", encoding="utf-8")
    console.print(f"\n[dim]Saved → {dest}[/dim]")


# ── speak ─────────────────────────────────────────────────────────────────────

@app.command()
def speak(
    source: str = typer.Argument(
        ..., help="Text to read aloud, or path to a file (.txt, .md, .pdf, .json)"
    ),
    voice: Optional[str] = typer.Option(
        None, "--voice", "-v", help="TTS voice name (default from config/RAG_TTS_VOICE)"
    ),
    save: Optional[Path] = typer.Option(
        None, "--save", help="Save audio to an MP3 file instead of playing"
    ),
    max_chars: int = typer.Option(
        0, "--max-chars", help="Truncate text to N characters (0 = no limit)"
    ),
):
    """Read text or a document aloud using TTS (edge-tts neural voices)."""
    from rag import config
    from rag.tts import speak as do_speak, extract_text

    src = Path(source)
    if src.exists() and src.is_file():
        try:
            text = extract_text(src)
        except Exception as e:
            console.print(f"[red]Could not read file: {e}[/red]")
            raise typer.Exit(1)
        console.print(f"[dim]Reading {src.name} ({len(text):,} chars)…[/dim]")
    else:
        text = source

    if not text.strip():
        console.print("[yellow]Nothing to read.[/yellow]")
        return

    limit = max_chars or config.TTS_MAX_CHARS
    if limit and len(text) > limit:
        console.print(f"[dim]Truncating to {limit:,} chars.[/dim]")

    dest = _resolve_output(save) if save else None
    if dest:
        console.print(f"[dim]Saving audio to {dest}…[/dim]")

    try:
        do_speak(text, voice=voice, save_to=dest, max_chars=limit)
        if dest:
            console.print(f"[green]Saved → {dest}[/green]")
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    except KeyboardInterrupt:
        console.print("\n[dim]Stopped.[/dim]")


# ── compare ───────────────────────────────────────────────────────────────────

@app.command()
def compare(
    question_or_file: str = typer.Argument(
        ...,
        help="Question to ask, or path to a .txt file with one question per line",
    ),
    models: str = typer.Option(
        ..., "--models", "-m",
        help="Comma-separated model names, e.g. 'llama3.2,mistral,phi3'",
    ),
    collection: Optional[str] = typer.Option(
        None, "--collection", "-c", help="Collection to query (default: 'default')"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Save results to file (.csv or .json)"
    ),
):
    """Run the same question(s) against multiple models and compare outputs."""
    import csv
    import json
    from rag import config
    from rag.chain import run_silent

    model_list = [m.strip() for m in models.split(",") if m.strip()]
    if not model_list:
        console.print("[red]No models specified.[/red]")
        raise typer.Exit(1)

    # Accept a question string or a file of questions (one per line)
    source = Path(question_or_file)
    if source.exists() and source.is_file():
        questions = [q.strip() for q in source.read_text(encoding="utf-8").splitlines() if q.strip()]
    else:
        questions = [question_or_file]

    collection = collection or config.COLLECTION
    results: list[dict] = []

    for question in questions:
        console.print(f"\n[bold blue]Q:[/bold blue] {question}")
        for model in model_list:
            with console.status(f"[dim]{model}…[/dim]"):
                try:
                    answer, elapsed = run_silent(question, collection, model=model)
                except Exception as e:
                    answer, elapsed = f"ERROR: {e}", 0.0
            preview = answer[:200] + ("…" if len(answer) > 200 else "")
            console.print(f"  [cyan]{model}[/cyan] [dim]({elapsed:.1f}s)[/dim]  {preview}")
            results.append({
                "question": question,
                "model": model,
                "answer": answer,
                "latency_s": round(elapsed, 2),
            })

    if not output:
        return

    dest = _resolve_output(output)
    suffix = dest.suffix.lower()
    if suffix == ".json":
        dest.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        # Default: CSV
        with dest.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["question", "model", "answer", "latency_s"])
            writer.writeheader()
            writer.writerows(results)

    console.print(f"\n[bold green]✓ Results saved to {dest}[/bold green]")


# ── entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app()

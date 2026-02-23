from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from rich.console import Console

from . import config

console = Console()

# Supported extensions → loader class
LOADERS = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".md": TextLoader,
    ".docx": Docx2txtLoader,
}


def _load_file(path: Path) -> list:
    suffix = path.suffix.lower()
    loader_cls = LOADERS.get(suffix)
    if loader_cls is None:
        console.print(f"  [yellow]Skipping unsupported file: {path.name}[/yellow]")
        return []
    try:
        # TextLoader benefits from autodetect_encoding on Windows
        if loader_cls is TextLoader:
            loader = TextLoader(str(path), autodetect_encoding=True)
        else:
            loader = loader_cls(str(path))
        docs = loader.load()
        console.print(f"  [dim]Loaded[/dim] [bold]{path.name}[/bold] [dim]({len(docs)} chunks)[/dim]")
        return docs
    except Exception as e:
        console.print(f"  [red]Error loading {path.name}: {e}[/red]")
        return []


def load_documents(path: Path) -> list:
    """Load documents from a single file or recursively from a directory."""
    if path.is_file():
        return _load_file(path)

    docs = []
    files = sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in LOADERS)
    if not files:
        console.print(f"[yellow]No supported files found in {path}[/yellow]")
        return []
    for f in files:
        docs.extend(_load_file(f))
    return docs


def index(path: Path, collection: str | None = None) -> int:
    """
    Load, chunk, embed, and store documents from *path*.
    Returns the number of chunks indexed.
    """
    collection = collection or config.COLLECTION

    console.print(f"\n[bold blue]Indexing[/bold blue] {path}")

    docs = load_documents(path)
    if not docs:
        console.print("[red]No documents loaded — nothing to index.[/red]")
        return 0

    console.print(f"\n[green]Loaded {len(docs)} raw chunks[/green]")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        add_start_index=True,
    )
    chunks = splitter.split_documents(docs)
    console.print(f"[green]Split into {len(chunks)} text chunks[/green]")

    console.print(
        f"\n[bold]Embedding with [cyan]{config.EMBED_MODEL}[/cyan] "
        f"(this may take a while the first time)...[/bold]"
    )

    embeddings = OllamaEmbeddings(model=config.EMBED_MODEL)

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(config.CHROMA_DIR),
        collection_name=collection,
    )

    console.print(
        f"\n[bold green]✓ Indexed {len(chunks)} chunks[/bold green] "
        f"→ collection [cyan]'{collection}'[/cyan] in [dim]{config.CHROMA_DIR}[/dim]"
    )
    return len(chunks)

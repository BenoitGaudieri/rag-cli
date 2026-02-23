import time

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from rich.console import Console

from . import config

console = Console()

_PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful assistant. Answer the question using ONLY the context provided below.
If the context does not contain enough information, say so clearly — do not make things up.

Context:
{context}

Question: {question}

Answer:"""
)


def _format_docs(docs: list) -> str:
    parts = []
    for doc in docs:
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "")
        loc = f"{src}, p.{page}" if page != "" else src
        parts.append(f"[{loc}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def _build_chain(collection: str):
    collection_dir = config.INDEX_DIR / collection
    if not collection_dir.exists():
        raise FileNotFoundError(
            f"No index found for collection '{collection}'. Run 'index' first."
        )

    embeddings = OllamaEmbeddings(model=config.EMBED_MODEL)
    vectorstore = FAISS.load_local(
        str(collection_dir), embeddings, allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": config.TOP_K, "fetch_k": config.TOP_K * 3},
    )

    llm = ChatOllama(model=config.LLM_MODEL, temperature=0)

    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | _PROMPT
        | llm
        | StrOutputParser()
    )
    return chain, retriever


def query(question: str, collection: str | None = None, show_sources: bool = False) -> str:
    collection = collection or config.COLLECTION

    try:
        chain, retriever = _build_chain(collection)
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        return ""

    console.print(f"\n[bold blue]Q:[/bold blue] {question}\n")
    console.print("[bold green]A:[/bold green] ", end="")

    full_response = ""
    try:
        for chunk in chain.stream(question):
            print(chunk, end="", flush=True)
            full_response += chunk
    except Exception as e:
        console.print(f"\n[red]Error during generation: {e}[/red]")
        return full_response

    print()

    if show_sources:
        docs = retriever.invoke(question)
        console.print("\n[dim]── Sources ──────────────────────────────[/dim]")
        for i, doc in enumerate(docs, 1):
            src = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "")
            loc = f"{src} (p.{page})" if page != "" else src
            preview = doc.page_content[:120].replace("\n", " ")
            console.print(f"  [dim]{i}. {loc}[/dim]")
            console.print(f"     [dim italic]\"{preview}…\"[/dim italic]")

    return full_response


def run_silent(
    question: str, collection: str | None = None, model: str | None = None
) -> tuple[str, float]:
    """
    Run a query without any printed output.
    Returns (answer, elapsed_seconds). Used by the `compare` command.
    """
    collection = collection or config.COLLECTION
    original_model = config.LLM_MODEL
    if model:
        config.LLM_MODEL = model
    try:
        chain, _ = _build_chain(collection)
        start = time.perf_counter()
        result = chain.invoke(question)
        elapsed = time.perf_counter() - start
        return result, elapsed
    finally:
        config.LLM_MODEL = original_model

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
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
    embeddings = OllamaEmbeddings(model=config.EMBED_MODEL)

    vectorstore = Chroma(
        persist_directory=str(config.CHROMA_DIR),
        embedding_function=embeddings,
        collection_name=collection,
    )

    # MMR: balances relevance + diversity across retrieved chunks
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
    """Run a RAG query and stream the answer to stdout."""
    collection = collection or config.COLLECTION

    try:
        chain, retriever = _build_chain(collection)
    except Exception as e:
        console.print(f"[red]Failed to load index: {e}[/red]")
        console.print("[dim]Have you run 'index' first?[/dim]")
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

    print()  # final newline after streamed output

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

# rag-cli

> Ask questions about your documents — entirely on your machine, zero API costs.

`rag-cli` is a command-line RAG (Retrieval-Augmented Generation) tool that lets you index a PDF, a text file, or an entire folder of documents, then query them in natural language using a local LLM. Everything runs locally via [Ollama](https://ollama.com): no cloud, no keys, no data leaving your machine.

```
$ python main.py query "What are the main configuration options?"

Q: What are the main configuration options?

A: According to the documentation, the three main configuration options are...

── Sources ──────────────────────────────
  1. docs/manual.pdf (p.12)
     "Configuration is handled through environment variables or a .env file..."
  2. docs/manual.pdf (p.14)
     "Advanced options can be set at runtime via the --model flag..."
```

---

## Stack

| Layer | Technology |
|---|---|
| LLM & Embeddings | [Ollama](https://ollama.com) (`llama3.2`, `nomic-embed-text`) |
| RAG Framework | [LangChain](https://python.langchain.com/) 1.x (LCEL) |
| Vector Store | [FAISS](https://github.com/facebookresearch/faiss) — persistent, local, no server needed |
| Document Parsing | `pypdf`, `docx2txt`, built-in text loaders |
| CLI | [Typer](https://typer.tiangolo.com/) + [Rich](https://rich.readthedocs.io/) |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                           rag-cli                               │
│                                                                 │
│  INDEX                              QUERY                       │
│                                                                 │
│  PDF / TXT / MD / DOCX              natural language question   │
│         │                                    │                  │
│         ▼                                    ▼                  │
│   Document Loader               OllamaEmbeddings (nomic)        │
│         │                                    │                  │
│         ▼                                    ▼                  │
│  RecursiveCharacter               FAISS MMR Retriever           │
│    TextSplitter                   (top-k relevant chunks)       │
│         │                                    │                  │
│         ▼                                    ▼                  │
│  OllamaEmbeddings  ──────►   FAISS     ChatPromptTemplate       │
│   (nomic-embed-text)        (on disk)        │                  │
│                                               ▼                 │
│                                       ChatOllama (llama3.2)     │
│                                               │                 │
│                                               ▼                 │
│                                       streamed answer           │
└─────────────────────────────────────────────────────────────────┘
```

**Retrieval strategy:** MMR (Maximum Marginal Relevance) — retrieved chunks are ranked by relevance to the query *and* diversity from each other, reducing redundancy in the context window.

---

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/download) installed and running

Pull the required models once:

```bash
ollama pull nomic-embed-text   # ~274 MB — embedding model
ollama pull llama3.2           # ~2 GB  — default LLM (or swap for mistral, etc.)
```

---

## Installation

```bash
git clone https://github.com/BenoitGaudieri/rag-cli
cd rag-cli

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

---

## Usage

### Index documents

```bash
# Single file
python main.py index ./docs/report.pdf

# Entire folder (PDF, TXT, MD, DOCX — recursive)
python main.py index ./docs/

# Named collection (to keep multiple knowledge bases separate)
python main.py index ./docs/ --collection myproject
```

### Query

```bash
# One-shot question
python main.py query "Summarise the key findings"

# Show the source chunks used to generate the answer
python main.py query "What are the installation steps?" --sources

# Interactive REPL — ask multiple questions in a session
python main.py query

# Override the LLM at runtime
python main.py query "Translate chapter 1 to Italian" --model mistral

# Query a named collection
python main.py query "..." --collection myproject
```

### Manage collections

```bash
python main.py list                          # list all collections + chunk counts
python main.py clear --collection myproject  # delete one collection
python main.py clear                         # delete everything
```

---

## Configuration

All defaults can be overridden via environment variables (or a `.env` file):

| Variable | Default | Description |
|---|---|---|
| `RAG_LLM_MODEL` | `llama3.2` | Ollama model for generation |
| `RAG_EMBED_MODEL` | `nomic-embed-text` | Ollama model for embeddings |
| `RAG_COLLECTION` | `default` | FAISS collection name |
| `RAG_INDEX_DIR` | `./faiss_db` | Persistence directory |
| `RAG_CHUNK_SIZE` | `1000` | Characters per text chunk |
| `RAG_CHUNK_OVERLAP` | `200` | Overlap between consecutive chunks |
| `RAG_TOP_K` | `5` | Number of chunks retrieved per query |

Example:

```bash
RAG_LLM_MODEL=mistral RAG_TOP_K=8 python main.py query "..."
```

---

## Project structure

```
rag-cli/
├── main.py          # CLI entry point — index / query / list / clear
├── rag/
│   ├── config.py    # all parameters, overridable via env vars
│   ├── indexer.py   # document loading, chunking, embedding → FAISS
│   └── chain.py     # LCEL RAG chain, MMR retriever, streaming output
├── requirements.txt
└── faiss_db/        # auto-created on first index (add to .gitignore)
```

---

## Supported file types

| Extension | Loader |
|---|---|
| `.pdf` | `PyPDFLoader` (pypdf) |
| `.txt` | `TextLoader` (UTF-8 autodetect) |
| `.md` | `TextLoader` |
| `.docx` | `Docx2txtLoader` |

---

## License

MIT

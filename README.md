# LawPak — AI-Powered Legal Research Assistant

LawPak is a final year project (FYP) that helps legal researchers, lawyers, and
students search, summarize, and explore Pakistani case law using a
Retrieval-Augmented Generation (RAG) pipeline combined with a legal knowledge
graph.

The system ingests court judgments, extracts structured legal entities
(judges, parties, sections, decisions), builds a searchable vector index over
case summaries, and answers natural-language legal queries using a locally
hosted LLM (Llama 3.1 via Ollama) — so no case data ever leaves the local
machine.

## Features

- **Conversational legal Q&A** over a corpus of court judgments, grounded in
  retrieved case text (RAG) rather than the model's raw memory.
- **Legal knowledge graph** (Neo4j) linking cases to judges, parties, statute
  sections, and decisions for structured queries and visual exploration
  (`kg.html`).
- **Document upload & analysis** — users can upload their own legal documents
  (PDF/TXT/DOCX), which are chunked, embedded, and summarized on the fly.
- **User accounts & session-based usage limits** via a Flask + SQLite
  backend with JWT authentication.
- **Local-first LLM inference** — runs entirely on Ollama (`llama3.1:8b` by
  default), with no dependency on a paid cloud LLM API.
- **Streamlit prototype** (`Web/final.py`) for rapid experimentation, alongside
  the production Flask API (`Web/backend/api_server.py`).

## Architecture

```
                ┌──────────────┐
   User query → │   Flask API   │ → JWT auth, usage limits, document upload
                │ (api_server)  │
                └───────┬──────┘
                        │
        ┌───────────────┼────────────────┐
        ▼                                ▼
┌───────────────┐                ┌───────────────┐
│  FAISS vector  │               │  Neo4j graph   │
│  store (RAG)   │               │  (entities &   │
│  case summaries│               │  relationships)│
└───────┬───────┘                └───────┬───────┘
        │                                │
        └───────────────┬────────────────┘
                         ▼
                 ┌───────────────┐
                 │ Ollama (local) │
                 │  Llama 3.1 8B  │
                 └───────────────┘
```

## Tech Stack

| Layer | Technology |
|---|---|
| Backend API | Flask, Flask-CORS, SQLite |
| Auth | JWT (PyJWT), bcrypt |
| LLM inference | Ollama (Llama 3.1 8B), local — no API key needed |
| Embeddings | HuggingFace Sentence-Transformers |
| Vector search | FAISS |
| Knowledge graph | Neo4j, PyVis (visualization) |
| Document parsing | pdfplumber, pytesseract (OCR), python-docx |
| Prototyping UI | Streamlit |
| Frontend | HTML/CSS/JS (`Web/static`) |

## Project Structure

```
.
├── Web/
│   ├── api.py                  # Lightweight local dev server entry point
│   ├── final.py                 # Streamlit prototype app
│   ├── backend_wrapper.py       # Bridges Streamlit UI to the RAG backend
│   ├── backend/
│   │   ├── api_server.py        # Main Flask API (production-style backend)
│   │   ├── auth.py              # JWT auth, password hashing
│   │   ├── database.py          # SQLite schema & queries
│   │   ├── document_processor.py # Upload → chunk → embed → summarize pipeline
│   │   └── user_rag.py          # Per-user document RAG
│   └── static/                  # Frontend HTML/CSS/JS + brand assets
├── scripts/                     # Offline data pipeline (run once to build indexes)
│   ├── extract_text.py          # PDF/text extraction from raw judgments
│   ├── clean_text.py            # Text cleaning
│   ├── chunk_docs.py            # Chunking for embeddings
│   ├── build_vector_store.py    # Builds FAISS index over chunks
│   ├── build_faiss_summaries.py # Builds FAISS index over case summaries
│   ├── summarize_llama_fast.py  # LLM-based case summarization
│   ├── build_full_kg.py         # Builds the Neo4j knowledge graph
│   ├── insert_summaries_into_kg.py
│   ├── merge_summaries.py
│   ├── LawPak.py / testfinal.py / auto_rag.py # RAG pipeline experiments
│   └── metric.py                # Evaluation against ground truth
├── knowledge_graph/
│   └── triplestoGraphLLMData.py # Loads extracted triples into Neo4j
├── kg.html                      # Standalone knowledge graph visualizer
├── requirements.txt
├── .env.example
├── start_local_windows.bat
└── stop_local_windows.bat
```

> **Note:** Generated artifacts — FAISS indexes (`vector_store/`,
> `vector_store_summaries/`), the SQLite database, and the extracted triples
> JSON — are not committed to this repository (see
> [Regenerating data](#regenerating-data) below). This keeps the repo small
> and avoids shipping any case data or user data collected during
> development/testing.

## Getting Started

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed locally, with the model pulled:
  ```bash
  ollama pull llama3.1:8b
  ```
- [Neo4j](https://neo4j.com/download/) (Desktop or Community Server) running
  locally, if you want to use the knowledge graph features.
- Tesseract OCR installed locally if you plan to process scanned PDFs
  (`pytesseract` depends on the system `tesseract` binary).

### Installation

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Configuration

```bash
cp .env.example .env
# then edit .env with your Neo4j password and any other local settings
```

Export the variables in your shell (or use `python-dotenv` if you prefer to
load `.env` automatically):

```bash
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=your_password
export OLLAMA_URL=http://localhost:11434/api/generate
export OLLAMA_MODEL=llama3.1:8b
export LAWPAKAI_SECRET_KEY=your_long_random_string
```

### Running the app

**Flask backend (recommended):**
```bash
python Web/backend/api_server.py
```
The API will be available at `http://localhost:5001`.

**Streamlit prototype:**
```bash
streamlit run Web/final.py
```

**Windows convenience scripts:**
```bat
start_local_windows.bat
stop_local_windows.bat
```

### Regenerating data

The vector indexes and knowledge graph are built from raw case documents,
which are not included in this repo. To rebuild them from your own corpus:

```bash
python scripts/extract_text.py
python scripts/clean_text.py
python scripts/chunk_docs.py
python scripts/build_vector_store.py
python scripts/build_faiss_summaries.py
python scripts/summarize_llama_fast.py
python scripts/build_full_kg.py
```

## Evaluation

`scripts/metric.py` compares retrieved knowledge-graph data against a
ground-truth set of cases to measure extraction accuracy.

## License

This project was developed as a Final Year Project (FYP). Add a license here
(e.g. MIT) if you intend to make the code reusable by others.

## Acknowledgements

Built using open-source tools including LangChain, FAISS, Neo4j, Streamlit,
and Meta's Llama models served locally via Ollama.

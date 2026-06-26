"""
Enhanced document processing pipeline for user-uploaded legal files.

Supported formats: PDF, DOCX, TXT
Flow:
    file bytes → extract text → clean → chunk (8K chars)
    → LLaMA summarize per chunk → LLaMA entity extraction
    → BGE embed chunks + summaries → dual FAISS index

Storage per document:
    backend/user_uploads/{user_id}/{doc_id}/
        original.<ext>           raw file preserved
        status.json              progress tracking
        index_chunks.faiss       FAISS IndexFlatIP (768-dim) for raw chunks
        chunks.pkl               [{"chunk": str, "chunk_id": int}] + metadata
        index_summaries.faiss    FAISS IndexFlatIP (768-dim) for LLaMA summaries
        summaries.pkl            [{"summary": str, "chunk_id": int}] + metadata
        entities.json            judges, parties, laws, decision extracted by LLaMA
"""

import io
import hashlib
import json
import os
import pickle
import re
import shutil
import time
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import requests

try:
    import faiss
except ModuleNotFoundError:
    faiss = None

UPLOADS_DIR = Path(__file__).parent / "user_uploads"
BGE_MODEL = "BAAI/bge-base-en-v1.5"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")

# BGE works best with focused passages; LLaMA summaries can use larger windows.
RETRIEVAL_CHUNK_CHARS = int(os.environ.get("UPLOAD_RETRIEVAL_CHUNK_CHARS", "2800"))
RETRIEVAL_OVERLAP_CHARS = int(os.environ.get("UPLOAD_RETRIEVAL_OVERLAP_CHARS", "350"))
SUMMARY_CHUNK_CHARS = int(os.environ.get("UPLOAD_SUMMARY_CHUNK_CHARS", "9000"))
SUMMARY_OVERLAP_CHARS = int(os.environ.get("UPLOAD_SUMMARY_OVERLAP_CHARS", "700"))
SCORE_THRESHOLD = float(os.environ.get("UPLOAD_RAG_SCORE_THRESHOLD", "0.18"))
OLLAMA_SUMMARY_TIMEOUT = int(os.environ.get("OLLAMA_SUMMARY_TIMEOUT", "75"))
OLLAMA_ENTITY_TIMEOUT = int(os.environ.get("OLLAMA_ENTITY_TIMEOUT", "60"))

_embedder = None


# ── Status tracking ───────────────────────────────────────────────────────────

def _write_status(doc_dir: Path, status: str, progress: int, stage: str, error: str = None):
    try:
        (doc_dir / "status.json").write_text(json.dumps({
            "status": status,
            "progress": progress,
            "stage": stage,
            "error": error,
        }))
    except Exception:
        pass


def read_doc_status(user_id: str, doc_id: str) -> dict:
    p = UPLOADS_DIR / user_id / doc_id / "status.json"
    if not p.exists():
        return {"status": "unknown", "progress": 0, "stage": "", "error": None}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {"status": "unknown", "progress": 0, "stage": "", "error": None}


# ── Embedder ──────────────────────────────────────────────────────────────────

def _get_embedder():
    global _embedder
    if _embedder is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embedder = SentenceTransformer(BGE_MODEL, device="cpu")
            print(f"  [embed] using {BGE_MODEL}", flush=True)
        except Exception as exc:
            print(f"  [embed] {BGE_MODEL} unavailable ({exc}); using lexical fallback", flush=True)
            _embedder = _LexicalEmbedder()
    return _embedder


class _LexicalEmbedder:
    """Dependency-free fallback with the same encode() shape as SentenceTransformer."""

    dim = 768

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False, batch_size=32):
        vectors = []
        for text in texts:
            vec = np.zeros(self.dim, dtype=np.float32)
            tokens = re.findall(r"[a-zA-Z0-9]+", (text or "").lower())
            for token in tokens:
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                raw = int.from_bytes(digest, "little")
                idx = raw % self.dim
                sign = 1.0 if (raw >> 8) & 1 else -1.0
                vec[idx] += sign
            if normalize_embeddings:
                norm = float(np.linalg.norm(vec))
                if norm > 0:
                    vec /= norm
            vectors.append(vec)
        return np.vstack(vectors) if vectors else np.zeros((0, self.dim), dtype=np.float32)


# ── Text Extraction ───────────────────────────────────────────────────────────

def _extract_pdf(file_bytes: bytes) -> tuple:
    """Returns (text, page_count). Raises ValueError if no text found."""
    import pdfplumber
    pages = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        page_count = len(pdf.pages)
        for page in pdf.pages:
            t = page.extract_text()
            if t and t.strip():
                pages.append(t.strip())
    text = "\n\n".join(pages)
    return text, page_count


def _extract_docx(file_bytes: bytes) -> tuple:
    """Returns (text, 0)."""
    import docx
    doc = docx.Document(io.BytesIO(file_bytes))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    text = "\n\n".join(paragraphs)
    return text, 0


def _extract_txt(file_bytes: bytes) -> tuple:
    """Returns (text, 0). Tries common encodings."""
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return file_bytes.decode(enc), 0
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not decode text file — unknown encoding")


def extract_text(file_bytes: bytes, filename: str) -> tuple:
    """
    Dispatch to the correct extractor based on file extension.
    Returns (text: str, page_count: int).
    """
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext == "pdf":
        text, pages = _extract_pdf(file_bytes)
    elif ext == "docx":
        text, pages = _extract_docx(file_bytes)
    elif ext in ("txt", "text"):
        text, pages = _extract_txt(file_bytes)
    else:
        raise ValueError(f"Unsupported file type: .{ext}")

    if not text.strip():
        raise ValueError(
            "No text could be extracted. PDF may be a scanned image."
            if ext == "pdf"
            else "File appears to be empty."
        )
    return text, pages


# ── Text Cleaning ─────────────────────────────────────────────────────────────

_HEADER_PATTERNS = [
    r"Page\s+\d+\s+of\s+\d+",
    r"^\s*Page\s+\d+\s*$",
    r"^\s*\d+\s*$",          # standalone page numbers
]


def _clean_text(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if any(re.search(pat, line, flags=re.IGNORECASE) for pat in _HEADER_PATTERNS):
            continue
        lines.append(line)
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


# ── Chunking ──────────────────────────────────────────────────────────────────

def _chunk_text(
    text: str,
    chunk_chars: int = RETRIEVAL_CHUNK_CHARS,
    overlap_chars: int = RETRIEVAL_OVERLAP_CHARS,
) -> list:
    """
    Sentence-aware sliding window chunker.
    Uses configurable windows so retrieval chunks stay focused while summary
    chunks can preserve larger legal sections.
    """
    text = "\n".join(line.strip() for line in text.splitlines()).strip()
    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_chars, length)

        if end < length:
            snap_start = start + int(chunk_chars * 0.72)
            for sep in (". ", "! ", "? ", "\n\n", "\n"):
                pos = text.rfind(sep, snap_start, end)
                if pos != -1:
                    end = pos + len(sep)
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= length:
            break

        start = max(start + 1, end - overlap_chars)

    return chunks


# ── LLaMA Summarization ───────────────────────────────────────────────────────

def _ollama_base_url() -> str:
    parsed = urlsplit(OLLAMA_URL)
    return urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))


def _ollama_available(timeout: int = 3) -> bool:
    try:
        r = requests.get(f"{_ollama_base_url()}/api/tags", timeout=timeout)
        return r.ok
    except Exception:
        return False


def _ollama_generate(prompt: str, timeout: int, num_predict: int, temperature: float = 0.1) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
            "num_ctx": 8192,
        },
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "").strip()


def _extractive_summary(text: str, max_sentences: int = 8) -> str:
    """Fast fallback when Ollama is unavailable or too slow."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return "[Summary unavailable: no extracted text]"

    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    priority_words = (
        "petitioner", "appellant", "respondent", "judge", "justice",
        "section", "ppc", "crpc", "constitution", "evidence", "bail",
        "appeal", "petition", "conviction", "acquittal", "dismissed",
        "allowed", "sentence", "decision", "order", "held",
    )
    picked = []
    for sentence in sentences:
        lower = sentence.lower()
        if any(word in lower for word in priority_words):
            picked.append(sentence.strip())
        if len(picked) >= max_sentences:
            break
    if len(picked) < 4:
        picked = [s.strip() for s in sentences[:max_sentences] if s.strip()]

    return "\n".join(f"- {s[:450]}" for s in picked[:max_sentences])


def _summarize_chunk(text: str, chunk_idx: int, retries: int = 1) -> str:
    """Call Ollama to produce a 6–9 line summary of one chunk."""
    prompt = (
        "You are summarizing a Pakistani legal judgment section for a RAG system.\n"
        "Write 6-9 concise lines. Include any names of judges, petitioner/appellant, "
        "respondent/state, legal sections/laws, material facts, arguments, reasoning, "
        "and final order/decision mentioned in this section. Do not invent missing facts.\n\n"
        f"TEXT:\n{text[:8000]}\n\nSUMMARY:\n"
    )
    for attempt in range(retries):
        try:
            summary = _ollama_generate(
                prompt,
                timeout=OLLAMA_SUMMARY_TIMEOUT,
                num_predict=450,
                temperature=0.1,
            )
            if len(summary) >= 30:
                return summary
            time.sleep(2)
        except Exception as e:
            print(f"  [summarize] chunk {chunk_idx} attempt {attempt + 1}: {e}", flush=True)
            time.sleep(1)
    return _extractive_summary(text)


def _summarize_document(section_summaries: list, text: str) -> str:
    joined = "\n\n".join(
        f"Section {i + 1}:\n{s}" for i, s in enumerate(section_summaries) if s
    )
    if not joined:
        return _extractive_summary(text, max_sentences=10)

    prompt = (
        "Create one overall case brief from these section summaries.\n"
        "Use this exact structure:\n"
        "Case Background:\nParties and Judges:\nImportant Laws/Sections:\nKey Facts:\n"
        "Court Reasoning:\nFinal Decision:\n\n"
        f"SECTION SUMMARIES:\n{joined[:10000]}\n\nCASE BRIEF:\n"
    )
    try:
        summary = _ollama_generate(
            prompt,
            timeout=OLLAMA_SUMMARY_TIMEOUT,
            num_predict=700,
            temperature=0.05,
        )
        if len(summary) >= 50:
            return summary
    except Exception as e:
        print(f"  [document-summary] fallback: {e}", flush=True)
    return _extractive_summary(text, max_sentences=10)


# ── Entity Extraction ─────────────────────────────────────────────────────────

def _heuristic_entities(text: str) -> dict:
    head = text[:7000]
    tail = text[-5000:] if len(text) > 5000 else text
    both = f"{head}\n{tail}"

    laws = sorted(set(
        re.findall(
            r"\b(?:section|s\.|u/s)\s*\d+[A-Za-z/-]*(?:\s*(?:PPC|Cr\.?P\.?C\.?|QSO|Constitution|Control of Narcotic Substances Act))?",
            both,
            flags=re.IGNORECASE,
        )
        + re.findall(r"\b(?:PPC|Cr\.?P\.?C\.?|QSO)\s*(?:section\s*)?\d+[A-Za-z/-]*", both, flags=re.IGNORECASE)
    ))

    judges = []
    for pat in (
        r"(?:before|present)\s*:\s*([A-Z][A-Za-z .,'-]{4,120})",
        r"(?:Mr\.?\s+Justice|Justice)\s+([A-Z][A-Za-z .,'-]{3,80})",
        r"\b([A-Z][A-Za-z .,'-]{3,80}),\s*J\.",
    ):
        for m in re.finditer(pat, head, flags=re.IGNORECASE):
            name = re.sub(r"\s+", " ", m.group(1)).strip(" .,:;-")
            if name and name.lower() not in {j.lower() for j in judges}:
                judges.append(name)

    petitioner = ""
    respondent = ""
    party_match = re.search(
        r"([A-Z][^\n]{2,120}?)\s+(?:v\.?|versus|vs\.?)\s+([A-Z][^\n]{2,120})",
        head,
        flags=re.IGNORECASE,
    )
    if party_match:
        petitioner = re.sub(r"\s+", " ", party_match.group(1)).strip(" .,:;-")
        respondent = re.sub(r"\s+", " ", party_match.group(2)).strip(" .,:;-")

    decision = "unknown"
    tail_l = tail.lower()
    decision_patterns = [
        ("acquitted", ("acquitted", "acquittal")),
        ("convicted", ("convicted", "conviction maintained")),
        ("dismissed", ("dismissed", "petition is dismissed", "appeal is dismissed")),
        ("allowed", ("allowed", "petition is allowed", "appeal is allowed")),
        ("remanded", ("remanded", "case is remanded")),
    ]
    for label, needles in decision_patterns:
        if any(n in tail_l for n in needles):
            decision = label
            break

    case_type = ""
    m = re.search(r"\b(criminal|civil|constitutional|writ|jail|murder|bail)[^\n]{0,60}(appeal|petition|revision|case|application)\b", head, flags=re.IGNORECASE)
    if m:
        case_type = re.sub(r"\s+", " ", m.group(0)).strip()

    return {
        "judges": judges[:5],
        "petitioner": petitioner[:180],
        "respondent": respondent[:180],
        "laws": laws[:20],
        "decision": decision,
        "case_type": case_type,
    }


def _extract_entities(text: str, retries: int = 1) -> dict:
    """
    Use LLaMA to extract structured legal entities from the document.
    Samples the beginning (parties/judges) and end (final decision).
    """
    excerpt = text[:6000]
    if len(text) > 6000:
        excerpt += "\n\n...\n\n" + text[-2000:]

    prompt = (
        "Extract key legal entities from this court judgment or legal document.\n"
        "Return ONLY a valid JSON object with exactly these keys:\n"
        '{"judges": ["name1", "name2"], '
        '"petitioner": "full name or organization", '
        '"respondent": "full name or organization", '
        '"laws": ["PPC Section 302", "CrPC 265-K"], '
        '"decision": "acquitted|convicted|dismissed|allowed|remanded|other|unknown", '
        '"case_type": "brief description e.g. criminal appeal, murder case"}\n\n'
        f"DOCUMENT EXCERPT:\n{excerpt}\n\nJSON:"
    )
    default = _heuristic_entities(text)
    for attempt in range(retries):
        try:
            raw = _ollama_generate(
                prompt,
                timeout=OLLAMA_ENTITY_TIMEOUT,
                num_predict=350,
                temperature=0.0,
            )
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                parsed = json.loads(m.group(0))
                return {**default, **parsed}
            time.sleep(2)
        except Exception as e:
            print(f"  [entities] attempt {attempt + 1}: {e}", flush=True)
            time.sleep(1)
    return default


# ── Embedding & Indexing ──────────────────────────────────────────────────────

def _build_vector_index(texts: list):
    model = _get_embedder()
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=32,
    )
    embeddings = np.array(embeddings, dtype=np.float32)
    if faiss is None:
        return embeddings
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def _save_vector_index(index, faiss_path: Path, npy_path: Path):
    if faiss is not None and hasattr(index, "add"):
        faiss.write_index(index, str(faiss_path))
    else:
        np.save(str(npy_path), index)


def _load_vector_index(faiss_path: Path, npy_path: Path):
    if faiss is not None and faiss_path.exists():
        return faiss.read_index(str(faiss_path))
    if npy_path.exists():
        return np.load(str(npy_path))
    if faiss_path.exists() and faiss is None:
        raise RuntimeError("FAISS index exists but faiss is not installed in this Python environment")
    raise FileNotFoundError(f"No vector index found at {faiss_path} or {npy_path}")


def _search_vector_index(index, q_emb: np.ndarray, top_k: int):
    if top_k <= 0:
        return np.array([[]], dtype=np.float32), np.array([[]], dtype=np.int64)
    if faiss is not None and hasattr(index, "search"):
        return index.search(q_emb, top_k)
    scores = np.dot(index, q_emb[0])
    if scores.size == 0:
        return np.array([[]], dtype=np.float32), np.array([[]], dtype=np.int64)
    top = np.argsort(scores)[::-1][:top_k]
    return scores[top][None, :].astype(np.float32), top[None, :].astype(np.int64)


# ── Public API ────────────────────────────────────────────────────────────────

def process_document(user_id: str, doc_id: str, file_bytes: bytes, filename: str) -> dict:
    """
    Full enhanced pipeline:
        extract → clean → chunk → LLaMA summarize each chunk
        → LLaMA entity extract → BGE embed chunks + summaries → dual FAISS

    Writes status.json at each stage so the frontend can show progress.
    Returns: {"doc_id", "chunk_count", "char_count", "page_count"}
    """
    doc_dir = UPLOADS_DIR / user_id / doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)
    _write_status(doc_dir, "processing", 2, "starting")

    try:
        # 1. Save original file
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "bin"
        (doc_dir / f"original.{ext}").write_bytes(file_bytes)

        # 2. Extract text
        _write_status(doc_dir, "processing", 5, "extracting")
        text, page_count = extract_text(file_bytes, filename)

        # 3. Clean
        _write_status(doc_dir, "processing", 12, "cleaning")
        text = _clean_text(text)

        # 4. Chunk into focused retrieval passages plus larger summary sections.
        _write_status(doc_dir, "processing", 18, "chunking")
        chunks = _chunk_text(
            text,
            chunk_chars=RETRIEVAL_CHUNK_CHARS,
            overlap_chars=RETRIEVAL_OVERLAP_CHARS,
        )
        summary_chunks = _chunk_text(
            text,
            chunk_chars=SUMMARY_CHUNK_CHARS,
            overlap_chars=SUMMARY_OVERLAP_CHARS,
        )
        if not chunks:
            raise ValueError("Document produced no text chunks after processing")

        # 5. Embed raw chunks early. This is the most important search index.
        _write_status(doc_dir, "processing", 28, "embedding_chunks")
        chunk_index = _build_vector_index(chunks)

        # 6. Summarize larger sections via LLaMA. If Ollama is down, use a
        # fast extractive fallback so uploads do not remain stuck for 10+ minutes.
        ollama_ok = _ollama_available()
        summaries = []
        n = len(summary_chunks)
        _write_status(doc_dir, "processing", 38, f"summarizing chunk 0/{n}")
        if not ollama_ok:
            print(f"  [doc] {doc_id} Ollama unavailable; using extractive summaries", flush=True)
        for i, chunk in enumerate(summary_chunks):
            progress = 38 + int(32 * (i / max(n, 1)))
            _write_status(doc_dir, "processing", progress, f"summarizing chunk {i + 1}/{n}")
            summary = _summarize_chunk(chunk, i) if ollama_ok else _extractive_summary(chunk)
            summaries.append(summary)
            print(f"  [doc] {doc_id} summarized section {i + 1}/{n}", flush=True)

        _write_status(doc_dir, "processing", 72, "merging_summaries")
        document_summary = _summarize_document(summaries, text) if ollama_ok else _extractive_summary(text, max_sentences=10)
        (doc_dir / "document_summary.json").write_text(
            json.dumps({"summary": document_summary}, ensure_ascii=False, indent=2)
        )

        # 7. Extract entities via LLaMA plus regex fallback.
        _write_status(doc_dir, "processing", 78, "extracting_entities")
        entities = _extract_entities(text) if ollama_ok else _heuristic_entities(text)
        (doc_dir / "entities.json").write_text(
            json.dumps(entities, ensure_ascii=False, indent=2)
        )
        print(f"  [doc] {doc_id} entities: {entities}", flush=True)

        # 8. Embed summaries
        _write_status(doc_dir, "processing", 88, "embedding_summaries")
        summary_records = [
            {"summary": document_summary, "chunk_id": -1, "kind": "document_summary"}
        ] + [
            {"summary": s, "chunk_id": i, "kind": "section_summary"}
            for i, s in enumerate(summaries)
        ]
        summary_index = _build_vector_index([s["summary"] for s in summary_records])

        # 9. Persist everything
        _write_status(doc_dir, "processing", 96, "saving")
        _save_vector_index(
            chunk_index,
            doc_dir / "index_chunks.faiss",
            doc_dir / "index_chunks.npy",
        )
        _save_vector_index(
            summary_index,
            doc_dir / "index_summaries.faiss",
            doc_dir / "index_summaries.npy",
        )

        chunk_records = [{"chunk": c, "chunk_id": i} for i, c in enumerate(chunks)]

        with open(doc_dir / "chunks.pkl", "wb") as f:
            pickle.dump({
                "chunks": chunk_records,
                "filename": filename,
                "page_count": page_count,
                "char_count": len(text),
            }, f)

        with open(doc_dir / "summaries.pkl", "wb") as f:
            pickle.dump({
                "summaries": summary_records,
                "filename": filename,
            }, f)

        _write_status(doc_dir, "ready", 100, "done")

        return {
            "doc_id": doc_id,
            "chunk_count": len(chunks),
            "char_count": len(text),
            "page_count": page_count,
        }

    except Exception:
        _write_status(doc_dir, "failed", 0, "error")
        raise


def query_document(user_id: str, doc_id: str, query: str, top_k: int = 8) -> dict:
    """
    Embed query and retrieve from both chunk and summary FAISS indexes.

    Returns:
        {
            "chunks":    [{"chunk": str, "score": float, "chunk_id": int}],
            "summaries": [{"summary": str, "score": float, "chunk_id": int}],
            "entities":  dict,
            "filename":  str,
        }
    Raises FileNotFoundError if the document has not been processed.
    """
    doc_dir = UPLOADS_DIR / user_id / doc_id

    # Support FAISS when installed, NumPy fallback, and legacy index names.
    chunk_index_path = doc_dir / "index_chunks.faiss"
    chunk_npy_path = doc_dir / "index_chunks.npy"
    if not chunk_index_path.exists() and not chunk_npy_path.exists():
        legacy = doc_dir / "index.faiss"
        legacy_npy = doc_dir / "index.npy"
        if legacy.exists():
            chunk_index_path = legacy
        elif legacy_npy.exists():
            chunk_npy_path = legacy_npy
        else:
            raise FileNotFoundError(f"No index found for document {doc_id}. Was it processed?")

    chunks_pkl_path = doc_dir / "chunks.pkl"
    chunk_index = _load_vector_index(chunk_index_path, chunk_npy_path)
    with open(chunks_pkl_path, "rb") as f:
        chunk_data = pickle.load(f)

    # Normalise old format (chunks.pkl stored plain list of strings)
    raw_chunks = chunk_data.get("chunks", [])
    if raw_chunks and isinstance(raw_chunks[0], str):
        raw_chunks = [{"chunk": c, "chunk_id": i} for i, c in enumerate(raw_chunks)]

    # Load summary index if available
    summ_index_path = doc_dir / "index_summaries.faiss"
    summ_npy_path = doc_dir / "index_summaries.npy"
    summ_pkl_path = doc_dir / "summaries.pkl"
    summary_records = []
    summ_index = None
    if (summ_index_path.exists() or summ_npy_path.exists()) and summ_pkl_path.exists():
        summ_index = _load_vector_index(summ_index_path, summ_npy_path)
        with open(summ_pkl_path, "rb") as f:
            summ_data = pickle.load(f)
        summary_records = summ_data.get("summaries", [])

    # Load entities
    entities = {}
    entities_path = doc_dir / "entities.json"
    if entities_path.exists():
        try:
            entities = json.loads(entities_path.read_text())
        except Exception:
            pass

    document_summary = ""
    summary_path = doc_dir / "document_summary.json"
    if summary_path.exists():
        try:
            document_summary = json.loads(summary_path.read_text()).get("summary", "")
        except Exception:
            document_summary = ""

    # Embed query once, reuse for both searches
    model = _get_embedder()
    q_emb = model.encode([query], normalize_embeddings=True)
    q_emb = np.array(q_emb, dtype=np.float32)

    # Search chunk index
    k_c = min(top_k, len(raw_chunks))
    scores_c, indices_c = _search_vector_index(chunk_index, q_emb, k_c)
    chunk_results = []
    best_candidates = []
    for score, idx in zip(scores_c[0], indices_c[0]):
        if idx < 0:
            continue
        rec = raw_chunks[idx]
        item = {
            "chunk": rec["chunk"],
            "score": float(score),
            "chunk_id": rec.get("chunk_id", int(idx)),
        }
        best_candidates.append(item)
        if float(score) >= SCORE_THRESHOLD:
            chunk_results.append(item)

    # Legal questions often target the title page or final order. Always provide
    # beginning/end context, and fall back to top semantic hits if the threshold
    # was too strict for a short user question.
    if not chunk_results:
        chunk_results = best_candidates[: min(4, len(best_candidates))]

    forced = []
    if raw_chunks:
        forced.append({**raw_chunks[0], "score": 1.0, "chunk_id": raw_chunks[0].get("chunk_id", 0)})
    if len(raw_chunks) > 1:
        forced.append({**raw_chunks[-1], "score": 0.99, "chunk_id": raw_chunks[-1].get("chunk_id", len(raw_chunks) - 1)})

    seen_ids = {c.get("chunk_id") for c in chunk_results}
    for item in forced:
        if item.get("chunk_id") not in seen_ids:
            chunk_results.append(item)
            seen_ids.add(item.get("chunk_id"))
    chunk_results = sorted(chunk_results, key=lambda x: x.get("score", 0), reverse=True)[:top_k]

    # Search summary index
    summary_results = []
    if summ_index is not None and summary_records:
        k_s = min(6, len(summary_records))
        scores_s, indices_s = _search_vector_index(summ_index, q_emb, k_s)
        for score, idx in zip(scores_s[0], indices_s[0]):
            if idx >= 0 and float(score) >= SCORE_THRESHOLD:
                rec = summary_records[idx]
                summary_results.append({
                    "summary": rec["summary"],
                    "score": float(score),
                    "chunk_id": rec.get("chunk_id", int(idx)),
                    "kind": rec.get("kind", "section_summary"),
                })

    if document_summary and not any(s.get("kind") == "document_summary" for s in summary_results):
        summary_results.insert(0, {
            "summary": document_summary,
            "score": 1.0,
            "chunk_id": -1,
            "kind": "document_summary",
        })

    return {
        "chunks": chunk_results,
        "summaries": summary_results,
        "entities": entities,
        "filename": chunk_data.get("filename", ""),
        "document_summary": document_summary,
    }


def get_document_info(user_id: str, doc_id: str) -> dict:
    """Return metadata from stored pkl files without loading FAISS indexes."""
    doc_dir = UPLOADS_DIR / user_id / doc_id
    chunks_path = doc_dir / "chunks.pkl"
    if not chunks_path.exists():
        return {}

    with open(chunks_path, "rb") as f:
        data = pickle.load(f)

    raw_chunks = data.get("chunks", [])
    chunk_count = len(raw_chunks)

    entities = {}
    entities_path = doc_dir / "entities.json"
    if entities_path.exists():
        try:
            entities = json.loads(entities_path.read_text())
        except Exception:
            pass

    has_summaries = (doc_dir / "index_summaries.faiss").exists() or (doc_dir / "index_summaries.npy").exists()

    return {
        "filename": data.get("filename", ""),
        "chunk_count": chunk_count,
        "page_count": data.get("page_count", 0),
        "has_summaries": has_summaries,
        "entities": entities,
    }


def delete_document_files(user_id: str, doc_id: str):
    """Remove the entire document directory (all indexes, pkl, and originals)."""
    doc_dir = UPLOADS_DIR / user_id / doc_id
    if doc_dir.exists():
        shutil.rmtree(doc_dir)

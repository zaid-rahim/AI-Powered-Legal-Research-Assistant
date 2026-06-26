"""
Enhanced RAG pipeline for user-uploaded legal documents.

Uses dual FAISS retrieval (chunks + summaries) and LLaMA-extracted entity context.
Calls Ollama directly; answer is grounded only in the uploaded document.
"""

import os
import requests

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")

MAX_SUMMARY_CHARS = 5500   # chars from LLaMA summaries section
MAX_CHUNK_CHARS = 9000     # chars from raw document excerpts

SYSTEM_PROMPT = """\
You are LAWPAK, an AI legal research assistant specialised in Pakistani court judgments.

STRICT INSTRUCTIONS:
1. Answer ONLY using the document excerpts and extracted information provided below.
2. Do NOT invent facts, cite external cases, or rely on prior knowledge.
3. If the answer is not present in the document, respond exactly:
   "This information is not found in the uploaded document."
4. First read the document overview and entity facts, then use the raw excerpts
   as evidence for the exact answer.
5. Structure your answer as:
   **Case Overview:** (parties, type, judges/court if available)
   **Relevant Findings:** (what the document says about the question)
   **Key Points:** (bullet list of the most important details)
   **Conclusion:** (direct answer to the question)
6. Quote short phrases from the document where useful.
7. Reference the judge(s), parties, laws/sections, and final decision when relevant.\
"""


def _format_entities(entities: dict) -> str:
    if not entities:
        return ""
    parts = []
    if entities.get("judges"):
        judges = entities["judges"]
        if isinstance(judges, list):
            parts.append(f"Judge(s): {', '.join(judges)}")
        else:
            parts.append(f"Judge(s): {judges}")
    if entities.get("petitioner"):
        parts.append(f"Petitioner: {entities['petitioner']}")
    if entities.get("respondent"):
        parts.append(f"Respondent: {entities['respondent']}")
    if entities.get("laws"):
        laws = entities["laws"]
        if isinstance(laws, list):
            parts.append(f"Laws/Sections: {', '.join(laws)}")
        else:
            parts.append(f"Laws/Sections: {laws}")
    if entities.get("decision") and entities["decision"] not in ("", "unknown"):
        parts.append(f"Final Decision: {entities['decision']}")
    if entities.get("case_type"):
        parts.append(f"Case Type: {entities['case_type']}")
    return "\n".join(parts)


def _build_context(chunks: list, summaries: list, entities: dict, document_summary: str = "") -> str:
    """
    Assemble the LLM context block in order:
        1. Extracted entities (structured facts)
        2. Section summaries (high-level understanding)
        3. Raw document excerpts (detailed evidence)
    """
    parts = []

    entity_str = _format_entities(entities)
    if entity_str:
        parts.append(f"DOCUMENT ENTITIES (extracted by AI):\n{entity_str}")

    if document_summary:
        parts.append(f"OVERALL DOCUMENT SUMMARY:\n{document_summary}")

    # Summaries — provide high-level understanding of each section
    summ_chars = 0
    for s in summaries:
        text = s.get("summary", "")
        if not text:
            continue
        if summ_chars + len(text) > MAX_SUMMARY_CHARS:
            break
        if s.get("kind") == "document_summary" or s.get("chunk_id") == -1:
            continue
        label = f"SECTION {s.get('chunk_id', 0) + 1} SUMMARY"
        parts.append(f"[{label}]:\n{text}")
        summ_chars += len(text)

    # Raw chunks — detailed text for precise quotation
    chunk_chars = 0
    for c in chunks:
        text = c.get("chunk", "")
        if not text:
            continue
        label = f"DOCUMENT EXCERPT (part {c.get('chunk_id', 0) + 1})"
        if chunk_chars + len(text) > MAX_CHUNK_CHARS:
            remaining = MAX_CHUNK_CHARS - chunk_chars
            if remaining > 300:
                parts.append(f"[{label}]:\n{text[:remaining].rstrip()} [...]")
            break
        parts.append(f"[{label}]:\n{text}")
        chunk_chars += len(text)

    return "\n\n---\n\n".join(parts)


def _build_document_kg(doc_name: str, entities: dict) -> dict:
    """Small client-friendly graph from extracted uploaded-document facts."""
    root_id = doc_name or "uploaded_document"
    nodes = [{"id": root_id, "label": root_id, "type": "Document"}]
    edges = []

    def add_node(value, node_type, rel):
        if not value:
            return
        node_id = f"{node_type}:{value}"
        nodes.append({"id": node_id, "label": value, "type": node_type})
        edges.append({"from": root_id, "to": node_id, "label": rel, "type": node_type})

    for judge in entities.get("judges", []) if isinstance(entities.get("judges"), list) else [entities.get("judges")]:
        add_node(judge, "Judge", "HEARD_BY")
    add_node(entities.get("petitioner"), "Petitioner", "HAS_PETITIONER")
    add_node(entities.get("respondent"), "Respondent", "HAS_RESPONDENT")
    for law in entities.get("laws", []) if isinstance(entities.get("laws"), list) else [entities.get("laws")]:
        add_node(law, "Law", "INVOLVES_SECTION")
    add_node(entities.get("decision"), "Decision", "HAS_DECISION")
    add_node(entities.get("case_type"), "CaseType", "HAS_CASE_TYPE")

    dedup_nodes = {}
    for node in nodes:
        dedup_nodes[node["id"]] = node
    return {"nodes": list(dedup_nodes.values()), "edges": edges}


def _fallback_answer(question: str, chunks: list, summaries: list, entities: dict) -> str:
    entity_lines = _format_entities(entities) or "No structured entity facts were extracted."
    evidence = []
    q_words = {w for w in question.lower().split() if len(w) > 3}
    for c in chunks:
        text = c.get("chunk", "")
        sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
        for sentence in sentences:
            lower = sentence.lower()
            if q_words and any(w in lower for w in q_words):
                evidence.append(sentence)
            if len(evidence) >= 5:
                break
        if len(evidence) >= 5:
            break
    if not evidence and summaries:
        evidence = [summaries[0].get("summary", "")[:800]]
    if not evidence and chunks:
        evidence = [chunks[0].get("chunk", "")[:800]]

    evidence_text = "\n".join(f"- {e[:500]}" for e in evidence if e)
    return (
        "**Case Overview:**\n"
        f"{entity_lines}\n\n"
        "**Relevant Findings:**\n"
        f"{evidence_text or 'This information is not found in the uploaded document.'}\n\n"
        "**Key Points:**\n"
        f"{evidence_text or '- No matching passage was retrieved.'}\n\n"
        "**Conclusion:**\n"
        "Generated from retrieved document text because the local Ollama model is unavailable."
    )


def run_user_doc_rag(
    question: str,
    doc_data,          # dict from query_document() or legacy list of chunks
    model: str = None,
    doc_name: str = "",
) -> dict:
    """
    Generate a structured answer grounded only in the provided document.

    Args:
        question:  The user's query string.
        doc_data:  Either:
                   - new dict: {"chunks": [...], "summaries": [...], "entities": {...}}
                   - legacy list: [{"chunk": str, "score": float}] (old pipeline)
        model:     Ollama model name override (defaults to llama3.1:8b).
        doc_name:  Display name of the document (used in source attribution).

    Returns:
        {"answer": str, "sources": list, "summary": str|None, "kg": None}
    """
    model = model or DEFAULT_MODEL

    # Handle both new dict format and old list format
    if isinstance(doc_data, list):
        chunks = doc_data
        summaries = []
        entities = {}
    else:
        chunks = doc_data.get("chunks", [])
        summaries = doc_data.get("summaries", [])
        entities = doc_data.get("entities", {})
        document_summary = doc_data.get("document_summary", "")

    if not chunks and not summaries:
        return {
            "answer": (
                "No relevant content was found in your document for this question. "
                "Try rephrasing your question, or check that your document covers this topic."
            ),
            "sources": [],
            "summary": None,
            "kg": None,
        }

    if isinstance(doc_data, list):
        document_summary = ""

    context = _build_context(chunks, summaries, entities, document_summary)
    doc_label = f'"{doc_name}"' if doc_name else "the uploaded document"

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"DOCUMENT: {doc_label}\n\n"
        f"{context}\n\n"
        f"QUESTION: {question}\n\n"
        "Answer:"
    )

    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "num_predict": 900,
                    "num_ctx": 8192,
                },
            },
            timeout=300,
        )
        resp.raise_for_status()
        answer = resp.json().get("response", "").strip()
        if not answer:
            answer = "The model returned an empty response. Please try again."
    except requests.exceptions.Timeout:
        answer = _fallback_answer(question, chunks, summaries, entities)
    except Exception as exc:
        print(f"  [doc-rag] Ollama unavailable; fallback answer used: {exc}", flush=True)
        answer = _fallback_answer(question, chunks, summaries, entities)

    # First section summary becomes the "summary" card in the UI
    doc_summary = document_summary or (summaries[0]["summary"] if summaries else None)

    kg = _build_document_kg(doc_name or "uploaded_document", entities)

    sources = [
        {
            "text": c["chunk"][:300] + ("..." if len(c["chunk"]) > 300 else ""),
            "score": round(c.get("score", 0.0), 3),
            "source": doc_name or "uploaded_document",
            "chunk_index": c.get("chunk_id", i),
        }
        for i, c in enumerate(chunks)
    ]

    return {
        "answer": answer,
        "sources": sources,
        "summary": doc_summary,
        "kg": kg,
    }

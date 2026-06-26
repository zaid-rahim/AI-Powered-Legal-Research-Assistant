import re
import pickle
import traceback
import requests
import os
from pathlib import Path
from neo4j import GraphDatabase
import faiss
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings


#  CONFIGURATION 

# Directory paths for the FAISS vector indices
VECTOR_SUMMARY_DIR = Path("vector_store_summaries")
VECTOR_CHUNK_DIR   = Path("vector_store")

# Neo4j Database Connection Details with environmental variable fallbacks
NEO4J_URI      = os.getenv("NEO4J_URI", "bolt://localhost:7687").strip()
NEO4J_USER     = os.getenv("NEO4J_USER", "neo4j").strip()
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

# Flexible database selection: attempts to use environment variables, 
# falling back to "legalkg" or "neo4j" if specific ones fail.
NEO4J_DATABASE = (os.getenv("NEO4J_DATABASE") or os.getenv("NEO4J_DB") or "legalkg").strip()
_DB_FALLBACKS  = [d.strip() for d in os.getenv("NEO4J_DB_FALLBACKS", "neo4j,legalkg").split(",") if d.strip()]

# Build a prioritized list of database candidates to attempt connection with
NEO4J_DB_CANDIDATES = []
for _db in [NEO4J_DATABASE] + _DB_FALLBACKS:
    if _db and _db not in NEO4J_DB_CANDIDATES:
        NEO4J_DB_CANDIDATES.append(_db)

ACTIVE_NEO4J_DB = None

# Ollama API Configuration
OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:8b"

# RAG & Context Limits
MAX_CONTEXT_CHARS         = 16000     # Absolute maximum characters to handle
LLM_CONTEXT_CHARS         = 6500      # Soft limit for LLM prompt context
SEMANTIC_TOP_K            = 10        # How many FAISS results to retrieve initially
SIMILARITY_THRESHOLD      = 0.50      # Minimum FAISS similarity score to accept
PRIMARY_FULL_TEXT_CHARS   = 8000      # Amount of full text to pull for highly relevant cases
SECONDARY_FULL_TEXT_CHARS = 1500      # Amount of text to pull for secondary/supporting cases

# LLM Inference Parameters
LLAMA_TEMPERATURE = 0.0               # 0.0 ensures highly deterministic and factual answers
LLAMA_NUM_PREDICT = 800               # Max tokens for the generated response
OLLAMA_TIMEOUT    = 300               # Wait up to 5 minutes for generation


#  LOAD EMBEDDING MODEL 

print("\nLoading BGE embedder...")
# Initialize the HuggingFace embedding model.
# Must match the model used to create the vector store.
embedder = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={"device": "cpu"},    # Using CPU here to save GPU VRAM for the LLM
    encode_kwargs={"normalize_embeddings": True, "batch_size": 32}
)


#  FAISS LOADING 

def find_index_and_pkl(folder: Path):
    """Locates the required FAISS and pickle metadata files in a directory."""
    faiss_possible = [folder/"index.faiss", folder/"summaries.faiss"]
    pkl_possible   = [folder/"index.pkl",   folder/"summaries.pkl"]
    
    faiss_path = next((p for p in faiss_possible if p.exists()), None)
    pkl_path   = next((p for p in pkl_possible   if p.exists()), None)
    
    return faiss_path, pkl_path


def load_faiss_and_docs(folder: Path):
    """Loads a FAISS index and its corresponding metadata array into memory."""
    faiss_path, pkl_path = find_index_and_pkl(folder)
    if not faiss_path or not pkl_path:
        raise FileNotFoundError(f"Missing faiss or pkl in folder: {folder}")
        
    print("Loading faiss:", faiss_path)
    index = faiss.read_index(str(faiss_path))
    
    print("Loading docs:", pkl_path)
    with open(pkl_path, "rb") as f:
        docs = pickle.load(f)
        
    normalized = []
    # Standardize the format of the metadata (can be a list, dict, or object)
    items = docs if isinstance(docs, list) else list(docs.values()) if isinstance(docs, dict) else [docs]
    
    for i, it in enumerate(items):
        if isinstance(it, dict):
            # Try to grab the text from various common keys
            text = (it.get("merged_summary") or it.get("summary") or
                    it.get("text") or it.get("page_content") or "")
            doc_id = (it.get("case_id") or it.get("source") or
                      it.get("id") or f"doc_{i}")
        else:
            text   = str(it)
            doc_id = f"doc_{i}"
            
        normalized.append({"id": doc_id, "text": text})
        
    print("Loaded docs:", len(normalized))
    return index, normalized


print("Loading FAISS summary store...")
index_summ, docs_summ = load_faiss_and_docs(VECTOR_SUMMARY_DIR)

print("Loading FAISS chunk store...")
index_chunk, docs_chunk = load_faiss_and_docs(VECTOR_CHUNK_DIR)


#  NEO4J 

print("Connecting to Neo4j...")
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def _db_candidates():
    """Returns a list of databases to try, putting the active one first."""
    if ACTIVE_NEO4J_DB:
        return [ACTIVE_NEO4J_DB] + [d for d in NEO4J_DB_CANDIDATES if d != ACTIVE_NEO4J_DB]
    return list(NEO4J_DB_CANDIDATES)


def _probe_neo4j():
    """Attempts to connect to the configured Neo4j databases until one works."""
    global ACTIVE_NEO4J_DB
    last_exc = None
    for db in _db_candidates():
        try:
            with driver.session(database=db) as session:
                session.run("RETURN 1").consume()
            ACTIVE_NEO4J_DB = db
            print(f"Connected. Using database: {db}")
            return
        except Exception as exc:
            last_exc = exc
            print(f"  [neo4j] DB probe failed for '{db}': {exc}")
            
    if last_exc:
        raise last_exc
    raise RuntimeError("No Neo4j database candidates configured.")


def _neo4j_run(query: str, **params):
    """Executes a Cypher query against the active database."""
    global ACTIVE_NEO4J_DB
    last_exc = None
    for db in _db_candidates():
        try:
            with driver.session(database=db) as session:
                rows = list(session.run(query, **params))
            ACTIVE_NEO4J_DB = db
            return rows
        except Exception as exc:
            last_exc = exc
            print(f"  [neo4j] query failed on '{db}': {exc}")
            
    if last_exc:
        raise last_exc
    return []


def _neo4j_single(query: str, **params):
    """Executes a Cypher query and returns only the first row (or None)."""
    rows = _neo4j_run(query, **params)
    return rows[0] if rows else None

# Test connection on startup
_probe_neo4j()


#  HYBRID SEMANTIC SEARCH 

def hybrid_semantic_search(question: str, top_k=SEMANTIC_TOP_K):
    """
    Performs a hybrid search: Dense vector search + keyword overlap boost.
    This ensures that visually similar legal terminology is prioritized.
    """
    try:
        print("  [semantic] searching summaries...")
        
        # 1. Embed query
        q_emb = np.array([embedder.embed_query(question)]).astype("float32")
        
        # 2. Vector search via FAISS
        scores, indices = index_summ.search(q_emb, top_k)
        
        # 3. Clean query for keyword matching
        question_words = set(re.sub(r"[^a-z0-9 ]", " ", question.lower()).split())
        candidates = []
        
        # 4. Iteratively score candidates
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(docs_summ):
                continue
            if score < SIMILARITY_THRESHOLD:
                continue
                
            doc = docs_summ[idx]
            # Boost score by 0.02 for every keyword that directly overlaps
            keyword_overlap = sum(w in doc["text"].lower() for w in question_words)
            candidates.append((score + 0.02 * keyword_overlap, doc))
            
        # 5. Sort by boosted score and return top 5
        candidates.sort(key=lambda x: x[0], reverse=True)
        top_docs = [doc for _, doc in candidates[:5]]
        
        print("  [semantic] hits:", [d["id"] for d in top_docs])
        return top_docs
    except Exception:
        print("  [semantic] ERROR:")
        traceback.print_exc()
        return []


#  CHUNK RETRIEVAL 

def _chunk_matches_case(doc_id: str, cid_lower: str) -> bool:
    """Safely determines if a specific chunk ID belongs to a given case ID."""
    if doc_id == cid_lower:
        return True
    # Match pattern: "Crl.A.123_chunk_4" starts with "Crl.A.123" + a separator
    if doc_id.startswith(cid_lower) and len(doc_id) > len(cid_lower):
        return doc_id[len(cid_lower)] in ("_", "-", "/", " ", "\\")
    return False


def retrieve_chunks(case_ids: list, question: str, max_chunks=12):
    """
    Retrieves the most semantically relevant text chunks for a specified list of cases.
    It searches a massive pool, filtering only those chunks that belong to `case_ids`.
    """
    if not case_ids:
        return []
    try:
        print("  [chunks] retrieving for:", case_ids)
        
        # Vectorize the question
        q_emb = np.array([embedder.embed_query(question)]).astype("float32")
        
        # Grab a large pool of chunks (top 100)
        k = min(100, index_chunk.ntotal)
        _, indices = index_chunk.search(q_emb, k)
        
        case_ids_lower = [c.lower() for c in case_ids]
        selected, seen = [], set()

        # Iterate through retrieved FAISS chunks
        for idx in indices[0]:
            if idx < 0 or idx >= len(docs_chunk):
                continue
                
            doc    = docs_chunk[idx]
            doc_id = doc["id"].lower()
            
            # Keep the chunk only if it belongs to one of our target case_ids
            for cid_lower in case_ids_lower:
                if _chunk_matches_case(doc_id, cid_lower):
                    if doc_id not in seen:
                        selected.append(doc)
                        seen.add(doc_id)
                    break
                    
            if len(selected) >= max_chunks:
                break

        # Fallback Mechanism: If FAISS missed the target cases entirely,
        # manually scan the database linearly to pull out the first matching chunks.
        if not selected:
            print("  [chunks] FAISS missed — running full scan fallback...")
            for doc in docs_chunk:
                doc_id = doc["id"].lower()
                for cid_lower in case_ids_lower:
                    if _chunk_matches_case(doc_id, cid_lower):
                        if doc_id not in seen:
                            selected.append(doc)
                            seen.add(doc_id)
                        break
                if len(selected) >= max_chunks:
                    break

        print(f"  [chunks] selected {len(selected)} chunks")
        return selected
    except Exception:
        print("  [chunks] ERROR:")
        traceback.print_exc()
        return []


#  KG FUNCTIONS 

def safe_kg(fn, *args):
    """Wrapper to prevent the entire pipeline from failing if Neo4j crashes."""
    try:
        return fn(*args)
    except Exception:
        print(f"  [kg] ERROR in {fn.__name__}:")
        traceback.print_exc()
        return [] if fn.__name__.startswith("kg_cases") else None


def kg_get_summary(case_id: str):
    """Fetches the high-level summary of a specific case from the graph."""
    q = """
    MATCH (c)-[:HAS_SUMMARY]->(s:Summary)
    WHERE
        toLower(coalesce(c.id, c.case_id, c.name, s.case_id, s.id)) = toLower($cid)
        OR toLower(coalesce(c.id, c.case_id, c.name, s.case_id, s.id)) STARTS WITH toLower($cid + "/")
    RETURN s.text AS txt
    ORDER BY size(coalesce(s.text, "")) DESC
    LIMIT 1
    """
    row = _neo4j_single(q, cid=case_id)
    return row["txt"] if row else None


def kg_get_graph_facts(case_id: str):
    """Converts the Neo4j relationships into plain text sentences for the LLM to read."""
    q = """
    MATCH (c)-[r]->(n)
    WHERE (c:Case OR c:Entity)
      AND (
          toLower(coalesce(c.id, c.case_id, c.name)) = toLower($cid)
          OR toLower(coalesce(c.id, c.case_id, c.name)) STARTS WITH toLower($cid + "/")
      )
    RETURN type(r) AS rel,
           coalesce(head(labels(n)), "Entity") AS label,
           coalesce(n.name, n.value, n.case_id, n.id, n.code, n.number) AS node_name
    """
    lines = []
    for row in _neo4j_run(q, cid=case_id):
        # Example output: "HAS_JUDGE -> Judge: Qazi Faez Isa"
        lines.append(f"  {row['rel']} -> {row['label']}: {row['node_name']}")
    return "\n".join(lines) if lines else None


def kg_get_graph_edges(case_id: str):
    """Extracts raw JSON structure of nodes/edges for the frontend UI visualization."""
    q = """
    MATCH (c)
    WHERE (c:Case OR c:Entity)
      AND (
          toLower(coalesce(c.id, c.case_id, c.name)) = toLower($cid)
          OR toLower(coalesce(c.id, c.case_id, c.name)) STARTS WITH toLower($cid + "/")
      )
    OPTIONAL MATCH (c)-[r]->(n)
    RETURN
        coalesce(c.id, c.case_id, c.name) AS case_id,
        type(r) AS rel,
        coalesce(head(labels(n)), "Entity") AS label,
        coalesce(n.name, n.value, n.case_id, n.id, n.code, n.number) AS node_name
    """
    edges = []
    for row in _neo4j_run(q, cid=case_id):
        if row["rel"] and row["node_name"]:
            edges.append({
                "from":  row["case_id"],
                "to":    row["node_name"],
                "label": row["rel"],
                "type":  row["label"]
            })
    return edges

# --- Entity Routing ---
# These functions execute reverse-lookups. 
# They start with a specific entity (e.g. Judge name) and return all associated cases.

def kg_cases_by_person(name: str):
    q = """
    MATCH (c)-[r]->(p)
    WHERE (c:Case OR c:Entity)
      AND type(r) IN ["HAS_PETITIONER", "HAS_RESPONDENT", "HAS_COMPLAINANT", "HAS_ACCUSED", "HAS_WITNESS"]
      AND toLower(coalesce(p.name, p.value, p.id, p.code, p.number)) CONTAINS toLower($name)
    RETURN DISTINCT coalesce(c.id, c.case_id, c.name) AS case_id
    LIMIT 10
    """
    return [r["case_id"] for r in _neo4j_run(q, name=name)]


def kg_cases_by_judge(name: str):
    q = """
    MATCH (c)-[r]->(j)
    WHERE (c:Case OR c:Entity)
      AND type(r) IN ["HEARD_BY", "HAS_JUDGE"]
      AND toLower(coalesce(j.name, j.value, j.id)) CONTAINS toLower($name)
    RETURN DISTINCT coalesce(c.id, c.case_id, c.name) AS case_id
    LIMIT 10
    """
    return [r["case_id"] for r in _neo4j_run(q, name=name)]


def kg_cases_by_section(section_no: str):
    q = """
    MATCH (c)-[r]->(s)
    WHERE (c:Case OR c:Entity)
      AND type(r) IN ["INVOLVES_SECTION", "REFERS_TO_SECTION", "HAS_SECTION", "HAS_LAW"]
      AND toLower(coalesce(s.name, s.code, s.value, s.id, s.number)) CONTAINS toLower($sec)
    RETURN DISTINCT coalesce(c.id, c.case_id, c.name) AS case_id
    LIMIT 10
    """
    return [r["case_id"] for r in _neo4j_run(q, sec=section_no)]


def kg_cases_by_decision(keyword: str):
    q = """
    MATCH (c)-[r]->(d)
    WHERE (c:Case OR c:Entity)
      AND type(r) IN ["HAS_DECISION", "DECISION", "DECIDED_IN"]
      AND toLower(coalesce(d.name, d.value, d.id, d.case_id)) CONTAINS toLower($kw)
    RETURN DISTINCT coalesce(c.id, c.case_id, c.name) AS case_id
    LIMIT 10
    """
    return [r["case_id"] for r in _neo4j_run(q, kw=keyword)]


#  CASE ID DETECTION 

# Regex to detect formal case IDs in plain text
CASE_PATTERNS = [
    r"\bCrl\.?A\.?\s*\d+\b",
    r"\bCriminal\s*Appeal\s*No\.?\s*\d+\/\d{4}\b",
]

def normalize_case_id(found: str) -> str:
    found = found.replace(" ", "")
    if "/" in found:
        num = re.search(r"(\d+)", found)
        if num:
            return f"Crl.A.{num.group(1)}"
    return found

def detect_case_id(text: str):
    for p in CASE_PATTERNS:
        m = re.search(p, text, flags=re.IGNORECASE)
        if m:
            return normalize_case_id(m.group(0))
    return None


def clean_name_for_search(name: str) -> str:
    """Removes special characters to ensure Neo4j CONTAINS search doesn't break."""
    name = name.split("@")[0].strip()
    name = re.sub(r"[^A-Za-z\s]", "", name).strip()
    return name


#  LLM GENERATION 

SYSTEM_PROMPT = """You are LAWPAK, a Pakistani legal research assistant specializing in Supreme Court criminal cases.

INSTRUCTIONS:
1. Answer using ONLY the provided context. Do not invent facts.
2. For analytical questions (e.g. "how does the court treat X"), extract and synthesize the relevant legal reasoning found across ALL cases in the context.
3. For listing questions (e.g. "which cases involved X"), list EVERY case ID found in the context that is relevant — do not stop after 1 or 2.
4. Always cite case IDs clearly (e.g. Crl.A.1, Crl.A.76).
5. If and ONLY IF the context contains absolutely zero relevant information, respond: INSUFFICIENT_DOCUMENTATION.
6. Structure your answer with: Background, Key Legal Principle, Cases and Holdings, Conclusion.
"""

ALLOWED_MODELS = {
    "llama3.1:latest",
    "llama3:8b",
    "llama4:latest",
    "mistral:latest",
}

def call_ollama(prompt: str, model: str | None = None) -> str:
    """Sends the context + prompt to the local Ollama LLM for generation."""
    chosen = model if (model and model in ALLOWED_MODELS) else OLLAMA_MODEL
    try:
        r = requests.post(
            OLLAMA_URL,
            json={
                "model": chosen,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": LLAMA_TEMPERATURE,
                    "num_predict": LLAMA_NUM_PREDICT,
                    "num_ctx": 8192,     # Allocate an 8k context window
                    "num_gpu": 99        # Maximize GPU layer offloading
                }
            },
            timeout=OLLAMA_TIMEOUT
        )
        return r.json().get("response", "").strip()
    except Exception:
        print("  [ollama] ERROR:")
        traceback.print_exc()
        return "ERROR: Could not reach Ollama."


#  MAIN RAG PIPELINE 

def run_rag(question: str, model: str | None = None) -> dict:
    """
    The orchestrator function:
    1. Parse question for cases/entities.
    2. Hybrid FAISS search.
    3. Build context from KG, FAISS, and text files.
    4. Pass context to LLM.
    """
    print("\n" + "="*60)
    print("QUERY:", question)
    print("MODEL:", model if (model and model in ALLOWED_MODELS) else OLLAMA_MODEL)
    print("="*60)

    routed_ids    = []
    context_parts = []
    used_sources  = []

    # ── STEP 1: Entity routing ───────────────────────────────────
    # We parse the query and extract case IDs directly from the Neo4j KG if it mentions
    # a specific judge, person, section, etc.

    case_id = detect_case_id(question)
    if case_id:
        print("[route] case id:", case_id)
        routed_ids.append(case_id)

    section_match = re.search(r"section\s*(\d+)", question, re.IGNORECASE)
    if section_match:
        sec = section_match.group(1)
        print("[route] section:", sec)
        routed_ids.extend(safe_kg(kg_cases_by_section, sec))

    judge_match = re.search(
        r"(?:Justice|MR\.?\s*JUSTICE)\s+([A-Za-z][A-Za-z\s]{2,40})",
        question, re.IGNORECASE
    )
    if judge_match:
        jname = clean_name_for_search(judge_match.group(1)).strip()
        print("[route] judge:", jname)
        routed_ids.extend(safe_kg(kg_cases_by_judge, jname))

    decision_keywords = ["dismissed", "acquitted", "death", "conviction"]
    for word in decision_keywords:
        if word in question.lower():
            print("[route] decision kw:", word)
            routed_ids.extend(safe_kg(kg_cases_by_decision, word))
            break

    person_matches = re.findall(r"[A-Z][a-z]+(?:\s[A-Z][a-z]+)+", question)
    if person_matches:
        raw_name  = person_matches[0]
        safe_name = clean_name_for_search(raw_name)
        if safe_name and safe_name.lower() not in ("justice", "mr justice"):
            print("[route] person:", safe_name)
            routed_ids.extend(safe_kg(kg_cases_by_person, safe_name))

    # Remove duplicates
    routed_ids = list(dict.fromkeys(routed_ids))

    # ── STEP 2: Semantic search ──────────────────────────────────
    # Perform a dense vector search across the general case summaries to find
    # similar issues even if they weren't explicitly named.
    safe_question = re.sub(r"[^\w\s\?\.\,]", " ", question).strip()
    semantic_docs = hybrid_semantic_search(safe_question)
    semantic_ids  = [d["id"] for d in semantic_docs]

    # ── STEP 3: Merge IDs ────────────────────────────────────────
    # Combine explicit (routed) cases and implicit (semantic) cases
    all_case_ids = list(dict.fromkeys(routed_ids + semantic_ids))
    print("[merge] all case ids:", all_case_ids)

    # Prioritize cases specifically requested over semantic matches
    primary_ids   = routed_ids if routed_ids else all_case_ids[:3]
    secondary_ids = [cid for cid in all_case_ids if cid not in primary_ids]


    # ── STEP 4: Build context ────────────────────────────────────

    def add_case(cid: str, text_limit: int):
        """Helper to append various forms of context for a single case to the master list."""
        # Add summary
        kg_summ = safe_kg(kg_get_summary, cid)
        if kg_summ:
            context_parts.append(f"[KG_SUMMARY:{cid}]\n{kg_summ}")
            used_sources.append(f"{cid}:kg_summary")

        # Add KG graph facts
        graph = safe_kg(kg_get_graph_facts, cid)
        if graph:
            context_parts.append(f"[KG_GRAPH:{cid}]\n{graph}")
            used_sources.append(f"{cid}:kg_graph")

        # Add raw full text directly from file
        clean_path = Path("data_clean") / f"{cid}.txt"
        if clean_path.exists():
            try:
                text = clean_path.read_text(encoding="utf-8")
                trunc = text[:text_limit]
                context_parts.append(f"[FULL_TEXT:{cid}]\n{trunc}")
                used_sources.append(f"{cid}:full_text")
                print(f"  [full_text] {cid}: {len(text)} chars, using {len(trunc)}")
            except Exception:
                print(f"  [full_text] ERROR reading {cid}:")
                traceback.print_exc()

    # Process Primary Cases
    for cid in primary_ids[:5]:
        add_case(cid, PRIMARY_FULL_TEXT_CHARS)

    primary_chunks = retrieve_chunks(primary_ids[:5], safe_question, max_chunks=8)
    for c in primary_chunks:
        context_parts.append(f"[CHUNK:{c['id']}]\n{c['text']}")
        used_sources.append(c["id"])

    # Add Semantically Matched Summaries
    for d in semantic_docs:
        if d["id"] not in primary_ids:
            context_parts.append(f"[VECTOR_SUMMARY:{d['id']}]\n{d['text']}")
            used_sources.append(f"{d['id']}:vsumm")

    # Process Secondary Cases (shorter context)
    for cid in secondary_ids[:4]:
        add_case(cid, SECONDARY_FULL_TEXT_CHARS)

    secondary_chunks = retrieve_chunks(secondary_ids[:4], safe_question, max_chunks=4)
    for c in secondary_chunks:
        context_parts.append(f"[CHUNK:{c['id']}]\n{c['text']}")
        used_sources.append(c["id"])


    # ── STEP 5: Call LLM ─────────────────────────────────────────
    
    if not context_parts:
        return {"answer": "INSUFFICIENT_DOCUMENTATION", "sources": [], "summary": None, "kg": None}

    # Truncate context string to fit inside the LLM prompt window
    full_context = "\n\n".join(context_parts)[:LLM_CONTEXT_CHARS]
    print(f"[context] {len(full_context)} chars across {len(used_sources)} sources")

    prompt = f"""{SYSTEM_PROMPT}

CONTEXT:
{full_context}

QUESTION:
{question}

Answer thoroughly. Cite every relevant case ID.
"""

    answer = call_ollama(prompt, model=model)


    # ── STEP 6: Build summary (first available KG summary) ───────
    # The UI typically displays a single high-level case summary alongside the answer.
    summary_text = None
    for cid in primary_ids[:5]:
        kg_summ = safe_kg(kg_get_summary, cid)
        if kg_summ:
            summary_text = kg_summ
            break   

    # ── STEP 7: Build KG edges for visualisation ─────────────────
    # Return JSON representation of the graph so the UI can render nodes and edges.
    kg_edges = []
    for cid in primary_ids[:5]:
        try:
            edges = kg_get_graph_edges(cid)
            kg_edges.extend(edges)
        except Exception:
            traceback.print_exc()

    return {
        "answer":  answer,
        "sources": list(dict.fromkeys(used_sources)),
        "summary": summary_text if summary_text else None,
        "kg":      kg_edges if kg_edges else None
    }


#  CLI 

if __name__ == "__main__":
    print("\nLAWPAK — Full Retrieval Legal RAG")
    # Simple terminal interactive loop
    while True:
        try:
            q = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() == "quit":
            break
            
        try:
            result = run_rag(q)
            print("\n--- ANSWER ---")
            print(result["answer"])
            print("\n--- SOURCES ---")
            for s in result["sources"]:
                print(" *", s)
            print("\n--- SUMMARY ---")
            print(result["summary"] or "None")
            print("\n--- KG EDGES ---")
            for e in (result["kg"] or []):
                print(" *", e)
        except Exception:
            print("\n[FATAL ERROR in run_rag]")
            traceback.print_exc()

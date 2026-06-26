import os
import re
import pickle
import requests
from pathlib import Path
from neo4j import GraphDatabase
import faiss
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings


#  CONFIGURATION 

# Directory paths for the FAISS vector indices
VECTOR_SUMMARY_DIR = Path("vector_store_summaries")
VECTOR_CHUNK_DIR   = Path("vector_store")

# Neo4j Database Connection Details
NEO4J_URI  = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

# Local LLM API endpoint (Ollama)
OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:8b"

# RAG & Search Parameters
MAX_CONTEXT_CHARS = 4000      # Max length of text to feed to the LLM to prevent overflow
SEMANTIC_TOP_K = 8            # How many initial FAISS results to retrieve
SIMILARITY_THRESHOLD = 0.55   # Minimum score required to consider a FAISS result valid

LLAMA_TEMPERATURE = 0.0       # 0.0 forces the model to be deterministic and factual
LLAMA_NUM_PREDICT = 700       # Maximum length of the generated answer


#  LOAD EMBEDDING MODEL 

print("\nLoading BGE embedder...")

# Initialize the model used to convert text queries into vectors.
# This must be the EXACT SAME model used during the build_vector_store phase.
embedder = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={"device": "cuda"},
    encode_kwargs={
        "normalize_embeddings": True,
        "batch_size": 32
    }
)


#  FAISS LOADING 

def find_index_and_pkl(folder: Path):
    """Helper to locate the FAISS index file and the metadata pickle file within a folder."""
    faiss_possible = [folder/"index.faiss", folder/"summaries.faiss"]
    pkl_possible   = [folder/"index.pkl",   folder/"summaries.pkl"]
    
    # Return the first matching file that actually exists
    faiss_path = next((p for p in faiss_possible if p.exists()), None)
    pkl_path   = next((p for p in pkl_possible if p.exists()), None)
    return faiss_path, pkl_path


def load_faiss_and_docs(folder: Path):
    """Loads a FAISS index and its corresponding document metadata from disk."""
    faiss_path, pkl_path = find_index_and_pkl(folder)
    if not faiss_path or not pkl_path:
        raise FileNotFoundError(f"Missing faiss or pkl in folder: {folder}")

    print("Loading faiss:", faiss_path)
    # Read the numerical index
    index = faiss.read_index(str(faiss_path))

    print("Loading docs:", pkl_path)
    # Read the metadata (which contains the actual text)
    with open(pkl_path, "rb") as f:
        docs = pickle.load(f)

    # Normalize the metadata format so search functions can handle it consistently
    normalized = []
    items = docs if isinstance(docs, list) else list(docs.values()) if isinstance(docs, dict) else [docs]

    for i, it in enumerate(items):
        if isinstance(it, dict):
            # Try to find the text using various common keys
            text = (
                it.get("merged_summary")
                or it.get("summary")
                or it.get("text")
                or it.get("page_content")
                or ""
            )
            # Try to find the unique ID
            doc_id = (
                it.get("case_id")
                or it.get("source")
                or it.get("id")
                or f"doc_{i}"
            )
        else:
            text = str(it)
            doc_id = f"doc_{i}"

        normalized.append({"id": doc_id, "text": text})

    print("Loaded docs:", len(normalized))
    return index, normalized


# Load both the Summaries Index and the Detailed Chunks Index into memory
print("Loading FAISS summary store...")
index_summ, docs_summ = load_faiss_and_docs(VECTOR_SUMMARY_DIR)

print("Loading FAISS chunk store...")
index_chunk, docs_chunk = load_faiss_and_docs(VECTOR_CHUNK_DIR)


#  NEO4J CONNECTION 

print("Connecting to Neo4j...")
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
# Test the connection with a simple query
with driver.session(database="legalkg") as session:
    session.run("RETURN 1")
print("Connected.")


#  HYBRID SEMANTIC SEARCH 

def hybrid_semantic_search(question):
    """
    Combines dense vector search (FAISS) with a lightweight keyword overlap mechanism.
    This helps surface documents that are semantically similar AND share exact keywords.
    """
    print("Running hybrid semantic search...")

    # 1. Convert the user's question into a vector
    q_emb = embedder.embed_query(question)
    q_emb = np.array([q_emb]).astype("float32")

    # 2. Retrieve the top K closest vectors from FAISS
    scores, indices = index_summ.search(q_emb, SEMANTIC_TOP_K)

    candidates = []
    # Tokenize the question for keyword matching
    question_words = set(question.lower().split())

    # 3. Iterate through FAISS results and apply the hybrid scoring logic
    for score, idx in zip(scores[0], indices[0]):
        if idx >= len(docs_summ):
            continue

        # Ignore results that fall below our confidence threshold
        if score < SIMILARITY_THRESHOLD:
            continue

        doc = docs_summ[idx]
        text_lower = doc["text"].lower()

        # Count how many words from the question appear in the document text
        keyword_overlap = sum(word in text_lower for word in question_words)

        # Boost the vector similarity score based on exact keyword matches
        final_score = score + (0.02 * keyword_overlap)

        candidates.append((final_score, doc))

    # Sort the candidates by the new, combined score
    candidates.sort(key=lambda x: x[0], reverse=True)

    # Return only the top 3 results to avoid overwhelming the LLM context
    top_docs = [doc for _, doc in candidates[:3]]

    print("Hybrid results:", [d["id"] for d in top_docs])
    return top_docs


def retrieve_chunks(case_ids, question):
    """
    Once relevant cases are identified (either via KG or Summary Search), 
    this function fetches specific, detailed text chunks belonging to those cases.
    """
    q_emb = embedder.embed_query(question)
    q_emb = np.array([q_emb]).astype("float32")

    # Retrieve a wide net of chunks (top 30)
    scores, indices = index_chunk.search(q_emb, 30)

    selected = []
    seen = set()

    for idx in indices[0]:
        if idx >= len(docs_chunk):
            continue

        doc = docs_chunk[idx]
        doc_id = doc["id"].lower()

        # Filter the FAISS results: Only keep chunks that belong to our target case_ids
        for cid in case_ids:
            if doc_id.startswith(cid.lower()):
                # Ensure we don't add the exact same chunk twice
                if doc_id not in seen:
                    selected.append(doc)
                    seen.add(doc_id)
                break

        # Stop once we have enough detailed context chunks
        if len(selected) >= 6:
            break

    return selected


#  KG RETRIEVAL FUNCTIONS 
# These functions query the Neo4j Knowledge Graph directly.

def kg_get_summary(case_id):
    """Retrieves the summary text for a specific case from the graph."""
    q = """
    MATCH (c:Case)-[:HAS_SUMMARY]->(s:Summary)
    WHERE toLower(c.id) = toLower($cid)
    RETURN s.text AS txt LIMIT 1
    """
    with driver.session(database="legalkg") as session:
        row = session.run(q, cid=case_id).single()
        return row["txt"] if row else None


def kg_get_graph(case_id):
    """Retrieves all relationships connected to a specific case for visualization."""
    q = """
    MATCH (c:Case)-[r]->(n)
    WHERE toLower(c.id) = toLower($cid)
    RETURN c.id AS case_id, type(r) AS rel,
           labels(n)[0] AS label,
           coalesce(n.name, n.value, n.case_id, n.id) AS node_name
    """
    graph_data = []

    with driver.session(database="legalkg") as session:
        results = session.run(q, cid=case_id)
        for row in results:
            graph_data.append({
                "from": row["case_id"],
                "to": row["node_name"],
                "label": row["rel"],
                "type": row["label"]
            })

    return graph_data


#  ENTITY ROUTING FUNCTIONS 
# These functions translate specific entities found in the user's query 
# into relevant case IDs using the Knowledge Graph.

def kg_cases_by_person(name):
    q = """
    MATCH (c:Case)-[:HAS_PETITIONER|HAS_RESPONDENT]->(p)
    WHERE toLower(p.name) CONTAINS toLower($name)
    RETURN DISTINCT c.id AS case_id
    LIMIT 10
    """
    with driver.session(database="legalkg") as session:
        results = session.run(q, name=name)
        return [r["case_id"] for r in results]


def kg_cases_by_judge(name):
    q = """
    MATCH (c:Case)-[:HEARD_BY]->(j:Judge)
    WHERE toLower(j.name) CONTAINS toLower($name)
    RETURN DISTINCT c.id AS case_id
    LIMIT 10
    """
    with driver.session(database="legalkg") as session:
        results = session.run(q, name=name)
        return [r["case_id"] for r in results]


def kg_cases_by_section(section_no):
    q = """
    MATCH (c:Case)-[:INVOLVES_SECTION]->(s:Section)
    WHERE toLower(s.name) CONTAINS toLower($sec)
    RETURN DISTINCT c.id AS case_id
    LIMIT 10
    """
    with driver.session(database="legalkg") as session:
        results = session.run(q, sec=section_no)
        return [r["case_id"] for r in results]


def kg_cases_by_decision(keyword):
    q = """
    MATCH (c:Case)-[:HAS_DECISION]->(d:Decision)
    WHERE toLower(d.name) CONTAINS toLower($kw)
    RETURN DISTINCT c.id AS case_id
    LIMIT 10
    """
    with driver.session(database="legalkg") as session:
        results = session.run(q, kw=keyword)
        return [r["case_id"] for r in results]


#  CASE DETECTION (REGEX) 

# Regex patterns to detect formal case citations in the user query.
CASE_PATTERNS = [
    r"\bCrl\.?A\.?\s*\d+\b",
    r"\bCriminal\s*Appeal\s*No\.?\s*\d+\/\d{4}\b",
]

def normalize_case_id(found: str):
    """Converts a raw match into the standardized 'Crl.A.123' format."""
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


#  LLM GENERATION 

SYSTEM_PROMPT = """
You are LAWPAK, a legal research assistant.
Use ONLY the provided context.
If information is missing, respond with INSUFFICIENT_DOCUMENTATION.
"""

def call_ollama(prompt):
    """Sends the finalized prompt (System + Context + Question) to the LLM."""
    r = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": LLAMA_TEMPERATURE,
                "num_predict": LLAMA_NUM_PREDICT
            }
        }
    )
    return r.json().get("response", "").strip()


#  MAIN RAG PIPELINE 

def run_rag(question):
    """
    The core pipeline orchestrating the entire logic flow:
    1. Parse question for entities/cases.
    2. Retrieve relevant case IDs via Graph or FAISS.
    3. Gather summaries and chunks for those case IDs.
    4. Compile context and send to LLM.
    """
    print("\nQuery:", question)

    detected_case_ids = []

    # --- STEP 1: Rule-Based Routing ---
    
    # 1A. Direct Case Mention
    case_id = detect_case_id(question)
    if case_id:
        print("Case detected:", case_id)
        detected_case_ids = [case_id]

    # 1B. Entity Mentions (if no direct case was found)
    if not detected_case_ids:
        # Check for Sections
        section_match = re.search(r"section\s*(\d+)", question, re.IGNORECASE)
        if section_match:
            section_no = section_match.group(1)
            print("Section detected:", section_no)
            detected_case_ids = kg_cases_by_section(section_no)

        # Check for Judges
        judge_match = re.search(r"(Justice|MR\.?\s*JUSTICE)\s+[A-Za-z\s]+", question, re.IGNORECASE)
        if judge_match:
            judge_name = judge_match.group(0)
            print("Judge detected:", judge_name)
            detected_case_ids = kg_cases_by_judge(judge_name)

        # Check for Decision outcomes
        decision_keywords = ["dismissed", "acquitted", "death", "conviction"]
        for word in decision_keywords:
            if word in question.lower():
                print("Decision keyword detected:", word)
                detected_case_ids = kg_cases_by_decision(word)
                break

        # Check for Persons (Capitalized Words)
        person_match = re.findall(r"[A-Z][a-z]+\s[A-Z][a-z]+", question)
        if person_match:
            name = person_match[0]
            print("Person detected:", name)
            detected_case_ids = kg_cases_by_person(name)


    # --- STEP 2: Context Gathering ---

    context_parts = []
    used_sources = []

    # If our rules found specific cases, use them directly
    if detected_case_ids:
        
        # Get high-level summaries from the KG
        for cid in detected_case_ids[:3]:  
            summary = kg_get_summary(cid)
            if summary:
                context_parts.append(f"[SUMMARY:{cid}]\n{summary}")
                used_sources.append(cid)

        # Get detailed text snippets from FAISS
        chunks = retrieve_chunks(detected_case_ids[:3], question)
        for c in chunks:
            context_parts.append(f"[CHUNK:{c['id']}]\n{c['text']}")
            used_sources.append(c["id"])

    # If our rules failed, fallback to pure Semantic Search
    else:
        # Search the summary vectors
        docs = hybrid_semantic_search(question)

        # If nothing similar is found, abort early
        if not docs:
            return {
                "answer": "INSUFFICIENT_DOCUMENTATION",
                "sources": [],
                "summary": None,
                "kg": None
            }

        detected_case_ids = [d["id"] for d in docs]

        # Add summaries to context
        for d in docs:
            context_parts.append(f"[SUMMARY:{d['id']}]\n{d['text']}")
            used_sources.append(d["id"])

        # Add detailed chunks to context
        chunks = retrieve_chunks(detected_case_ids, question)
        for c in chunks:
            context_parts.append(f"[CHUNK:{c['id']}]\n{c['text']}")
            used_sources.append(c["id"])

    # --- STEP 3: LLM Generation ---

    # Failsafe if context is empty
    if not context_parts:
        return {
            "answer": "INSUFFICIENT_DOCUMENTATION",
            "sources": [],
            "summary": None,
            "kg": None
        }

    # Combine context and truncate to avoid token limits
    full_context = "\n\n".join(context_parts)[:MAX_CONTEXT_CHARS]

    # Construct the final prompt
    prompt = f"""{SYSTEM_PROMPT}

CONTEXT:
{full_context}

QUESTION:
{question}

Provide a structured legal explanation.
"""

    # Generate answer
    answer = call_ollama(prompt)

    # Return structured result
    return {
        "answer": answer,
        "sources": used_sources,
        "summary": None,
        "kg": None
    }


#  COMMAND LINE INTERFACE 

if __name__ == "__main__":
    print("\nHybrid LAWPAK Legal RAG")
    # Interactive loop for testing in the terminal
    while True:
        q = input("\nQuestion: ")
        if q.lower() == "quit":
            break
        
        # Execute the pipeline and print the dictionary output
        print(run_rag(q))
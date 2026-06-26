"""
LawPak Web Server — serves the HTML frontend AND the RAG API.

Run from the LLAMA_4_LEGAL_V2_GENERALIZATION_V2/ directory:
    python3 Web/api.py

Then open:  http://<server-ip>:5000
API:        POST http://<server-ip>:5000/chat   body: {"question":"..."}

This script acts as the main entry point for the LawPak application's web interface.
It spins up a lightweight HTTP server using Python's built-in libraries.
"""

import json
import mimetypes
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from pathlib import Path
from urllib.parse import urlparse, parse_qs

# ── Paths 
# We define paths dynamically based on the location of this script (api.py).
# This ensures the server can be run from anywhere without breaking file references.

# BASE_DIR: The root of the project (one level up from the 'Web' directory).
BASE_DIR   = Path(__file__).resolve().parent.parent   
# STATIC_DIR: The directory containing HTML, CSS, JS, and image assets.
STATIC_DIR = Path(__file__).resolve().parent / "static"  

# The RAG pipeline in the 'scripts' folder expects to be executed from the BASE_DIR
# so it can find 'vector_store/', 'data_clean/', etc. using relative paths.
os.chdir(BASE_DIR)

# Add the project root to sys.path so we can import modules from the 'scripts' package.
sys.path.insert(0, str(BASE_DIR))

# Import the core logic function 'run_rag' from the backend script.
from scripts.testfinal import run_rag  # noqa: E402

PORT = 5001

# ── Chat-history storage (per-user, server-side) 
# We store user chat histories in a simple JSON file on the server.
HISTORY_FILE = BASE_DIR / "Web" / "chat_history.json"

# A lock is necessary because multiple users (threads) might try to read/write 
# the history file at the exact same time, which would corrupt the JSON.
_HISTORY_LOCK = threading.Lock()

def _load_history():
    """Loads the chat history JSON file from disk safely."""
    if not HISTORY_FILE.exists():
        return {}
    try:
        return json.loads(HISTORY_FILE.read_text(encoding="utf-8") or "{}")
    except Exception:
        return {}

def _save_history(data):
    """
    Saves the chat history to disk atomically.
    It writes to a temporary file first, then renames it. 
    This prevents data loss if the server crashes in the middle of writing.
    """
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = HISTORY_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tmp.replace(HISTORY_FILE)

def _norm_email(e):
    """Normalizes an email address for use as a database key."""
    return (e or "").strip().lower()

# ── MIME types ─────
# Explicitly register MIME types to ensure browsers interpret the served files correctly.
mimetypes.add_type("application/javascript", ".mjs")
mimetypes.add_type("image/svg+xml", ".svg")


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """
    A multi-threaded HTTP server.
    
    Why use ThreadingMixIn?
    Standard HTTPServer handles one request at a time. LLaMA inference can take 
    several minutes. If we didn't use threads, the entire website would freeze 
    for everyone while one person's question was being answered.
    """
    daemon_threads = True


class Handler(BaseHTTPRequestHandler):
    """
    Handles incoming HTTP requests (GET, POST, OPTIONS).
    """
    
    # Keep the connection alive for up to 400s to accommodate slow LLM generations
    timeout = 400

    # ── CORS preflight ─
    def do_OPTIONS(self):
        """Handles CORS preflight requests from browsers."""
        self.send_response(200)
        self._cors()
        self.end_headers()

    # ── GET — static files + /status diagnostic ─
    def do_GET(self):
        """Handles GET requests: serving HTML/JS/CSS files and API endpoints."""
        url_path = self.path.split("?")[0]

        # 1. Diagnostic endpoint — hit this to verify the Ollama LLM is reachable
        if url_path == "/status":
            self._ollama_status()
            return

        # 2. Fetch Chat History endpoint
        if url_path == "/history":
            qs = parse_qs(urlparse(self.path).query)
            email = _norm_email((qs.get("email") or [""])[0])
            
            if not email:
                self._json({"error": "email required"}, status=400)
                return
                
            with _HISTORY_LOCK:
                data = _load_history()
            self._json({"history": data.get(email, [])})
            return

        # 3. Serve Static Files (Frontend)
        # Default to index.html if the root is requested.
        if url_path == "/":
            url_path = "/index.html"

        # Prevent directory traversal attacks by using pathlib
        file_path = STATIC_DIR / url_path.lstrip("/")

        if file_path.is_file():
            # Read the file and guess its MIME type
            data = file_path.read_bytes()
            mime = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
            
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        else:
            self._json({"error": "not found"}, status=404)

    def _ollama_status(self):
        """Ping Ollama and return the raw response for diagnostics."""
        import requests as _req
        OLLAMA_URL   = "http://localhost:11434/api/generate"
        OLLAMA_MODEL = "llama3.1"
        try:
            r = _req.post(OLLAMA_URL, json={
                "model": OLLAMA_MODEL,
                "prompt": "Say: OK",
                "stream": False,
                "options": {"num_predict": 5}
            }, timeout=30)
            raw = r.json()
            answer = raw.get("response", "").strip()
            self._json({
                "ollama_reachable": True,
                "model": OLLAMA_MODEL,
                "answer": answer,
                "answer_empty": answer == "",
                "raw_keys": list(raw.keys()),
                "error_in_response": raw.get("error"),
            })
        except Exception as exc:
            self._json({
                "ollama_reachable": False,
                "error": str(exc),
            })

    # ── POST /chat — RAG query ; /history/save ; /history/clear ───────────────
    def do_POST(self):
        """Handles POST requests: saving history, clearing history, and the main Chat API."""
        url_path = self.path.split("?")[0]

        # 1. Save a message to history
        if url_path == "/history/save":
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                email = _norm_email(body.get("email"))
                entry = body.get("entry")
                
                if not email or not isinstance(entry, dict):
                    self._json({"error": "email and entry required"}, status=400)
                    return
                    
                with _HISTORY_LOCK:
                    data = _load_history()
                    data.setdefault(email, []).append(entry)
                    _save_history(data)
                self._json({"ok": True})
            except Exception as exc:
                self._json({"error": str(exc)}, status=500)
            return

        # 2. Clear a user's history
        if url_path == "/history/clear":
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                email = _norm_email(body.get("email"))
                
                if not email:
                    self._json({"error": "email required"}, status=400)
                    return
                    
                with _HISTORY_LOCK:
                    data = _load_history()
                    # Remove the user's key from the dictionary entirely
                    data.pop(email, None)
                    _save_history(data)
                self._json({"ok": True})
            except Exception as exc:
                self._json({"error": str(exc)}, status=500)
            return

        # Ensure only /chat is processed below
        if url_path != "/chat":
            self._json({"error": "not found"}, status=404)
            return

        # 3. Main Chat (RAG) Endpoint
        try:
            # Parse the incoming JSON body
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            
            question = body.get("question", "").strip()
            model    = body.get("model")

            if not question:
                self._json({"error": "question is required"}, status=400)
                return

            # Pass the question to the backend RAG logic
            # This function will query the database, search FAISS, and call the LLM.
            result = run_rag(question, model=model)

            answer  = (result.get("answer") or "").strip()
            sources = result.get("sources", [])
            print(f"[api] answer={repr(answer[:80])}  sources={len(sources)}")

            # Send the structured result back to the frontend
            self._json({
                "answer":  answer,
                "sources": sources,
                "summary": result.get("summary"),
                "kg":      result.get("kg"),
            })

        except Exception as exc:
            # Catch all unexpected errors (e.g., database down, LLM crash)
            self._json({"error": str(exc)}, status=500)

    # ── Helpers ─────
    def _json(self, payload, status=200):
        """Helper to send a JSON response to the client."""
        data = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self._cors()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _cors(self):
        """Appends Cross-Origin Resource Sharing headers."""
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def log_message(self, fmt, *args):
        """Overrides the default logger to make it cleaner."""
        print(f"[lawpak] {self.address_string()} — {fmt % args}")


if __name__ == "__main__":
    # Start the server on all available network interfaces (0.0.0.0)
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"LawPak running at  http://0.0.0.0:{PORT}")
    print(f"Project root:      {BASE_DIR}")
    print(f"Static files:      {STATIC_DIR}")
    server.serve_forever()

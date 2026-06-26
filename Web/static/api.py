"""
LawPak Web Server — serves the HTML frontend AND the RAG API.

Run from the LLAMA_4_LEGAL_V2_GENERALIZATION_V2/ directory:
    python3 Web/api.py

Then open:  http://<server-ip>:5000
API:        POST http://<server-ip>:5000/chat   body: {"question":"..."}
"""

import json
import mimetypes
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from pathlib import Path

# ── Paths 
# This file lives at Web/api.py → parent is LLAMA_4_LEGAL_V2_GENERALIZATION_V2/
BASE_DIR   = Path(__file__).resolve().parent.parent   # project root
STATIC_DIR = Path(__file__).resolve().parent / "static"  # Web/static/

# RAG pipeline uses relative paths (vector_store/, data_clean/) so we must
# run with BASE_DIR as the working directory.
os.chdir(BASE_DIR)

sys.path.insert(0, str(BASE_DIR))
from scripts.testfinal import run_rag  # noqa: E402

PORT = 5000

# ── MIME types 
mimetypes.add_type("application/javascript", ".mjs")
mimetypes.add_type("image/svg+xml", ".svg")


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """Handles each request in a separate thread so LLaMA inference
    (which can take 2-5 min) never blocks other requests or the connection."""
    daemon_threads = True


class Handler(BaseHTTPRequestHandler):
    # Keep the connection alive for up to 400 s (> OLLAMA_TIMEOUT=300)
    timeout = 400

    # ── CORS preflight 
    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.end_headers()

    # ── GET — static files + /status diagnostic 
    def do_GET(self):
        url_path = self.path.split("?")[0]

        # Diagnostic endpoint — hit this to verify Ollama is reachable
        if url_path == "/status":
            self._ollama_status()
            return

        if url_path == "/":
            url_path = "/index.html"

        file_path = STATIC_DIR / url_path.lstrip("/")

        if file_path.is_file():
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
        OLLAMA_MODEL = "llama3.1:8b"
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

    # ── POST /chat — RAG query 
    def do_POST(self):
        if self.path.split("?")[0] != "/chat":
            self._json({"error": "not found"}, status=404)
            return

        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            question = body.get("question", "").strip()

            if not question:
                self._json({"error": "question is required"}, status=400)
                return

            result = run_rag(question)

            answer  = (result.get("answer") or "").strip()
            sources = result.get("sources", [])
            print(f"[api] answer={repr(answer[:80])}  sources={len(sources)}")

            self._json({
                "answer":  answer,
                "sources": sources,
                "summary": result.get("summary"),
                "kg":      result.get("kg"),
            })

        except Exception as exc:
            self._json({"error": str(exc)}, status=500)

    # ── Helpers 
    def _json(self, payload, status=200):
        data = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self._cors()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def log_message(self, fmt, *args):
        print(f"[lawpak] {self.address_string()} — {fmt % args}")


if __name__ == "__main__":
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"LawPak running at  http://0.0.0.0:{PORT}")
    print(f"Project root:      {BASE_DIR}")
    print(f"Static files:      {STATIC_DIR}")
    server.serve_forever()

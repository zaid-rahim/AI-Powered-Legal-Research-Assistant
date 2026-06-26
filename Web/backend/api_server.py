"""
LawPakAI — Flask API Server (SQLite Backend)
All 15 endpoints from legal_db_report.docx Section 7.

Run:  python3 backend/api_server.py
API:  http://localhost:5001
"""

import json
import os
import re
import sys
import time
import threading
import uuid
from datetime import datetime, timezone

from flask import Flask, request, jsonify, g, send_from_directory
from flask_cors import CORS

from database import (
    get_db, init_db, assign_free_plan, update_session_on_message,
    can_user_query, get_user_plan, get_daily_usage_count,
    user_can_upload_documents, add_user_document, mark_document_ready,
    mark_document_failed, get_user_documents, get_user_document,
    delete_user_document_record,
)
from auth import (
    hash_password, verify_password,
    create_access_token, create_refresh_token,
    hash_refresh_token, decode_access_token,
    get_refresh_token_expiry, auth_required, get_client_ip,
)

# ── App Setup ─────────────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})  # Tighten in production

# University server RAG endpoint (existing api.py)
RAG_API_URL = os.environ.get("RAG_API_URL", "http://localhost:5000/chat")

# Document upload config
ALLOWED_EXTENSIONS = {"pdf", "docx", "txt", "text"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

# Ensure the backend dir is on sys.path so document_processor / user_rag can be imported
_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# Static files — serve frontend HTML
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_email(email: str) -> bool:
    return bool(re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email))


def _audit(conn, user_id, action, details=None):
    """Insert audit_log row (append-only, report Section 3.11)."""
    conn.execute(
        "INSERT INTO audit_log (user_id, action, ip_address, user_agent, details, created_at) VALUES (?, ?, ?, ?, ?, datetime('now'))",
        (user_id, action, get_client_ip(), request.headers.get("User-Agent", ""), json.dumps(details or {})),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# STATIC FILE SERVING (frontend pages)
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "service": "lawpakai-doc-api", "port": 5002})


@app.route("/")
def serve_index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:filename>")
def serve_static(filename):
    return send_from_directory(FRONTEND_DIR, filename)


# ═══════════════════════════════════════════════════════════════════════════════
# 0. POST /api/auth/guest_token — Email-only token for localStorage-auth users
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/auth/guest_token", methods=["POST"])
def guest_token():
    """
    Issues a JWT to a user who is already authenticated via the frontend's
    localStorage system (no password required here — the frontend already
    verified identity). Creates the backend user record on first call.
    """
    data  = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    name  = (data.get("name")  or "User").strip()
    plan  = (data.get("plan")  or "free").strip().lower()

    if not email or not _validate_email(email):
        return jsonify({"error": "Valid email required"}), 400

    conn = get_db()
    try:
        user = conn.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
        if user:
            user_id = user["id"]
        else:
            user_id = _uuid()
            conn.execute(
                """INSERT INTO users (id, email, password_hash, full_name, created_at, updated_at)
                   VALUES (?, ?, ?, ?, datetime('now'), datetime('now'))""",
                (user_id, email, "", name),
            )
            assign_free_plan(conn, user_id)
            conn.commit()

        # Sync plan from localStorage → backend DB so upload gate works
        if plan in ("pro", "max"):
            plan_row = conn.execute(
                "SELECT id FROM subscription_plans WHERE name = ?", (plan,)
            ).fetchone()
            if plan_row:
                existing = conn.execute(
                    "SELECT id FROM user_subscriptions WHERE user_id = ? AND status = 'active'",
                    (user_id,)
                ).fetchone()
                if existing:
                    conn.execute(
                        "UPDATE user_subscriptions SET plan_id = ? WHERE id = ?",
                        (plan_row["id"], existing["id"])
                    )
                else:
                    conn.execute(
                        """INSERT INTO user_subscriptions (id, user_id, plan_id, status, created_at)
                           VALUES (?, ?, ?, 'active', datetime('now'))""",
                        (_uuid(), user_id, plan_row["id"])
                    )
                conn.commit()

        return jsonify({"access_token": create_access_token(user_id)}), 200
    except Exception as e:
        conn.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()


# ═══════════════════════════════════════════════════════════════════════════════
# 1. POST /api/auth/register — New user registration
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/auth/register", methods=["POST"])
def register():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password", "")
    full_name = (data.get("full_name") or "").strip()

    # Validation (report Section 9: validate all user input)
    if not email or not _validate_email(email):
        return jsonify({"error": "Valid email is required"}), 400
    if len(password) < 8:
        return jsonify({"error": "Password must be at least 8 characters"}), 400
    if not full_name:
        return jsonify({"error": "Full name is required"}), 400

    conn = get_db()
    try:
        # Check duplicate email
        existing = conn.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
        if existing:
            return jsonify({"error": "Email already registered"}), 409

        # Create user
        user_id = _uuid()
        conn.execute(
            """INSERT INTO users (id, email, password_hash, full_name, created_at, updated_at)
               VALUES (?, ?, ?, ?, datetime('now'), datetime('now'))""",
            (user_id, email, hash_password(password), full_name),
        )

        # Auto-assign free plan (replaces PostgreSQL trigger)
        assign_free_plan(conn, user_id)

        # Create auth session
        refresh_raw = create_refresh_token()
        session_id = _uuid()
        conn.execute(
            """INSERT INTO auth_sessions (id, user_id, refresh_token_hash, device_info, ip_address, expires_at, created_at)
               VALUES (?, ?, ?, ?, ?, ?, datetime('now'))""",
            (session_id, user_id, hash_refresh_token(refresh_raw),
             json.dumps({"user_agent": request.headers.get("User-Agent", "")}),
             get_client_ip(), get_refresh_token_expiry()),
        )

        # Audit log
        _audit(conn, user_id, "register")
        conn.commit()

        return jsonify({
            "user": {"id": user_id, "email": email, "full_name": full_name, "role": "user"},
            "access_token": create_access_token(user_id),
            "refresh_token": refresh_raw,
        }), 201

    except Exception as e:
        conn.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()


# ═══════════════════════════════════════════════════════════════════════════════
# 2. POST /api/auth/login
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/auth/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password", "")

    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400

    conn = get_db()
    try:
        user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()

        if not user or not verify_password(password, user["password_hash"]):
            _audit(conn, user["id"] if user else None, "login_failed", {"email": email})
            conn.commit()
            return jsonify({"error": "Invalid email or password"}), 401

        if not user["is_active"]:
            return jsonify({"error": "Account is disabled"}), 403

        # Update last_login_at
        conn.execute("UPDATE users SET last_login_at = datetime('now') WHERE id = ?", (user["id"],))

        # Create new session
        refresh_raw = create_refresh_token()
        session_id = _uuid()
        conn.execute(
            """INSERT INTO auth_sessions (id, user_id, refresh_token_hash, device_info, ip_address, expires_at, created_at)
               VALUES (?, ?, ?, ?, ?, ?, datetime('now'))""",
            (session_id, user["id"], hash_refresh_token(refresh_raw),
             json.dumps({"user_agent": request.headers.get("User-Agent", "")}),
             get_client_ip(), get_refresh_token_expiry()),
        )

        _audit(conn, user["id"], "login")
        conn.commit()

        return jsonify({
            "user": {"id": user["id"], "email": user["email"], "full_name": user["full_name"], "role": user["role"]},
            "access_token": create_access_token(user["id"], user["role"]),
            "refresh_token": refresh_raw,
        })

    except Exception as e:
        conn.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()


# ═══════════════════════════════════════════════════════════════════════════════
# 3. POST /api/auth/refresh — Refresh access token
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/auth/refresh", methods=["POST"])
def refresh():
    data = request.get_json(silent=True) or {}
    refresh_token = data.get("refresh_token", "")

    if not refresh_token:
        return jsonify({"error": "Refresh token is required"}), 400

    token_hash = hash_refresh_token(refresh_token)
    conn = get_db()
    try:
        session = conn.execute(
            """SELECT s.*, u.role, u.is_active FROM auth_sessions s
               JOIN users u ON s.user_id = u.id
               WHERE s.refresh_token_hash = ? AND s.revoked_at IS NULL""",
            (token_hash,),
        ).fetchone()

        if not session:
            return jsonify({"error": "Invalid refresh token"}), 401

        if session["expires_at"] < _now():
            return jsonify({"error": "Refresh token expired"}), 401

        if not session["is_active"]:
            return jsonify({"error": "Account is disabled"}), 403

        return jsonify({
            "access_token": create_access_token(session["user_id"], session["role"]),
        })

    finally:
        conn.close()


# ═══════════════════════════════════════════════════════════════════════════════
# 4. POST /api/auth/logout — Revoke session
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/auth/logout", methods=["POST"])
@auth_required
def logout():
    data = request.get_json(silent=True) or {}
    refresh_token = data.get("refresh_token", "")

    conn = get_db()
    try:
        if refresh_token:
            # Revoke specific session
            conn.execute(
                "UPDATE auth_sessions SET revoked_at = datetime('now') WHERE refresh_token_hash = ? AND user_id = ?",
                (hash_refresh_token(refresh_token), g.user_id),
            )
        else:
            # Revoke all sessions ("log out everywhere")
            conn.execute(
                "UPDATE auth_sessions SET revoked_at = datetime('now') WHERE user_id = ? AND revoked_at IS NULL",
                (g.user_id,),
            )

        _audit(conn, g.user_id, "logout")
        conn.commit()
        return jsonify({"message": "Logged out successfully"})

    finally:
        conn.close()


# ═══════════════════════════════════════════════════════════════════════════════
# 5. GET /api/me — Current user profile + plan
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/me", methods=["GET"])
@auth_required
def get_me():
    conn = get_db()
    try:
        user = conn.execute("SELECT * FROM users WHERE id = ?", (g.user_id,)).fetchone()
        if not user:
            return jsonify({"error": "User not found"}), 404

        plan = get_user_plan(conn, g.user_id)
        usage_today = get_daily_usage_count(conn, g.user_id)

        return jsonify({
            "user": {
                "id": user["id"],
                "email": user["email"],
                "full_name": user["full_name"],
                "role": user["role"],
                "avatar_url": user["avatar_url"],
                "created_at": user["created_at"],
            },
            "plan": {
                "name": plan["name"] if plan else "free",
                "display_name": plan["display_name"] if plan else "Free",
                "max_queries_per_day": plan["max_queries_per_day"] if plan else 20,
                "queries_used_today": usage_today,
                "billing_cycle": plan["billing_cycle"] if plan else "monthly",
            } if plan else None,
        })

    finally:
        conn.close()


# ═══════════════════════════════════════════════════════════════════════════════
# 6. GET /api/plans — List subscription plans (public)
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/plans", methods=["GET"])
def get_plans():
    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT * FROM subscription_plans WHERE is_active = 1 ORDER BY price_monthly_pkr ASC"
        ).fetchall()
        plans = [dict(r) for r in rows]
        return jsonify({"plans": plans})
    finally:
        conn.close()


# ═══════════════════════════════════════════════════════════════════════════════
# 7. POST /api/subscribe — Subscribe to a plan
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/subscribe", methods=["POST"])
@auth_required
def subscribe():
    data = request.get_json(silent=True) or {}
    plan_name = data.get("plan_name", "")
    billing_cycle = data.get("billing_cycle", "monthly")

    if plan_name not in ("pro", "max"):
        return jsonify({"error": "Invalid plan. Choose 'pro' or 'max'"}), 400
    if billing_cycle not in ("monthly", "yearly"):
        return jsonify({"error": "Invalid billing cycle"}), 400

    conn = get_db()
    try:
        plan = conn.execute("SELECT * FROM subscription_plans WHERE name = ?", (plan_name,)).fetchone()
        if not plan:
            return jsonify({"error": "Plan not found"}), 404

        # Deactivate current subscription
        conn.execute(
            "UPDATE user_subscriptions SET status = 'cancelled', updated_at = datetime('now') WHERE user_id = ? AND status = 'active'",
            (g.user_id,),
        )

        # Create new subscription
        sub_id = _uuid()
        conn.execute(
            """INSERT INTO user_subscriptions
               (id, user_id, plan_id, billing_cycle, status,
                current_period_start, current_period_end, created_at, updated_at)
               VALUES (?, ?, ?, ?, 'active', datetime('now'),
                       datetime('now', '+1 month'), datetime('now'), datetime('now'))""",
            (sub_id, g.user_id, plan["id"], billing_cycle),
        )

        # Record payment (simulated — Stripe integration later)
        amount = plan["price_monthly_pkr"] if billing_cycle == "monthly" else plan["price_yearly_pkr"]
        payment_id = _uuid()
        conn.execute(
            """INSERT INTO payments
               (id, user_id, subscription_id, amount_pkr, status, payment_method, created_at)
               VALUES (?, ?, ?, ?, 'succeeded', 'card', datetime('now'))""",
            (payment_id, g.user_id, sub_id, amount),
        )

        _audit(conn, g.user_id, "plan_upgrade", {"new_plan": plan_name, "billing_cycle": billing_cycle})
        conn.commit()

        return jsonify({
            "message": f"Subscribed to {plan['display_name']} ({billing_cycle})",
            "subscription_id": sub_id,
            "plan": plan_name,
        })

    except Exception as e:
        conn.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()


# ═══════════════════════════════════════════════════════════════════════════════
# 8. POST /api/webhooks/stripe — Stripe webhook (placeholder)
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/webhooks/stripe", methods=["POST"])
def stripe_webhook():
    # TODO: Implement Stripe signature verification when moving to production
    return jsonify({"message": "Webhook received (not yet implemented)"}), 200


# ═══════════════════════════════════════════════════════════════════════════════
# 9. GET /api/sessions — List user's chat sessions (sidebar)
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/sessions", methods=["GET"])
@auth_required
def list_sessions():
    conn = get_db()
    try:
        rows = conn.execute(
            """SELECT id, title, is_pinned, message_count, last_message_at, created_at
               FROM chat_sessions
               WHERE user_id = ? AND is_archived = 0
               ORDER BY is_pinned DESC, last_message_at DESC NULLS LAST""",
            (g.user_id,),
        ).fetchall()
        return jsonify({"sessions": [dict(r) for r in rows]})
    finally:
        conn.close()


# ═══════════════════════════════════════════════════════════════════════════════
# 10. POST /api/sessions — Create new chat session
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/sessions", methods=["POST"])
@auth_required
def create_session():
    data = request.get_json(silent=True) or {}
    title = (data.get("title") or "New Chat").strip()

    conn = get_db()
    try:
        session_id = _uuid()
        conn.execute(
            """INSERT INTO chat_sessions (id, user_id, title, created_at, updated_at)
               VALUES (?, ?, ?, datetime('now'), datetime('now'))""",
            (session_id, g.user_id, title),
        )
        conn.commit()
        return jsonify({"session_id": session_id, "title": title}), 201
    finally:
        conn.close()


# ═══════════════════════════════════════════════════════════════════════════════
# 11. DELETE /api/sessions/<id> — Delete chat session
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/sessions/<session_id>", methods=["DELETE"])
@auth_required
def delete_session(session_id):
    conn = get_db()
    try:
        # Verify ownership
        row = conn.execute(
            "SELECT id FROM chat_sessions WHERE id = ? AND user_id = ?",
            (session_id, g.user_id),
        ).fetchone()
        if not row:
            return jsonify({"error": "Session not found"}), 404

        conn.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))
        conn.commit()
        return jsonify({"message": "Session deleted"})
    finally:
        conn.close()


# ═══════════════════════════════════════════════════════════════════════════════
# 12. GET /api/sessions/<id>/messages — Load conversation history
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/sessions/<session_id>/messages", methods=["GET"])
@auth_required
def get_messages(session_id):
    conn = get_db()
    try:
        # Verify ownership
        session = conn.execute(
            "SELECT id FROM chat_sessions WHERE id = ? AND user_id = ?",
            (session_id, g.user_id),
        ).fetchone()
        if not session:
            return jsonify({"error": "Session not found"}), 404

        rows = conn.execute(
            """SELECT id, role, content, sources, kg_edges, summary, latency_ms, model_used, created_at
               FROM chat_messages WHERE session_id = ? ORDER BY created_at ASC""",
            (session_id,),
        ).fetchall()

        messages = []
        for r in rows:
            msg = dict(r)
            # Parse JSON strings back to objects
            msg["sources"] = json.loads(msg["sources"]) if msg["sources"] else []
            msg["kg_edges"] = json.loads(msg["kg_edges"]) if msg["kg_edges"] else None
            messages.append(msg)

        return jsonify({"messages": messages})
    finally:
        conn.close()


# ═══════════════════════════════════════════════════════════════════════════════
# 13. POST /api/chat — Send message + get AI response
#     This is the core endpoint: auth → limit check → run_rag → save → respond
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/chat", methods=["POST"])
@auth_required
def chat():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    session_id = data.get("session_id")
    model = (data.get("model") or "").strip() or None

    if not question:
        return jsonify({"error": "Question is required"}), 400

    conn = get_db()
    try:
        # Step 1: Check daily usage limit (report Section 8.2, step 4)
        if not can_user_query(conn, g.user_id):
            plan = get_user_plan(conn, g.user_id)
            limit = plan["max_queries_per_day"] if plan else 20
            return jsonify({
                "error": "Daily query limit reached",
                "limit": limit,
                "plan": plan["name"] if plan else "free",
                "upgrade_url": "/pricing.html",
            }), 429

        # Step 2: Create session if not provided
        if not session_id:
            session_id = _uuid()
            title = question[:80] + ("..." if len(question) > 80 else "")
            conn.execute(
                """INSERT INTO chat_sessions (id, user_id, title, created_at, updated_at)
                   VALUES (?, ?, ?, datetime('now'), datetime('now'))""",
                (session_id, g.user_id, title),
            )
        else:
            # Verify session ownership
            session = conn.execute(
                "SELECT id FROM chat_sessions WHERE id = ? AND user_id = ?",
                (session_id, g.user_id),
            ).fetchone()
            if not session:
                return jsonify({"error": "Session not found"}), 404

        # Step 3: Save user message
        user_msg_id = _uuid()
        now = _now()
        conn.execute(
            """INSERT INTO chat_messages (id, session_id, role, content, created_at)
               VALUES (?, ?, 'user', ?, ?)""",
            (user_msg_id, session_id, question, now),
        )
        update_session_on_message(conn, session_id, now)

        # Step 4: Call RAG pipeline (existing api.py on university server)
        import requests as http_client
        start_time = time.time()
        try:
            rag_payload = {"question": question}
            if model:
                rag_payload["model"] = model
            rag_response = http_client.post(
                RAG_API_URL,
                json=rag_payload,
                timeout=300,  # 5 min timeout for LLaMA inference
            )
            rag_data = rag_response.json()
        except Exception as rag_err:
            rag_data = {
                "answer": f"AI service temporarily unavailable: {str(rag_err)}",
                "sources": [],
                "summary": None,
                "kg": None,
            }

        latency_ms = int((time.time() - start_time) * 1000)

        answer = rag_data.get("answer", "")
        sources = rag_data.get("sources", [])
        summary = rag_data.get("summary")
        kg = rag_data.get("kg")

        # Step 5: Save AI response (report Section 8.2, steps 7-8)
        ai_msg_id = _uuid()
        ai_now = _now()
        conn.execute(
            """INSERT INTO chat_messages
               (id, session_id, role, content, sources, kg_edges, summary, latency_ms, created_at)
               VALUES (?, ?, 'assistant', ?, ?, ?, ?, ?, ?)""",
            (ai_msg_id, session_id, answer,
             json.dumps(sources), json.dumps(kg) if kg else None,
             summary, latency_ms, ai_now),
        )
        update_session_on_message(conn, session_id, ai_now)

        # Step 6: Track usage (report Section 8.2, step 9)
        conn.execute(
            """INSERT INTO usage_tracking
               (id, user_id, event_type, session_id, message_id, latency_ms, metadata, created_at)
               VALUES (?, ?, 'ai_query', ?, ?, ?, ?, datetime('now'))""",
            (_uuid(), g.user_id, session_id, ai_msg_id, latency_ms,
             json.dumps({"question_length": len(question), "sources_count": len(sources)})),
        )

        conn.commit()

        return jsonify({
            "answer": answer,
            "sources": sources,
            "summary": summary,
            "kg": kg,
            "session_id": session_id,
            "message_id": ai_msg_id,
            "latency_ms": latency_ms,
        })

    except Exception as e:
        conn.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()


# ═══════════════════════════════════════════════════════════════════════════════
# 14. POST /api/feedback — Rate an AI response
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/feedback", methods=["POST"])
@auth_required
def submit_feedback():
    data = request.get_json(silent=True) or {}
    message_id = data.get("message_id", "")
    rating = data.get("rating")
    comment = data.get("comment", "")

    if not message_id or rating not in (-1, 1):
        return jsonify({"error": "message_id and rating (-1 or 1) required"}), 400

    conn = get_db()
    try:
        fb_id = _uuid()
        conn.execute(
            """INSERT OR REPLACE INTO feedback (id, user_id, message_id, rating, comment, created_at)
               VALUES (?, ?, ?, ?, ?, datetime('now'))""",
            (fb_id, g.user_id, message_id, rating, comment),
        )
        conn.commit()
        return jsonify({"message": "Feedback recorded"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()


# ═══════════════════════════════════════════════════════════════════════════════
# 15. POST /api/contact — Contact form submission (public)
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/contact", methods=["POST"])
def contact():
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    email = (data.get("email") or "").strip()
    subject = (data.get("subject") or "").strip()
    message = (data.get("message") or "").strip()

    if not name or not email or not message:
        return jsonify({"error": "Name, email, and message are required"}), 400
    if not _validate_email(email):
        return jsonify({"error": "Invalid email format"}), 400

    conn = get_db()
    try:
        conn.execute(
            """INSERT INTO contact_submissions (id, name, email, subject, message, created_at)
               VALUES (?, ?, ?, ?, ?, datetime('now'))""",
            (_uuid(), name, email, subject, message),
        )
        conn.commit()
        return jsonify({"message": "Contact form submitted successfully"}), 201
    finally:
        conn.close()


# ═══════════════════════════════════════════════════════════════════════════════
# BONUS: GET /api/usage — User's usage stats
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/usage", methods=["GET"])
@auth_required
def get_usage():
    conn = get_db()
    try:
        plan = get_user_plan(conn, g.user_id)
        today_count = get_daily_usage_count(conn, g.user_id)

        total = conn.execute(
            "SELECT COUNT(*) as cnt FROM usage_tracking WHERE user_id = ? AND event_type = 'ai_query'",
            (g.user_id,),
        ).fetchone()["cnt"]

        return jsonify({
            "today": today_count,
            "total": total,
            "daily_limit": plan["max_queries_per_day"] if plan else 20,
            "plan": plan["name"] if plan else "free",
        })
    finally:
        conn.close()


# ═══════════════════════════════════════════════════════════════════════════════
# DOCUMENT UPLOAD ENDPOINTS (Pro / Max plan only)
# ═══════════════════════════════════════════════════════════════════════════════

def _allowed_file(filename: str) -> bool:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return ext in ALLOWED_EXTENSIONS


def _process_document_background(user_id: str, doc_id: str, file_bytes: bytes, filename: str):
    """
    Background thread: extract → chunk → embed → FAISS.
    Updates the user_documents row to 'ready' or 'failed' when done.
    Opens its own DB connection (Flask g is not available in threads).
    """
    from document_processor import process_document
    from database import get_db, mark_document_ready, mark_document_failed

    conn = get_db()
    try:
        meta = process_document(user_id, doc_id, file_bytes, filename)
        mark_document_ready(conn, doc_id, meta["chunk_count"])
        conn.commit()
        print(f"[doc] {doc_id} ready — {meta['chunk_count']} chunks", flush=True)
    except Exception as exc:
        print(f"[doc] {doc_id} FAILED — {exc}", flush=True)
        try:
            mark_document_failed(conn, doc_id, str(exc)[:500])
            conn.commit()
        except Exception:
            pass
    finally:
        conn.close()


@app.route("/api/documents/upload", methods=["POST"])
@auth_required
def upload_document():
    """
    POST /api/documents/upload   multipart/form-data  field: file
    Supported: PDF, DOCX, TXT  ·  Max 20 MB  ·  Pro / Max plan only.

    Returns 202 immediately with {doc_id, status:"processing"}.
    Processing runs in a background thread.
    Poll GET /api/documents/<doc_id>/status until status is 'ready' or 'failed'.
    """
    conn = get_db()
    try:
        if not user_can_upload_documents(conn, g.user_id):
            return jsonify({
                "error": "Document upload requires a Pro or Max plan",
                "upgrade_url": "/pricing.html",
            }), 403

        if "file" not in request.files:
            return jsonify({"error": "No file part in request"}), 400

        f = request.files["file"]
        if not f or not f.filename:
            return jsonify({"error": "No file selected"}), 400

        if not _allowed_file(f.filename):
            return jsonify({
                "error": f"Unsupported file type. Allowed: PDF, DOCX, TXT",
            }), 400

        file_bytes = f.read()
        file_size = len(file_bytes)
        if file_size == 0:
            return jsonify({"error": "Uploaded file is empty"}), 400

        doc_id = _uuid()
        safe_name = os.path.basename(f.filename)

        # Insert DB record as 'processing' before starting thread
        add_user_document(conn, doc_id, g.user_id, safe_name, file_size)
        _audit(conn, g.user_id, "document_upload_start", {"doc_id": doc_id, "filename": safe_name})
        conn.commit()

        # Kick off background processing — returns immediately
        t = threading.Thread(
            target=_process_document_background,
            args=(g.user_id, doc_id, file_bytes, safe_name),
            daemon=True,
        )
        t.start()

        return jsonify({
            "doc_id": doc_id,
            "filename": safe_name,
            "file_size": file_size,
            "status": "processing",
            "message": "Document is being processed. Poll /api/documents/{doc_id}/status.",
        }), 202

    except Exception as e:
        conn.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()


@app.route("/api/documents/<doc_id>/status", methods=["GET"])
@auth_required
def document_status(doc_id):
    """
    GET /api/documents/<doc_id>/status
    Returns current processing status: processing | ready | failed
    Poll this endpoint after upload until status != 'processing'.
    """
    conn = get_db()
    try:
        doc = get_user_document(conn, doc_id, g.user_id)
        if not doc:
            return jsonify({"error": "Document not found"}), 404

        from document_processor import read_doc_status
        progress_info = read_doc_status(g.user_id, doc_id)

        return jsonify({
            "doc_id": doc_id,
            "status": doc["status"],
            "filename": doc["filename"],
            "chunk_count": doc["chunk_count"],
            "file_size": doc["file_size"],
            "error_msg": doc.get("error_msg"),
            "created_at": doc["created_at"],
            "progress": progress_info.get("progress", 0),
            "stage": progress_info.get("stage", ""),
        })
    finally:
        conn.close()


@app.route("/api/documents", methods=["GET"])
@auth_required
def list_documents():
    """GET /api/documents — list all documents uploaded by the current user."""
    conn = get_db()
    try:
        docs = get_user_documents(conn, g.user_id)
        return jsonify({"documents": docs})
    finally:
        conn.close()


@app.route("/api/documents/<doc_id>", methods=["DELETE"])
@auth_required
def delete_document(doc_id):
    """DELETE /api/documents/<doc_id> — remove DB record and FAISS index files."""
    conn = get_db()
    try:
        doc = get_user_document(conn, doc_id, g.user_id)
        if not doc:
            return jsonify({"error": "Document not found"}), 404

        delete_user_document_record(conn, doc_id, g.user_id)

        from document_processor import delete_document_files
        delete_document_files(g.user_id, doc_id)

        _audit(conn, g.user_id, "document_delete", {"doc_id": doc_id, "filename": doc["filename"]})
        conn.commit()
        return jsonify({"message": "Document deleted"})
    except Exception as e:
        conn.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()


@app.route("/api/documents/<doc_id>/chat", methods=["POST"])
@auth_required
def chat_with_document(doc_id):
    """
    POST /api/documents/<doc_id>/chat
    Body: {"question": str, "session_id": str (optional), "model": str (optional)}
    Retrieves relevant chunks from user's FAISS index and generates answer via LLaMA.
    Pro / Max plan only.
    """
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    session_id = data.get("session_id")
    model = (data.get("model") or "").strip() or None

    if not question:
        return jsonify({"error": "Question is required"}), 400

    conn = get_db()
    try:
        if not user_can_upload_documents(conn, g.user_id):
            return jsonify({"error": "Document chat requires a Pro or Max plan"}), 403

        doc = get_user_document(conn, doc_id, g.user_id)
        if not doc:
            return jsonify({"error": "Document not found"}), 404
        if doc["status"] == "processing":
            return jsonify({"error": "Document is still being processed. Please wait."}), 409
        if doc["status"] == "failed":
            return jsonify({
                "error": "Document processing failed. Please re-upload.",
                "details": doc.get("error_msg", ""),
            }), 409

        # Daily query limit applies to document queries too
        if not can_user_query(conn, g.user_id):
            plan = get_user_plan(conn, g.user_id)
            return jsonify({
                "error": "Daily query limit reached",
                "limit": plan["max_queries_per_day"] if plan else 20,
                "upgrade_url": "/pricing.html",
            }), 429

        # Session management
        if not session_id:
            session_id = _uuid()
            title = f"[Doc] {question[:60]}" + ("..." if len(question) > 60 else "")
            conn.execute(
                "INSERT INTO chat_sessions (id, user_id, title, created_at, updated_at) VALUES (?, ?, ?, datetime('now'), datetime('now'))",
                (session_id, g.user_id, title),
            )
        else:
            row = conn.execute(
                "SELECT id FROM chat_sessions WHERE id = ? AND user_id = ?",
                (session_id, g.user_id),
            ).fetchone()
            if not row:
                return jsonify({"error": "Session not found"}), 404

        # Save user turn
        user_msg_id = _uuid()
        now = _now()
        conn.execute(
            "INSERT INTO chat_messages (id, session_id, role, content, created_at) VALUES (?, ?, 'user', ?, ?)",
            (user_msg_id, session_id, question, now),
        )
        update_session_on_message(conn, session_id, now)

        # FAISS retrieval + LLaMA generation
        start_time = time.time()
        try:
            from document_processor import query_document
            from user_rag import run_user_doc_rag
            doc_data = query_document(g.user_id, doc_id, question)
            rag_data = run_user_doc_rag(
                question, doc_data, model=model, doc_name=doc["filename"]
            )
        except Exception as rag_err:
            rag_data = {
                "answer": f"Error querying document: {rag_err}",
                "sources": [],
                "summary": None,
                "kg": None,
            }

        latency_ms = int((time.time() - start_time) * 1000)
        answer = rag_data.get("answer", "")
        sources = rag_data.get("sources", [])

        # Save AI turn
        ai_msg_id = _uuid()
        ai_now = _now()
        conn.execute(
            """INSERT INTO chat_messages
               (id, session_id, role, content, sources, summary, latency_ms, model_used, created_at)
               VALUES (?, ?, 'assistant', ?, ?, ?, ?, ?, ?)""",
            (ai_msg_id, session_id, answer, json.dumps(sources),
             rag_data.get("summary"), latency_ms, model or "llama3.1:8b", ai_now),
        )
        update_session_on_message(conn, session_id, ai_now)

        # Track usage
        conn.execute(
            """INSERT INTO usage_tracking
               (id, user_id, event_type, session_id, message_id, latency_ms, metadata, created_at)
               VALUES (?, ?, 'ai_query', ?, ?, ?, ?, datetime('now'))""",
            (_uuid(), g.user_id, session_id, ai_msg_id, latency_ms,
             json.dumps({"mode": "document", "doc_id": doc_id, "q_len": len(question)})),
        )
        conn.commit()

        return jsonify({
            "answer": answer,
            "sources": sources,
            "summary": rag_data.get("summary"),
            "kg": rag_data.get("kg"),
            "session_id": session_id,
            "message_id": ai_msg_id,
            "latency_ms": latency_ms,
            "doc_id": doc_id,
            "doc_name": doc["filename"],
        })

    except Exception as e:
        conn.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()


# ═══════════════════════════════════════════════════════════════════════════════
# STARTUP
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    init_db()
    print("\n" + "=" * 60)
    print("  LawPakAI API Server (SQLite)")
    print("=" * 60)
    port = int(os.environ.get("PORT", 5002))
    print(f"  API:      http://localhost:{port}/api/")
    print(f"  Frontend: http://localhost:{port}/")
    print(f"  RAG:      {RAG_API_URL}")
    print(f"  Database: backend/lawpakai.db")
    print("=" * 60 + "\n")
    app.run(host="0.0.0.0", port=port, debug=False)

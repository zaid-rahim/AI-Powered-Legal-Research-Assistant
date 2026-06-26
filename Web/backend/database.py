"""
LawPakAI — SQLite Database Layer
All 11 tables from legal_db_report.docx, adapted for SQLite.
Triggers replaced with Python-side logic. UUIDs generated in Python.
"""

import sqlite3
import uuid
import os
from datetime import datetime, timezone
from pathlib import Path

DB_DIR = Path(__file__).resolve().parent
DB_PATH = DB_DIR / "lawpakai.db"


def _now() -> str:
    """ISO 8601 UTC timestamp string for SQLite."""
    return datetime.now(timezone.utc).isoformat()


def _uuid() -> str:
    return str(uuid.uuid4())


def get_db() -> sqlite3.Connection:
    """Return a connection with row_factory and WAL mode enabled."""
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


# ═══════════════════════════════════════════════════════════════
# SCHEMA — 11 tables matching legal_db_report.docx
# ═══════════════════════════════════════════════════════════════

SCHEMA_SQL = """
-- ── 1. users ─────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS users (
    id              TEXT PRIMARY KEY,
    email           TEXT UNIQUE NOT NULL,
    password_hash   TEXT NOT NULL,
    full_name       TEXT NOT NULL,
    role            TEXT NOT NULL DEFAULT 'user' CHECK (role IN ('user', 'admin')),
    is_active       INTEGER NOT NULL DEFAULT 1,
    email_verified  INTEGER NOT NULL DEFAULT 0,
    avatar_url      TEXT,
    last_login_at   TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

-- ── 2. auth_sessions ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS auth_sessions (
    id                  TEXT PRIMARY KEY,
    user_id             TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    refresh_token_hash  TEXT UNIQUE NOT NULL,
    device_info         TEXT,
    ip_address          TEXT,
    expires_at          TEXT NOT NULL,
    revoked_at          TEXT,
    created_at          TEXT NOT NULL DEFAULT (datetime('now'))
);

-- ── 3. subscription_plans ────────────────────────────────────
CREATE TABLE IF NOT EXISTS subscription_plans (
    id                        TEXT PRIMARY KEY,
    name                      TEXT UNIQUE NOT NULL,
    display_name              TEXT NOT NULL,
    price_monthly_pkr         INTEGER NOT NULL DEFAULT 0,
    price_yearly_pkr          INTEGER NOT NULL DEFAULT 0,
    max_queries_per_day       INTEGER NOT NULL,
    max_chat_sessions         INTEGER NOT NULL,
    feature_case_access       TEXT NOT NULL DEFAULT 'basic',
    feature_document_drafting INTEGER NOT NULL DEFAULT 0,
    feature_bilingual         TEXT NOT NULL DEFAULT 'basic',
    feature_citations         INTEGER NOT NULL DEFAULT 0,
    feature_priority_access   INTEGER NOT NULL DEFAULT 0,
    feature_deep_analysis     INTEGER NOT NULL DEFAULT 0,
    feature_dedicated_support INTEGER NOT NULL DEFAULT 0,
    feature_community         INTEGER NOT NULL DEFAULT 1,
    stripe_price_id_monthly   TEXT,
    stripe_price_id_yearly    TEXT,
    is_active                 INTEGER NOT NULL DEFAULT 1,
    created_at                TEXT NOT NULL DEFAULT (datetime('now'))
);

-- ── 4. user_subscriptions ────────────────────────────────────
CREATE TABLE IF NOT EXISTS user_subscriptions (
    id                      TEXT PRIMARY KEY,
    user_id                 TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    plan_id                 TEXT NOT NULL REFERENCES subscription_plans(id),
    billing_cycle           TEXT NOT NULL DEFAULT 'monthly' CHECK (billing_cycle IN ('monthly', 'yearly')),
    status                  TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active','cancelled','expired','past_due')),
    stripe_subscription_id  TEXT,
    current_period_start    TEXT NOT NULL,
    current_period_end      TEXT NOT NULL,
    cancel_at_period_end    INTEGER NOT NULL DEFAULT 0,
    created_at              TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at              TEXT NOT NULL DEFAULT (datetime('now'))
);

-- ── 5. payments ──────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS payments (
    id                        TEXT PRIMARY KEY,
    user_id                   TEXT NOT NULL REFERENCES users(id),
    subscription_id           TEXT REFERENCES user_subscriptions(id),
    stripe_payment_intent_id  TEXT UNIQUE,
    stripe_invoice_id         TEXT,
    amount_pkr                INTEGER NOT NULL,
    currency                  TEXT NOT NULL DEFAULT 'PKR',
    status                    TEXT NOT NULL CHECK (status IN ('succeeded','failed','pending','refunded')),
    payment_method            TEXT,
    receipt_url               TEXT,
    metadata                  TEXT NOT NULL DEFAULT '{}',
    created_at                TEXT NOT NULL DEFAULT (datetime('now'))
);

-- ── 6. chat_sessions ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS chat_sessions (
    id              TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title           TEXT,
    is_archived     INTEGER NOT NULL DEFAULT 0,
    is_pinned       INTEGER NOT NULL DEFAULT 0,
    message_count   INTEGER NOT NULL DEFAULT 0,
    last_message_at TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

-- ── 7. chat_messages ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS chat_messages (
    id          TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role        TEXT NOT NULL CHECK (role IN ('user','assistant','system')),
    content     TEXT NOT NULL,
    sources     TEXT NOT NULL DEFAULT '[]',
    kg_edges    TEXT,
    summary     TEXT,
    latency_ms  INTEGER,
    model_used  TEXT DEFAULT 'llama3.1:8b',
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

-- ── 8. feedback ──────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS feedback (
    id         TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL REFERENCES users(id),
    message_id TEXT NOT NULL REFERENCES chat_messages(id) ON DELETE CASCADE,
    rating     INTEGER NOT NULL CHECK (rating IN (-1, 1)),
    comment    TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(user_id, message_id)
);

-- ── 9. contact_submissions ───────────────────────────────────
CREATE TABLE IF NOT EXISTS contact_submissions (
    id         TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    email      TEXT NOT NULL,
    subject    TEXT,
    message    TEXT NOT NULL,
    status     TEXT NOT NULL DEFAULT 'new' CHECK (status IN ('new','read','replied')),
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- ── 10. usage_tracking ───────────────────────────────────────
CREATE TABLE IF NOT EXISTS usage_tracking (
    id          TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL REFERENCES users(id),
    event_type  TEXT NOT NULL,
    session_id  TEXT REFERENCES chat_sessions(id),
    message_id  TEXT REFERENCES chat_messages(id),
    latency_ms  INTEGER,
    metadata    TEXT NOT NULL DEFAULT '{}',
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

-- ── 11. audit_log ────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS audit_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     TEXT REFERENCES users(id),
    action      TEXT NOT NULL,
    ip_address  TEXT,
    user_agent  TEXT,
    details     TEXT NOT NULL DEFAULT '{}',
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

-- ── 12. user_documents ───────────────────────────────────────
CREATE TABLE IF NOT EXISTS user_documents (
    id          TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    filename    TEXT NOT NULL,
    file_size   INTEGER NOT NULL DEFAULT 0,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    status      TEXT NOT NULL DEFAULT 'processing'
                    CHECK (status IN ('processing', 'ready', 'failed')),
    error_msg   TEXT,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

# ═══════════════════════════════════════════════════════════════
# INDEXES — from report Section 4
# ═══════════════════════════════════════════════════════════════

INDEXES_SQL = """
CREATE INDEX IF NOT EXISTS idx_auth_user        ON auth_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_auth_expires     ON auth_sessions(expires_at);
CREATE INDEX IF NOT EXISTS idx_auth_token       ON auth_sessions(refresh_token_hash);
CREATE INDEX IF NOT EXISTS idx_subs_user_status ON user_subscriptions(user_id, status);
CREATE INDEX IF NOT EXISTS idx_pay_user         ON payments(user_id);
CREATE INDEX IF NOT EXISTS idx_pay_stripe       ON payments(stripe_payment_intent_id);
CREATE INDEX IF NOT EXISTS idx_chats_user       ON chat_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_chats_last_msg   ON chat_sessions(user_id, last_message_at DESC);
CREATE INDEX IF NOT EXISTS idx_msgs_session     ON chat_messages(session_id);
CREATE INDEX IF NOT EXISTS idx_msgs_created     ON chat_messages(session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_fb_message       ON feedback(message_id);
CREATE INDEX IF NOT EXISTS idx_usage_user_date  ON usage_tracking(user_id, created_at);
CREATE INDEX IF NOT EXISTS idx_usage_type       ON usage_tracking(event_type);
CREATE INDEX IF NOT EXISTS idx_audit_user       ON audit_log(user_id, created_at);
CREATE INDEX IF NOT EXISTS idx_audit_action     ON audit_log(action);
CREATE INDEX IF NOT EXISTS idx_docs_user        ON user_documents(user_id, created_at DESC);
"""

# ═══════════════════════════════════════════════════════════════
# SEED DATA — 3 subscription plans from pricing.html
# ═══════════════════════════════════════════════════════════════

SEED_PLANS_SQL = """
INSERT OR IGNORE INTO subscription_plans
    (id, name, display_name, price_monthly_pkr, price_yearly_pkr,
     max_queries_per_day, max_chat_sessions,
     feature_case_access, feature_document_drafting,
     feature_bilingual, feature_citations,
     feature_priority_access, feature_deep_analysis,
     feature_dedicated_support, feature_community)
VALUES
    ('{free_id}', 'free', 'Free', 0, 0,
     20, 5,
     'basic', 0, 'basic', 0, 0, 0, 0, 1),
    ('{pro_id}', 'pro', 'Pro', 1600, 14400,
     -1, 50,
     'full_50k', 1, 'full', 1, 0, 0, 0, 1),
    ('{max_id}', 'max', 'Max', 3600, 32400,
     -1, -1,
     'full_50k_extended', 1, 'full', 1, 1, 1, 1, 1);
"""


# ═══════════════════════════════════════════════════════════════
# PYTHON-SIDE LOGIC (replaces PostgreSQL triggers/functions)
# ═══════════════════════════════════════════════════════════════

def assign_free_plan(conn: sqlite3.Connection, user_id: str) -> str:
    """Auto-assign free plan on registration (replaces trg_user_free_plan)."""
    row = conn.execute(
        "SELECT id FROM subscription_plans WHERE name = 'free' LIMIT 1"
    ).fetchone()
    if not row:
        raise RuntimeError("Free plan not found — run init_db() first")

    sub_id = _uuid()
    conn.execute(
        """INSERT INTO user_subscriptions
           (id, user_id, plan_id, billing_cycle, status,
            current_period_start, current_period_end)
           VALUES (?, ?, ?, 'monthly', 'active', datetime('now'),
                   datetime('now', '+100 years'))""",
        (sub_id, user_id, row["id"]),
    )
    return sub_id


def update_session_on_message(conn: sqlite3.Connection, session_id: str, created_at: str):
    """Update chat_session stats after inserting a message (replaces trg_message_insert)."""
    conn.execute(
        """UPDATE chat_sessions
           SET message_count = message_count + 1,
               last_message_at = ?,
               updated_at = datetime('now')
           WHERE id = ?""",
        (created_at, session_id),
    )


def can_user_query(conn: sqlite3.Connection, user_id: str) -> bool:
    """Check if user is within their daily query limit (replaces can_user_query() function)."""
    row = conn.execute(
        """SELECT sp.max_queries_per_day
           FROM user_subscriptions us
           JOIN subscription_plans sp ON us.plan_id = sp.id
           WHERE us.user_id = ? AND us.status = 'active'
           ORDER BY us.created_at DESC LIMIT 1""",
        (user_id,),
    ).fetchone()

    if not row:
        return False  # no active subscription

    daily_limit = row["max_queries_per_day"]
    if daily_limit == -1:
        return True  # unlimited

    used = conn.execute(
        """SELECT COUNT(*) as cnt FROM usage_tracking
           WHERE user_id = ? AND event_type = 'ai_query'
             AND date(created_at) = date('now')""",
        (user_id,),
    ).fetchone()["cnt"]

    return used < daily_limit


def get_user_plan(conn: sqlite3.Connection, user_id: str) -> dict | None:
    """Get the user's active subscription plan details."""
    row = conn.execute(
        """SELECT sp.*, us.billing_cycle, us.status as sub_status,
                  us.current_period_end, us.id as subscription_id
           FROM user_subscriptions us
           JOIN subscription_plans sp ON us.plan_id = sp.id
           WHERE us.user_id = ? AND us.status = 'active'
           ORDER BY us.created_at DESC LIMIT 1""",
        (user_id,),
    ).fetchone()
    return dict(row) if row else None


def get_daily_usage_count(conn: sqlite3.Connection, user_id: str) -> int:
    """Count today's AI queries for a user."""
    row = conn.execute(
        """SELECT COUNT(*) as cnt FROM usage_tracking
           WHERE user_id = ? AND event_type = 'ai_query'
             AND date(created_at) = date('now')""",
        (user_id,),
    ).fetchone()
    return row["cnt"] if row else 0


# ── User document helpers ────────────────────────────────────────────────────

def user_can_upload_documents(conn: sqlite3.Connection, user_id: str) -> bool:
    """Return True only for pro and max plan subscribers."""
    row = conn.execute(
        """SELECT sp.name FROM user_subscriptions us
           JOIN subscription_plans sp ON us.plan_id = sp.id
           WHERE us.user_id = ? AND us.status = 'active'
           ORDER BY us.created_at DESC LIMIT 1""",
        (user_id,),
    ).fetchone()
    return row is not None and row["name"] in ("pro", "max")


def add_user_document(
    conn: sqlite3.Connection,
    doc_id: str,
    user_id: str,
    filename: str,
    file_size: int,
) -> None:
    conn.execute(
        """INSERT INTO user_documents (id, user_id, filename, file_size, status, created_at)
           VALUES (?, ?, ?, ?, 'processing', datetime('now'))""",
        (doc_id, user_id, filename, file_size),
    )


def mark_document_ready(
    conn: sqlite3.Connection, doc_id: str, chunk_count: int
) -> None:
    conn.execute(
        "UPDATE user_documents SET status='ready', chunk_count=? WHERE id=?",
        (chunk_count, doc_id),
    )


def mark_document_failed(
    conn: sqlite3.Connection, doc_id: str, error_msg: str
) -> None:
    conn.execute(
        "UPDATE user_documents SET status='failed', error_msg=? WHERE id=?",
        (error_msg, doc_id),
    )


def get_user_documents(conn: sqlite3.Connection, user_id: str) -> list:
    rows = conn.execute(
        """SELECT id, filename, file_size, chunk_count, status, error_msg, created_at
           FROM user_documents WHERE user_id = ? ORDER BY created_at DESC""",
        (user_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_user_document(
    conn: sqlite3.Connection, doc_id: str, user_id: str
) -> dict | None:
    row = conn.execute(
        "SELECT * FROM user_documents WHERE id = ? AND user_id = ?",
        (doc_id, user_id),
    ).fetchone()
    return dict(row) if row else None


def delete_user_document_record(
    conn: sqlite3.Connection, doc_id: str, user_id: str
) -> bool:
    cur = conn.execute(
        "DELETE FROM user_documents WHERE id = ? AND user_id = ?",
        (doc_id, user_id),
    )
    return cur.rowcount > 0


# ═══════════════════════════════════════════════════════════════
# INIT
# ═══════════════════════════════════════════════════════════════

def init_db():
    """Create all tables, indexes, and seed subscription plans."""
    conn = get_db()
    try:
        conn.executescript(SCHEMA_SQL)
        conn.executescript(INDEXES_SQL)

        # Seed plans with stable UUIDs (so re-runs don't duplicate)
        existing = conn.execute("SELECT COUNT(*) as cnt FROM subscription_plans").fetchone()["cnt"]
        if existing == 0:
            seed = SEED_PLANS_SQL.format(
                free_id=_uuid(),
                pro_id=_uuid(),
                max_id=_uuid(),
            )
            conn.executescript(seed)

        conn.commit()
        print(f"[database] Initialized SQLite at {DB_PATH}")
        print(f"[database] Tables: 11 | Indexes: 15 | Plans seeded: {3 if existing == 0 else 'already exist'}")
    finally:
        conn.close()


if __name__ == "__main__":
    init_db()

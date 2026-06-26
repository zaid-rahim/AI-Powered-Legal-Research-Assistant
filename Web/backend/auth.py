"""
LawPakAI — Authentication Utilities
JWT access/refresh tokens, bcrypt password hashing, middleware.
Follows report Section 9 security checklist.
"""

import hashlib
import os
import secrets
from datetime import datetime, timedelta, timezone
from functools import wraps

import bcrypt
import jwt
from flask import request, jsonify, g

# ── Config (from env vars, with safe defaults for dev) ────────────────────────

SECRET_KEY = os.environ.get("LAWPAKAI_SECRET_KEY", "dev-secret-change-in-production")
ACCESS_TOKEN_EXPIRY_MINUTES = 120      # 2 hours — covers long doc processing jobs
REFRESH_TOKEN_EXPIRY_DAYS = 30         # Long-lived (report Section 9)
BCRYPT_COST = 12                        # report: "cost >= 12"


# ═══════════════════════════════════════════════════════════════
# PASSWORD HASHING
# ═══════════════════════════════════════════════════════════════

def hash_password(plain: str) -> str:
    """Hash password with bcrypt. Returns string suitable for DB storage."""
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt(rounds=BCRYPT_COST)).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """Verify a plaintext password against a bcrypt hash."""
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


# ═══════════════════════════════════════════════════════════════
# JWT TOKENS
# ═══════════════════════════════════════════════════════════════

def create_access_token(user_id: str, role: str = "user") -> str:
    """Create a short-lived JWT access token (15 min)."""
    payload = {
        "sub": user_id,
        "role": role,
        "type": "access",
        "exp": datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRY_MINUTES),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")


def create_refresh_token() -> str:
    """Create a cryptographically random refresh token (raw, not JWT)."""
    return secrets.token_urlsafe(64)


def hash_refresh_token(token: str) -> str:
    """SHA-256 hash of refresh token for DB storage (never store raw)."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def decode_access_token(token: str) -> dict | None:
    """Decode and validate a JWT access token. Returns payload or None."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        if payload.get("type") != "access":
            return None
        return payload
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        return None


def get_refresh_token_expiry() -> str:
    """ISO 8601 expiry timestamp for a new refresh token."""
    return (datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRY_DAYS)).isoformat()


# ═══════════════════════════════════════════════════════════════
# FLASK MIDDLEWARE — auth_required decorator
# ═══════════════════════════════════════════════════════════════

def auth_required(f):
    """Decorator: require a valid JWT access token in Authorization header.
    Sets g.user_id and g.user_role on success."""
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return jsonify({"error": "Missing or invalid Authorization header"}), 401

        token = auth_header[7:]  # strip "Bearer "
        payload = decode_access_token(token)
        if not payload:
            return jsonify({"error": "Invalid or expired token"}), 401

        g.user_id = payload["sub"]
        g.user_role = payload.get("role", "user")
        return f(*args, **kwargs)

    return decorated


def get_client_ip() -> str:
    """Extract client IP, respecting X-Forwarded-For for reverse proxies."""
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.remote_addr or "unknown"

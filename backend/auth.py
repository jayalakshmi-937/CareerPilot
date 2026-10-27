"""
CareerPilot — Authentication & Password Security Utilities

Provides cryptographically secure PBKDF2-HMAC-SHA256 password hashing,
constant-time verification, and session token generation.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from typing import Optional


def hash_password(password: str) -> str:
    """Hash password using PBKDF2-HMAC-SHA256 with a unique random salt."""
    salt = secrets.token_hex(16)
    key = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        iterations=100_000,
    )
    return f"{salt}${key.hex()}"


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify plain password against PBKDF2 hash using constant-time comparison."""
    if not plain_password or not hashed_password:
        return False
    try:
        salt, key_hex = hashed_password.split("$", 1)
        test_key = hashlib.pbkdf2_hmac(
            "sha256",
            plain_password.encode("utf-8"),
            salt.encode("utf-8"),
            iterations=100_000,
        )
        return hmac.compare_digest(test_key.hex(), key_hex)
    except Exception:
        return False


def create_session_token() -> str:
    """Generate a secure cryptographic session token."""
    return secrets.token_urlsafe(32)

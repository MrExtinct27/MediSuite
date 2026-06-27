"""
AES-256 encryption for claim JSON files (HIPAA safeguard).

Key lifecycle:
  - Set CLAIM_ENCRYPTION_KEY in .env to a base64-url Fernet key (32 raw bytes).
  - If unset, an ephemeral key is generated per process and a warning is logged.
    Ephemeral keys mean claims written in one process cannot be read after restart.
  - Generate a persistent key with:  python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
"""

from __future__ import annotations

import json
import logging
import os

from cryptography.fernet import Fernet, InvalidToken

logger = logging.getLogger(__name__)

_ephemeral_key: bytes | None = None


def get_encryption_key() -> bytes:
    """Return the active Fernet key, generating an ephemeral one if CLAIM_ENCRYPTION_KEY is unset."""
    global _ephemeral_key
    raw = os.getenv("CLAIM_ENCRYPTION_KEY", "").strip()
    if raw:
        return raw.encode()
    if _ephemeral_key is None:
        _ephemeral_key = Fernet.generate_key()
        logger.warning(
            "No CLAIM_ENCRYPTION_KEY set in .env — using an ephemeral key. "
            "Encrypted claims will not be readable after server restart. "
            "Set CLAIM_ENCRYPTION_KEY to a persistent Fernet key."
        )
    return _ephemeral_key


def encrypt_claim(claim_data: dict) -> bytes:
    """Serialize claim_data to JSON and encrypt with AES-256 (Fernet)."""
    f = Fernet(get_encryption_key())
    return f.encrypt(json.dumps(claim_data, ensure_ascii=False).encode("utf-8"))


def decrypt_claim(encrypted_data: bytes) -> dict:
    """Decrypt and deserialize an encrypted claim file."""
    try:
        f = Fernet(get_encryption_key())
        return json.loads(f.decrypt(encrypted_data).decode("utf-8"))
    except InvalidToken as e:
        raise ValueError(
            "Claim decryption failed — key mismatch or corrupted file."
        ) from e

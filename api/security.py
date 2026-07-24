"""
Authentication primitives: password hashing (passlib/bcrypt), JWT issuing/decoding
(python-jose), and the get_current_user FastAPI dependency.

JWTs are signed with JWT_SECRET_KEY (from the environment — never hardcoded) using
HS256 and expire after ACCESS_TOKEN_EXPIRE_HOURS.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from db.database import get_db
from db.models import User

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 24
# bcrypt only hashes the first 72 bytes; enforce a max so hash/verify stay consistent.
MAX_PASSWORD_BYTES = 72

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# auto_error=False so a missing token yields None here and we raise our own 401.
_oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login", auto_error=False)

_CREDENTIALS_EXCEPTION = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Could not validate credentials.",
    headers={"WWW-Authenticate": "Bearer"},
)


def _secret_key() -> str:
    key = os.getenv("JWT_SECRET_KEY")
    if not key:
        raise RuntimeError("JWT_SECRET_KEY is not set; authentication cannot function.")
    return key


# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------


def hash_password(password: str) -> str:
    return _pwd_context.hash(password)


def verify_password(password: str, hashed_password: str) -> bool:
    return _pwd_context.verify(password, hashed_password)


# ---------------------------------------------------------------------------
# JWT
# ---------------------------------------------------------------------------


def create_access_token(subject: str) -> str:
    """Issue a signed JWT whose subject (sub) is the username."""
    expire = datetime.now(timezone.utc) + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    payload = {"sub": subject, "exp": expire}
    return jwt.encode(payload, _secret_key(), algorithm=ALGORITHM)


# ---------------------------------------------------------------------------
# Dependency
# ---------------------------------------------------------------------------


def get_current_user(
    token: Optional[str] = Depends(_oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    """
    Resolve the current user from the Bearer token. Raises 401 if the token is
    missing, malformed, expired, or points at a user that no longer exists.
    """
    if not token:
        raise _CREDENTIALS_EXCEPTION
    try:
        payload = jwt.decode(token, _secret_key(), algorithms=[ALGORITHM])
    except JWTError:
        # Covers invalid signature AND expired tokens (ExpiredSignatureError is a JWTError).
        raise _CREDENTIALS_EXCEPTION

    username = payload.get("sub")
    if not username:
        raise _CREDENTIALS_EXCEPTION

    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise _CREDENTIALS_EXCEPTION
    return user

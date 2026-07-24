"""
Auth routes: register, login, and current-user lookup.

POST /auth/register — create a user, return a JWT
POST /auth/login    — validate credentials, return a JWT
GET  /auth/me       — return the current user (requires a valid token)
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from api.security import (
    MAX_PASSWORD_BYTES,
    create_access_token,
    get_current_user,
    hash_password,
    verify_password,
)
from db.database import get_db
from db.models import User

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])

MIN_PASSWORD_LENGTH = 8


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class RegisterRequest(BaseModel):
    username: str
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str


class UserResponse(BaseModel):
    id: int
    username: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def register(body: RegisterRequest, db: Session = Depends(get_db)):
    """Create a new user (username must be unique, password min length enforced), return a token."""
    username = (body.username or "").strip()
    if not username:
        raise HTTPException(status_code=422, detail="Username is required.")
    if len(username) > 64:
        raise HTTPException(status_code=422, detail="Username must be at most 64 characters.")
    if len(body.password) < MIN_PASSWORD_LENGTH:
        raise HTTPException(
            status_code=422,
            detail=f"Password must be at least {MIN_PASSWORD_LENGTH} characters.",
        )
    if len(body.password.encode("utf-8")) > MAX_PASSWORD_BYTES:
        raise HTTPException(
            status_code=422,
            detail=f"Password must be at most {MAX_PASSWORD_BYTES} bytes.",
        )

    if db.query(User).filter(User.username == username).first() is not None:
        raise HTTPException(status_code=409, detail="Username is already taken.")

    user = User(username=username, hashed_password=hash_password(body.password))
    db.add(user)
    db.commit()
    db.refresh(user)

    logger.info("Registered new user: %s (id=%s)", user.username, user.id)
    token = create_access_token(user.username)
    return TokenResponse(access_token=token, username=user.username)


@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest, db: Session = Depends(get_db)):
    """Validate credentials and return a JWT."""
    username = (body.username or "").strip()
    user = db.query(User).filter(User.username == username).first()
    # Same error whether the user is unknown or the password is wrong (no user enumeration).
    if user is None or not verify_password(body.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    token = create_access_token(user.username)
    return TokenResponse(access_token=token, username=user.username)


@router.get("/me", response_model=UserResponse)
def me(current_user: User = Depends(get_current_user)):
    """Return the authenticated user."""
    return UserResponse(id=current_user.id, username=current_user.username)

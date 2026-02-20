"""SQLAlchemy engine, session factory, and declarative Base."""

from __future__ import annotations

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./medisuite.db")

# SQLite needs check_same_thread=False for use across threads (FastAPI workers)
_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=_connect_args)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    """FastAPI dependency: yield a DB session and close it after the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all tables. Called once at application startup."""
    from db import models  # noqa: F401 — ensure models are registered on Base
    Base.metadata.create_all(bind=engine)
if __name__ == "__main__":
    print("Creating database tables...")
    init_db()
    print("✅ Database tables created successfully")
    print(f"   Database: {DATABASE_URL}")
    
    from sqlalchemy import inspect
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    for table in tables:
        print(f"   Table: {table}")
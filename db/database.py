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


def _migrate_add_missing_columns() -> None:
    """
    Lightweight, idempotent migration for SQLite: add summary columns that may be
    missing on databases created before they were introduced. create_all() only
    creates missing tables, not missing columns, so we ALTER existing tables here.
    """
    from sqlalchemy import inspect, text

    # Column name -> SQL type used in ALTER TABLE ADD COLUMN
    expected_columns = {
        "claims": {
            "service_date": "VARCHAR(32)",
            "avg_confidence": "FLOAT",
            "processing_stage": "VARCHAR(32)",
        },
    }

    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())

    with engine.begin() as conn:
        for table, columns in expected_columns.items():
            if table not in existing_tables:
                continue
            existing_cols = {col["name"] for col in inspector.get_columns(table)}
            for col_name, col_type in columns.items():
                if col_name not in existing_cols:
                    conn.execute(text(f'ALTER TABLE {table} ADD COLUMN {col_name} {col_type}'))


def init_db() -> None:
    """Create all tables. Called once at application startup."""
    from db import models  # noqa: F401 — ensure models are registered on Base
    Base.metadata.create_all(bind=engine)
    _migrate_add_missing_columns()
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
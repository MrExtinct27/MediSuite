"""SQLAlchemy ORM models: Claim and AuditLog."""

from __future__ import annotations
from typing import Optional

from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from db.database import Base


class Claim(Base):
    __tablename__ = "claims"

    claim_id: Mapped[str] = mapped_column(String(64), primary_key=True, index=True)
    patient_name: Mapped[Optional[str]] = mapped_column(String(256))
    patient_dob: Mapped[Optional[str]] = mapped_column(String(32))
    patient_insurance_id: Mapped[Optional[str]] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(32), default="pending")  # pending | generated | failed
    form_path: Mapped[Optional[str]] = mapped_column(Text)
    validation_passed: Mapped[Optional[bool]] = mapped_column(Boolean)
    icd10_codes: Mapped[Optional[list]] = mapped_column(JSON)   # selected_codes list
    cpt4_codes: Mapped[Optional[list]] = mapped_column(JSON)    # selected_codes list
    validation_errors: Mapped[Optional[list]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )


class AuditLog(Base):
    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    claim_id: Mapped[Optional[str]] = mapped_column(String(64), index=True)
    event: Mapped[str] = mapped_column(String(128))
    detail: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

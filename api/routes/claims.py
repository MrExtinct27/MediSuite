"""
Claims API routes.
POST /claims/process  — submit a claim document for end-to-end processing
GET  /claims/{claim_id} — retrieve a previously generated claim record
"""

from __future__ import annotations

import logging
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from sqlalchemy.orm import Session

from db.database import get_db
from db.models import Claim
from orchestrator.graph import graph

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/claims", tags=["claims"])


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class ProcessClaimResponse(BaseModel):
    claim_id: str
    processing_status: str
    validation_passed: Optional[bool]
    validation_errors: list[dict]
    claim_form_path: Optional[str]


class ClaimRecord(BaseModel):
    claim_id: str
    patient_name: Optional[str]
    patient_dob: Optional[str]
    patient_insurance_id: Optional[str]
    status: str
    form_path: Optional[str]
    validation_passed: Optional[bool]
    icd10_codes: Optional[list]
    cpt4_codes: Optional[list]
    validation_errors: Optional[list]

    class Config:
        from_attributes = True


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/process", response_model=ProcessClaimResponse)
async def process_claim(
    file: Optional[UploadFile] = File(None),
    raw_text: Optional[str] = Form(None),
    claim_id: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    """
    Submit a clinical document for automated ICD-10 / CPT-4 coding and claim generation.
    Provide either a document file (PDF / DOCX / TXT) or raw_text.
    """
    if file is None and not raw_text:
        raise HTTPException(status_code=422, detail="Provide either a file upload or raw_text.")

    generated_claim_id = claim_id or str(uuid.uuid4())

    # Write uploaded file to a temp path so document_agent can read it
    file_path: Optional[str] = None
    if file is not None:
        import tempfile, os, shutil
        suffix = os.path.splitext(file.filename or "")[-1] or ".bin"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            file_path = tmp.name

    initial_state = {
        "claim_id": generated_claim_id,
        "raw_document_text": raw_text or "",
        "file_path": file_path,
        "revalidation_count": 0,
    }

    try:
        final_state = graph.invoke(initial_state)
    except Exception as e:
        logger.exception("Graph execution failed for claim %s: %s", generated_claim_id, e)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")
    finally:
        if file_path:
            try:
                import os
                os.unlink(file_path)
            except OSError:
                pass

    return ProcessClaimResponse(
        claim_id=final_state.get("claim_id", generated_claim_id),
        processing_status=final_state.get("processing_status", "unknown"),
        validation_passed=final_state.get("validation_passed"),
        validation_errors=final_state.get("validation_errors") or [],
        claim_form_path=final_state.get("claim_form_path"),
    )


@router.get("/{claim_id}", response_model=ClaimRecord)
def get_claim(claim_id: str, db: Session = Depends(get_db)):
    """Retrieve a previously processed claim by ID."""
    claim = db.get(Claim, claim_id)
    if claim is None:
        raise HTTPException(status_code=404, detail=f"Claim {claim_id!r} not found.")
    return claim

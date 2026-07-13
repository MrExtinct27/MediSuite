"""
Claims API routes.
POST /claims/process  — submit a claim document for end-to-end processing
GET  /claims/{claim_id} — retrieve a previously generated claim record
"""

from __future__ import annotations

import logging
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File, Form
from openai import AuthenticationError, OpenAIError, RateLimitError
from pydantic import BaseModel
from sqlalchemy.orm import Session
from pathlib import Path
import json

from agents.coding_agent import _retrieval_cache
from api.limiter import limiter
from datetime import datetime

from db.database import get_db
from db.models import Claim, ReviewQueue
from orchestrator.graph import graph
from orchestrator.state import resolve_llm_model
from utils.encryption import decrypt_claim

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/claims", tags=["claims"])
cache_router = APIRouter(prefix="/cache", tags=["cache"])


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


class ClaimSummary(BaseModel):
    """Lightweight summary used by the dashboard claims list."""
    claim_id: str
    patient_name: Optional[str]
    service_date: Optional[str]
    validation_status: str  # "passed" | "failed" | "processing"
    avg_confidence: float   # 0.0 – 1.0, mean confidence across all codes


class ClaimStatusResponse(BaseModel):
    """Lightweight live-progress payload polled ~once/second by the Submit Claim page."""
    claim_id: str
    processing_stage: str  # document | coding | validation | claim | complete
    status: str            # processing | generated | failed | pending
    complete: bool


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/process", response_model=ProcessClaimResponse)
@limiter.limit("10/minute")
async def process_claim(
    request: Request,
    file: Optional[UploadFile] = File(None),
    raw_text: Optional[str] = Form(None),
    claim_id: Optional[str] = Form(None),
    patient_name: Optional[str] = Form(None),
    patient_dob: Optional[str] = Form(None),
    patient_insurance_id: Optional[str] = Form(None),
    patient_sex: Optional[str] = Form(None),
    patient_address: Optional[str] = Form(None),
    insurance_provider: Optional[str] = Form(None),
    llm_model: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    """
    Submit a clinical document for automated ICD-10 / CPT-4 coding and claim generation.
    Provide either a document file (PDF / DOCX / TXT) or raw_text.

    The patient_* / insurance_provider form fields are OPTIONAL fallback values:
    if the clinical note contains the info, the note's extracted value always wins.
    A form value is only used when the note did not provide that field.
    """
    if file is None and not raw_text:
        raise HTTPException(status_code=422, detail="Provide either a file upload or raw_text.")

    generated_claim_id = claim_id or str(uuid.uuid4())

    # Per-request LLM model selection. Falls back to the default when the value
    # is missing or not one of the three allowed models — never rejected.
    selected_llm_model = resolve_llm_model(llm_model)

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
        "llm_model": selected_llm_model,
        # Fallback-only patient values; note-extracted values take precedence downstream.
        "form_overrides": {
            "patient_name": patient_name,
            "patient_dob": patient_dob,
            "patient_insurance_id": patient_insurance_id,
            "patient_sex": patient_sex,
            "patient_address": patient_address,
            "insurance_provider": insurance_provider,
        },
    }

    # Create the claim row up front (status=processing, stage=document) so the
    # frontend can immediately poll GET /claims/{id}/status for live progress.
    # The pipeline agents update processing_stage as they run; claim_agent flips
    # it to "complete" at the end.
    try:
        existing = db.get(Claim, generated_claim_id)
        if existing is None:
            db.add(
                Claim(
                    claim_id=generated_claim_id,
                    patient_name=patient_name or None,
                    patient_dob=patient_dob or None,
                    patient_insurance_id=patient_insurance_id or None,
                    status="processing",
                    processing_stage="document",
                )
            )
        else:
            existing.status = "processing"
            existing.processing_stage = "document"
        db.commit()
    except Exception as e:
        db.rollback()
        logger.exception("Failed to create processing row for claim %s: %s", generated_claim_id, e)

    try:
        final_state = await graph.ainvoke(initial_state)
    except RateLimitError as e:
        logger.exception("LLM rate limit / quota exceeded for claim %s: %s", generated_claim_id, e)
        raise HTTPException(
            status_code=429,
            detail=(
                "LLM rate limit or quota exceeded — the claim was not processed. "
                "Check the model's usage/billing and try again."
            ),
        )
    except AuthenticationError as e:
        logger.exception("LLM authentication failed for claim %s: %s", generated_claim_id, e)
        raise HTTPException(
            status_code=502,
            detail=(
                "LLM authentication failed — the API key is missing or invalid. "
                "The claim was not processed."
            ),
        )
    except OpenAIError as e:
        logger.exception("LLM API error for claim %s: %s", generated_claim_id, e)
        raise HTTPException(
            status_code=502,
            detail=f"LLM API error — the claim was not processed: {e}",
        )
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


def _avg_confidence_for(claim: Claim) -> float:
    """
    Resolve a claim's average confidence. Prefer the stored summary column; fall
    back to recomputing from the stored ICD-10 / CPT-4 code JSON for legacy rows
    that predate the avg_confidence column.
    """
    if claim.avg_confidence is not None:
        return float(claim.avg_confidence)

    confidences = [
        c.get("confidence")
        for c in (claim.icd10_codes or []) + (claim.cpt4_codes or [])
        if isinstance(c, dict) and c.get("confidence") is not None
    ]
    return sum(confidences) / len(confidences) if confidences else 0.0


def _summary_validation_status(claim: Claim) -> str:
    """
    Map a claim to a dashboard status: "processing" while the pipeline is still
    running (so it isn't mislabeled "failed" just because validation_passed is
    not set yet), otherwise "passed" / "failed" from the validation result.
    """
    if claim.status == "processing" and (claim.processing_stage or "") != "complete":
        return "processing"
    return "passed" if claim.validation_passed else "failed"


@router.get("/", response_model=list[ClaimSummary])
def list_claims(db: Session = Depends(get_db)):
    """
    Return a summary of all claims (newest first) for the dashboard, read entirely
    from the SQLite records — never by parsing encrypted .enc files.
    """
    try:
        claims = db.query(Claim).order_by(Claim.created_at.desc()).all()
        return [
            ClaimSummary(
                claim_id=c.claim_id,
                patient_name=c.patient_name,
                service_date=c.service_date,
                validation_status=_summary_validation_status(c),
                avg_confidence=_avg_confidence_for(c),
            )
            for c in claims
        ]
    except Exception as exc:
        logger.exception("Failed to list claims: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to list claims.") from exc


@router.get("/{claim_id}/result")
def get_claim_result(claim_id: str):
    """
    Return the full claim object. Tries encrypted .enc first, falls back to
    legacy .json for claims written before encryption was enabled.
    """
    claims_dir = Path(__file__).resolve().parents[2] / "claims"
    claims_dir.mkdir(exist_ok=True)

    enc_path = claims_dir / f"{claim_id}.enc"
    json_path = claims_dir / f"{claim_id}.json"

    if enc_path.exists():
        try:
            return decrypt_claim(enc_path.read_bytes())
        except ValueError as exc:
            logger.exception("Decryption failed for %s: %s", enc_path, exc)
            raise HTTPException(status_code=500, detail="Claim decryption failed — key mismatch or corrupted file.") from exc
        except Exception as exc:
            logger.exception("Failed to read encrypted claim %s: %s", enc_path, exc)
            raise HTTPException(status_code=500, detail="Failed to read claim file.") from exc

    if json_path.exists():
        try:
            return json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logger.exception("Invalid JSON in claim file %s: %s", json_path, exc)
            raise HTTPException(status_code=500, detail="Stored claim JSON is invalid.") from exc
        except Exception as exc:
            logger.exception("Failed to read claim file %s: %s", json_path, exc)
            raise HTTPException(status_code=500, detail="Failed to read claim JSON.") from exc

    raise HTTPException(status_code=404, detail=f"Claim file for {claim_id!r} not found.")


@router.get("/{claim_id}/status", response_model=ClaimStatusResponse)
def get_claim_status(claim_id: str, db: Session = Depends(get_db)):
    """
    Fast live-progress check for the Submit Claim page (polled ~1x/second).

    Reads only the Claim row's stage/status columns — never decrypts the claim
    file — so it stays cheap enough to poll continuously while the pipeline runs.
    """
    claim = db.get(Claim, claim_id)
    if claim is None:
        raise HTTPException(status_code=404, detail=f"Claim {claim_id!r} not found.")

    # Legacy rows (created before processing_stage existed) have no stage: infer
    # "complete" from a generated status, otherwise assume it is still at document.
    stage = claim.processing_stage or ("complete" if claim.status == "generated" else "document")
    complete = stage == "complete" or claim.status == "generated"

    return ClaimStatusResponse(
        claim_id=claim.claim_id,
        processing_stage=stage,
        status=claim.status,
        complete=complete,
    )


@router.get("/review-queue")
def get_review_queue(db: Session = Depends(get_db)):
    """Return all claims currently pending human review, newest first."""
    items = (
        db.query(ReviewQueue)
        .filter(ReviewQueue.status == "pending")
        .order_by(ReviewQueue.created_at.desc())
        .all()
    )
    return {
        "review_queue": [
            {
                "id": item.id,
                "claim_id": item.claim_id,
                "reason": item.reason,
                "low_confidence_codes": item.low_confidence_codes,
                "status": item.status,
                "created_at": item.created_at.isoformat() if item.created_at else None,
            }
            for item in items
        ],
        "count": len(items),
    }


@router.post("/{claim_id}/approve")
def approve_claim(
    claim_id: str,
    reviewer_notes: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    """Approve a claim that was flagged for human review."""
    item = db.query(ReviewQueue).filter(ReviewQueue.claim_id == claim_id).first()
    if item is None:
        raise HTTPException(status_code=404, detail=f"Claim {claim_id!r} not found in review queue.")
    item.status = "approved"
    item.reviewer_decision = "approved"
    item.reviewer_notes = reviewer_notes
    item.reviewed_at = datetime.utcnow()
    db.commit()
    return {"status": "approved", "claim_id": claim_id}


@router.post("/{claim_id}/reject")
def reject_claim(
    claim_id: str,
    reviewer_notes: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    """Reject a claim that was flagged for human review."""
    item = db.query(ReviewQueue).filter(ReviewQueue.claim_id == claim_id).first()
    if item is None:
        raise HTTPException(status_code=404, detail=f"Claim {claim_id!r} not found in review queue.")
    item.status = "rejected"
    item.reviewer_decision = "rejected"
    item.reviewer_notes = reviewer_notes
    item.reviewed_at = datetime.utcnow()
    db.commit()
    return {"status": "rejected", "claim_id": claim_id}


@router.get("/{claim_id}", response_model=ClaimRecord)
def get_claim(claim_id: str, db: Session = Depends(get_db)):
    """Retrieve a previously processed claim by ID."""
    claim = db.get(Claim, claim_id)
    if claim is None:
        raise HTTPException(status_code=404, detail=f"Claim {claim_id!r} not found.")
    return claim


# ---------------------------------------------------------------------------
# Cache endpoints
# ---------------------------------------------------------------------------


@cache_router.get("/stats")
def cache_stats() -> dict:
    """Return the number of cached ChromaDB queries and their cache keys."""
    return {
        "cache_size": len(_retrieval_cache),
        "cached_queries": list(_retrieval_cache.keys()),
    }


@cache_router.post("/clear")
def cache_clear() -> dict:
    """Evict all cached ChromaDB retrieval results."""
    _retrieval_cache.clear()
    return {"cleared": True}

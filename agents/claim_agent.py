"""
Claim Agent — LangGraph node.
Generates a CMS-1500 structured JSON summary, persists it to disk,
and records the result in the database with an audit log entry.

No embedding calls and no LLM calls are made here.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from orchestrator.state import ClaimState
from utils.encryption import encrypt_claim

logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.80"))

# ---------------------------------------------------------------------------
# PHI masking — used for logging only; never for agent logic
# ---------------------------------------------------------------------------

_PHI_FIELDS = [
    "patient_name", "patient_dob", "patient_insurance_id",
    "name", "dob", "insurance_id",
]


def mask_phi_for_logging(state: dict) -> dict:
    """Return a shallow copy of state with PHI fields and raw document replaced."""
    masked = state.copy()
    for field in _PHI_FIELDS:
        if masked.get(field):
            masked[field] = "[PHI-REDACTED]"
    if "raw_document_text" in masked:
        masked["raw_document_text"] = "[DOCUMENT-REDACTED]"
    return masked

_CLAIMS_DIR = Path(os.getenv("CLAIMS_DIR", "./claims"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _selected_icd10(state: ClaimState) -> list[dict]:
    return (state.get("icd10_selected") or {}).get("selected_codes", [])


def _selected_cpt4(state: ClaimState) -> list[dict]:
    return (state.get("cpt4_selected") or {}).get("selected_codes", [])


def _coalesce(*values: Any) -> Any:
    """Return the first value that is not None and not an empty/whitespace string."""
    for value in values:
        if value is not None and str(value).strip():
            return value
    return None


def _resolve_patient_fields(state: ClaimState) -> dict[str, Any]:
    """
    Resolve patient fields using note-first precedence with form-value fallback.

    For each field: the value extracted from the note by the Document Agent wins;
    only if that is None/empty do we fall back to the corresponding form_overrides
    value; if both are missing the result is None.

    When no form values are supplied, form_overrides is empty and each field
    resolves to exactly the note-extracted value (identical to prior behavior).
    """
    entities: dict = state.get("extracted_entities") or {}
    overrides: dict = state.get("form_overrides") or {}

    return {
        "name": _coalesce(state.get("patient_name"), overrides.get("patient_name")),
        "dob": _coalesce(state.get("patient_dob"), overrides.get("patient_dob")),
        "insurance_id": _coalesce(
            state.get("patient_insurance_id"), overrides.get("patient_insurance_id")
        ),
        # New fields: Document Agent may surface these in extracted_entities; otherwise
        # fall back to the submitted form value.
        "sex": _coalesce(entities.get("patient_sex"), overrides.get("patient_sex")),
        "address": _coalesce(entities.get("patient_address"), overrides.get("patient_address")),
        "insurance_provider": _coalesce(
            entities.get("insurance_provider"), overrides.get("insurance_provider")
        ),
    }


def _build_claim_data(state: ClaimState) -> dict[str, Any]:
    """Assemble the CMS-1500 JSON structure from ClaimState."""
    entities: dict = state.get("extracted_entities") or {}
    dates: dict = entities.get("dates") or {}

    icd10_codes = _selected_icd10(state)
    cpt4_codes = _selected_cpt4(state)

    patient = _resolve_patient_fields(state)

    return {
        "claim_id": state.get("claim_id"),
        "form_type": "CMS-1500",
        "generated_at": datetime.now().isoformat(),
        "patient": {
            "name": patient["name"],
            "dob": patient["dob"],
            "insurance_id": patient["insurance_id"],
            "sex": patient["sex"],
            "address": patient["address"],
            "insurance_provider": patient["insurance_provider"],
        },
        "service_date": dates.get("service_date"),
        "provider_name": entities.get("provider_name"),
        "facility_name": entities.get("facility_name"),
        "diagnosis_codes": [
            {
                "code": c.get("code"),
                "description": c.get("disease") or c.get("description"),
                "confidence": c.get("confidence"),
            }
            for c in icd10_codes
        ],
        "procedure_codes": [
            {
                "code": c.get("code"),
                "description": c.get("description"),
                "confidence": c.get("confidence"),
            }
            for c in cpt4_codes
        ],
        "validation_status": "passed" if state.get("validation_passed") else "failed",
        "validation_errors": state.get("validation_errors") or [],
        "explainability": {
            "icd10_reasoning": [
                c.get("reasoning") for c in icd10_codes if c.get("reasoning")
            ],
            "cpt4_reasoning": [
                c.get("reasoning") for c in cpt4_codes if c.get("reasoning")
            ],
            "citations": [
                c.get("citation") for c in icd10_codes if c.get("citation")
            ],
            "icd10_reasoning_chain": (state.get("icd10_selected") or {}).get("reasoning_chain", ""),
            "cpt4_reasoning_chain": (state.get("cpt4_selected") or {}).get("reasoning_chain", ""),
        },
    }


def _save_claim_json(claim_id: str, claim_data: dict) -> str:
    """Encrypt and write claim data to ./claims/{claim_id}.enc, return the path."""
    _CLAIMS_DIR.mkdir(parents=True, exist_ok=True)
    path = _CLAIMS_DIR / f"{claim_id}.enc"
    path.write_bytes(encrypt_claim(claim_data))
    return str(path)


def _persist_to_db(
    state: ClaimState,
    claim_data: dict,
    form_path: str,
    low_confidence_codes: list[dict] | None = None,
) -> None:
    """Upsert Claim row, append AuditLog entry, and optionally enqueue for human review."""
    try:
        from db.database import SessionLocal
        from db.models import AuditLog, Claim, ReviewQueue

        with SessionLocal() as session:
            claim = session.get(Claim, claim_data["claim_id"])
            if claim is None:
                claim = Claim(claim_id=claim_data["claim_id"])
                session.add(claim)

            patient = _resolve_patient_fields(state)

            # Per-claim average confidence across all selected codes (summary stat)
            all_codes = claim_data["diagnosis_codes"] + claim_data["procedure_codes"]
            confidences = [
                c.get("confidence") for c in all_codes if c.get("confidence") is not None
            ]
            avg_confidence = sum(confidences) / len(confidences) if confidences else None

            claim.patient_name = patient["name"]
            claim.patient_dob = patient["dob"]
            claim.patient_insurance_id = patient["insurance_id"]
            claim.service_date = claim_data.get("service_date")
            claim.status = "generated"
            claim.form_path = form_path
            claim.validation_passed = state.get("validation_passed")
            claim.avg_confidence = avg_confidence
            claim.icd10_codes = claim_data["diagnosis_codes"]
            claim.cpt4_codes = claim_data["procedure_codes"]
            claim.validation_errors = state.get("validation_errors")

            audit = AuditLog(
                claim_id=claim_data["claim_id"],
                event="claim_generated",
                detail=f"form_path={form_path}; validation={'passed' if state.get('validation_passed') else 'failed'}",
            )
            session.add(audit)

            if low_confidence_codes:
                review_entry = ReviewQueue(
                    id=str(uuid.uuid4()),
                    claim_id=claim_data["claim_id"],
                    reason="Low confidence codes detected",
                    low_confidence_codes=low_confidence_codes,
                    status="pending",
                )
                session.add(review_entry)
                logger.info(
                    "claim_agent: queued claim %s for human review (%d low-confidence code(s))",
                    claim_data["claim_id"],
                    len(low_confidence_codes),
                )

            session.commit()
    except Exception as e:
        logger.exception("DB write failed for claim %s: %s", claim_data.get("claim_id"), e)


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------


def claim_agent(state: ClaimState) -> ClaimState:
    """
    LangGraph node: build CMS-1500 JSON, persist to disk and DB, log to AuditLog.

    Reads:  full ClaimState (icd10_selected, cpt4_selected, validation_*, extracted_entities, patient_*)
    Writes: state['claim_form_path']    — path to generated JSON file
            state['processing_status'] — 'claim_generated'
    """
    try:
        claim_id = state.get("claim_id") or str(uuid.uuid4())

        claim_data = _build_claim_data({**state, "claim_id": claim_id})

        # Identify any codes below the confidence threshold for human review
        all_codes = claim_data["diagnosis_codes"] + claim_data["procedure_codes"]
        low_confidence_codes = [
            c for c in all_codes if (c.get("confidence") or 0) < CONFIDENCE_THRESHOLD
        ]
        requires_human_review = bool(low_confidence_codes)

        form_path = _save_claim_json(claim_id, claim_data)
        _persist_to_db(
            state,
            claim_data,
            form_path,
            low_confidence_codes if requires_human_review else None,
        )

        masked = mask_phi_for_logging(state)
        logger.info(
            "claim_agent: generated %s (patient=[PHI-REDACTED] insurance=[PHI-REDACTED])",
            form_path,
        )
        logger.debug("claim_agent state (masked): %s", masked)

        # If validation did not pass (retries exhausted), flag the status so
        # consumers know the claim was force-generated with warnings.
        validation_passed = state.get("validation_passed")
        status = "claim_generated" if validation_passed else "claim_generated_with_warnings"

        return {
            **state,
            "claim_id": claim_id,
            "claim_form_path": form_path,
            "processing_status": status,
            "requires_human_review": requires_human_review,
            "low_confidence_codes": low_confidence_codes,
        }

    except Exception as e:
        logger.exception("claim_agent failed: %s", e)
        return {
            **state,
            "claim_form_path": None,
            "processing_status": "claim_failed",
            "requires_human_review": False,
            "low_confidence_codes": [],
        }

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

logger = logging.getLogger(__name__)

_CLAIMS_DIR = Path(os.getenv("CLAIMS_DIR", "./claims"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _selected_icd10(state: ClaimState) -> list[dict]:
    return (state.get("icd10_selected") or {}).get("selected_codes", [])


def _selected_cpt4(state: ClaimState) -> list[dict]:
    return (state.get("cpt4_selected") or {}).get("selected_codes", [])


def _build_claim_data(state: ClaimState) -> dict[str, Any]:
    """Assemble the CMS-1500 JSON structure from ClaimState."""
    entities: dict = state.get("extracted_entities") or {}
    dates: dict = entities.get("dates") or {}

    icd10_codes = _selected_icd10(state)
    cpt4_codes = _selected_cpt4(state)

    return {
        "claim_id": state.get("claim_id"),
        "form_type": "CMS-1500",
        "generated_at": datetime.now().isoformat(),
        "patient": {
            "name": state.get("patient_name"),
            "dob": state.get("patient_dob"),
            "insurance_id": state.get("patient_insurance_id"),
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
    """Write claim JSON to ./claims/{claim_id}.json and return the path."""
    _CLAIMS_DIR.mkdir(parents=True, exist_ok=True)
    path = _CLAIMS_DIR / f"{claim_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(claim_data, f, indent=2, ensure_ascii=False)
    return str(path)


def _persist_to_db(state: ClaimState, claim_data: dict, form_path: str) -> None:
    """Upsert Claim row and append an AuditLog entry."""
    try:
        from db.database import SessionLocal
        from db.models import AuditLog, Claim

        with SessionLocal() as session:
            claim = session.get(Claim, claim_data["claim_id"])
            if claim is None:
                claim = Claim(claim_id=claim_data["claim_id"])
                session.add(claim)

            claim.patient_name = state.get("patient_name")
            claim.patient_dob = state.get("patient_dob")
            claim.patient_insurance_id = state.get("patient_insurance_id")
            claim.status = "generated"
            claim.form_path = form_path
            claim.validation_passed = state.get("validation_passed")
            claim.icd10_codes = claim_data["diagnosis_codes"]
            claim.cpt4_codes = claim_data["procedure_codes"]
            claim.validation_errors = state.get("validation_errors")

            audit = AuditLog(
                claim_id=claim_data["claim_id"],
                event="claim_generated",
                detail=f"form_path={form_path}; validation={'passed' if state.get('validation_passed') else 'failed'}",
            )
            session.add(audit)
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
        form_path = _save_claim_json(claim_id, claim_data)
        _persist_to_db(state, claim_data, form_path)

        logger.info("claim_agent: generated %s", form_path)

        return {
            **state,
            "claim_id": claim_id,
            "claim_form_path": form_path,
            "processing_status": "claim_generated",
        }

    except Exception as e:
        logger.exception("claim_agent failed: %s", e)
        return {
            **state,
            "claim_form_path": None,
            "processing_status": "claim_failed",
        }

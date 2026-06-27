"""
Validation Agent — LangGraph node.
3-level validation pipeline run in sequence; all errors collected into state['validation_errors'].

Level 1 — Rule-based validation   (deterministic, fast — no API call)
Level 2 — LLM clinical logic      (GPT-4o via langchain-openai — one API call)
Level 3 — Confidence threshold    (deterministic — no API call)
"""

from __future__ import annotations

import json
import logging
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langsmith import traceable

from orchestrator.state import ClaimState

logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.80"))

# Path to the ICD-10 knowledge base used to confirm whether a code is billable.
_ICD10_JSON_PATH = Path(__file__).resolve().parents[1] / "knowledge_base" / "ICD10.json"


def _normalize_icd10(code: str) -> str:
    """Normalize an ICD-10 code for comparison: strip, remove dots, uppercase."""
    return (code or "").strip().replace(".", "").upper()


@lru_cache(maxsize=1)
def _valid_icd10_codes() -> frozenset[str]:
    """
    Load short (3-4 character) ICD-10 codes from ICD10.json into a normalized set.

    The specificity check only needs to confirm whether SHORT codes are billable
    (e.g. I10, J13, R51), so we discard the long codes that make up the bulk of the
    71K-entry dataset. Keeping only 3-4 char codes shrinks memory and speeds up the
    load significantly while preserving the billable-short-code fix.
    Loaded once and cached for the lifetime of the process.
    """
    try:
        with open(_ICD10_JSON_PATH, encoding="utf-8") as fh:
            data: list[dict] = json.load(fh)
        codes = {
            norm
            for rec in data
            if 3 <= len(norm := _normalize_icd10(rec.get("code", ""))) <= 4
        }
        logger.info("Loaded %d short ICD-10 codes for billability checks.", len(codes))
        return frozenset(codes)
    except Exception as exc:
        logger.exception("Failed to load ICD-10 codes from %s: %s", _ICD10_JSON_PATH, exc)
        return frozenset()

# ---------------------------------------------------------------------------
# PHI masking — used for LangSmith trace inputs only; never for agent logic
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


def _mask_state_trace_inputs(inputs: dict) -> dict:
    """process_inputs hook for @traceable — redacts PHI from the state dict."""
    if "state" in inputs:
        return {**inputs, "state": mask_phi_for_logging(inputs["state"])}
    return inputs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _error(field: str, message: str, severity: str) -> dict[str, str]:
    return {"field": field, "message": message, "severity": severity}


def _selected_icd10(state: ClaimState) -> list[dict]:
    return (state.get("icd10_selected") or {}).get("selected_codes", [])


def _selected_cpt4(state: ClaimState) -> list[dict]:
    return (state.get("cpt4_selected") or {}).get("selected_codes", [])


# ---------------------------------------------------------------------------
# Level 1 — Rule-based validation (deterministic, no API call)
# ---------------------------------------------------------------------------


def check_required_fields(state: ClaimState) -> list[dict]:
    """Verify that all required claim fields are present and non-empty."""
    errors: list[dict] = []

    required_str_fields = [
        ("patient_name",         "Missing required field: patient_name"),
        ("patient_dob",          "Missing required field: patient_dob"),
        ("patient_insurance_id", "Missing patient insurance ID"),
    ]
    for field, message in required_str_fields:
        value = state.get(field)
        if not value or not str(value).strip():
            errors.append(_error(field, message, "critical"))

    if not _selected_icd10(state):
        errors.append(_error("icd10_selected", "No ICD-10 codes assigned", "critical"))

    if not _selected_cpt4(state):
        errors.append(_error("cpt4_selected", "No CPT-4 codes assigned", "critical"))

    return errors


def check_icd10_specificity(state: ClaimState) -> list[dict]:
    """
    Flag ICD-10 codes that lack billable specificity.

    Many 3-character ICD-10 codes (e.g. I10, J13, R51) are themselves valid,
    billable codes. We therefore only flag a code as non-billable when it does
    NOT exist as an exact entry in the ICD10.json knowledge base. Any code that
    matches the dataset exactly is treated as billable and is never flagged.
    """
    errors: list[dict] = []
    valid_codes = _valid_icd10_codes()

    for entry in _selected_icd10(state):
        code = (entry.get("code") or "").strip()
        if not code:
            continue

        normalized = _normalize_icd10(code)

        # Exact match in the dataset → real billable code, do not flag.
        if normalized in valid_codes:
            continue

        # Only short codes that are NOT in the dataset are suspected category-level/incomplete.
        if len(normalized) <= 3:
            errors.append(
                _error(
                    f"icd10:{code}",
                    f"Code {code} may lack specificity (category-level code, not billable)",
                    "warning",
                )
            )
    return errors


def check_code_count_limits(state: ClaimState) -> list[dict]:
    """
    CMS-1500 form limits: max 12 diagnosis codes, max 6 procedure codes.
    """
    errors: list[dict] = []

    icd10_codes = _selected_icd10(state)
    if len(icd10_codes) > 12:
        errors.append(
            _error(
                "icd10_selected",
                f"CMS-1500 allows max 12 diagnosis codes; {len(icd10_codes)} assigned",
                "critical",
            )
        )

    cpt4_codes = _selected_cpt4(state)
    if len(cpt4_codes) > 6:
        errors.append(
            _error(
                "cpt4_selected",
                f"CMS-1500 standard form allows max 6 procedure codes; {len(cpt4_codes)} assigned",
                "warning",
            )
        )

    return errors


def check_service_date_present(state: ClaimState) -> list[dict]:
    """Service date is required on every claim."""
    errors: list[dict] = []
    dates = (state.get("extracted_entities") or {}).get("dates") or {}
    service_date = dates.get("service_date")
    if not service_date:
        errors.append(
            _error("dates.service_date", "Missing service date", "critical")
        )
    return errors


def _run_level1(state: ClaimState) -> list[dict]:
    errors: list[dict] = []
    errors.extend(check_required_fields(state))
    errors.extend(check_icd10_specificity(state))
    errors.extend(check_code_count_limits(state))
    errors.extend(check_service_date_present(state))
    return errors


# ---------------------------------------------------------------------------
# Level 2 — LLM clinical logic validation (GPT-4o, one API call)
# ---------------------------------------------------------------------------

_L2_SYSTEM_PROMPT = """\
You are a medical claims auditor and clinical coding validator. \
Review the assigned ICD-10 and CPT-4 codes for clinical logic and consistency.

Check for these issues:
1. Diagnosis-procedure consistency: Are the procedures appropriate for the diagnoses?
2. Code pairing conflicts: Are any codes mutually exclusive per CMS guidelines?
3. Missing secondary codes: Should any complicating conditions be separately coded?
4. Suspicious combinations: Any code combinations that seem clinically implausible?

Severity rules — use these strictly:
- CRITICAL: Only when a code is completely unrelated to the documented condition and \
would cause claim denial (e.g. a cardiology code on an orthopedic claim). \
Do NOT use CRITICAL for codes in the same family or category as the ideal code.
- WARNING: Use for near-miss codes — same procedure family or category as the best \
available option (e.g. different CT scan variants, different E/M levels). \
This is the correct severity when the coding agent selected the best available candidate \
from the knowledge base but a more specific code might exist.
- INFO: Documentation suggestions, secondary code recommendations, or observations \
that do not affect claim approval.

When validating CPT codes: only raise CRITICAL errors for codes that are completely \
unrelated to the documented procedure. For near-miss codes in the same category \
(e.g. different CT scan variants, different imaging modalities of the same region), \
use WARNING not CRITICAL. A claim should only fail when a code would clearly cause \
claim denial, not when a slightly more specific code could have been chosen.

Return ONLY valid JSON:
{
  "issues": [
    {
      "field": "field or code reference",
      "message": "specific issue description",
      "severity": "critical|warning|info"
    }
  ]
}

If no issues are found, return {"issues": []}.
JSON only, no markdown."""

_L2_USER_TEMPLATE = """\
Patient clinical entities:
{entities_json}

Assigned ICD-10 codes:
{icd10_json}

Assigned CPT-4 codes:
{cpt4_json}

JSON only, no markdown:"""


@traceable(name="validation_agent_llm_clinical_logic", process_inputs=_mask_state_trace_inputs)
def _run_level2(state: ClaimState) -> list[dict]:
    """GPT-4o clinical logic validation — one API call, reasoning only (no embeddings)."""
    entities = state.get("extracted_entities") or {}
    icd10_codes = _selected_icd10(state)
    cpt4_codes = _selected_cpt4(state)

    if not icd10_codes and not cpt4_codes:
        return []

    user_content = _L2_USER_TEMPLATE.format(
        entities_json=json.dumps(entities, indent=2),
        icd10_json=json.dumps(icd10_codes, indent=2),
        cpt4_json=json.dumps(cpt4_codes, indent=2),
    )

    try:
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        response = llm.invoke(
            [SystemMessage(content=_L2_SYSTEM_PROMPT), HumanMessage(content=user_content)]
        )
        content = response.content if hasattr(response, "content") else str(response)
        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```\w*\n?", "", content)
            content = re.sub(r"\n?```\s*$", "", content)
        parsed = json.loads(content)
        return parsed.get("issues", [])
    except json.JSONDecodeError as e:
        logger.warning("Level 2 LLM returned invalid JSON: %s", e)
        return []
    except Exception as e:
        logger.exception("Level 2 LLM call failed: %s", e)
        return []


# ---------------------------------------------------------------------------
# Level 3 — Confidence threshold check (deterministic, no API call)
# ---------------------------------------------------------------------------


def _run_level3(state: ClaimState) -> list[dict]:
    """
    Flag any selected codes whose confidence score falls below CONFIDENCE_THRESHOLD.
    >0.90 certain, 0.80–0.90 probable, <0.80 uncertain (per .cursorrules).
    """
    errors: list[dict] = []

    for entry in _selected_icd10(state):
        code = entry.get("code", "?")
        confidence = entry.get("confidence")
        if confidence is None:
            continue
        try:
            conf_float = float(confidence)
        except (ValueError, TypeError):
            continue
        if conf_float < CONFIDENCE_THRESHOLD:
            severity = "critical" if conf_float < 0.60 else "warning"
            errors.append(
                _error(
                    f"icd10:{code}",
                    f"ICD-10 code {code} has low confidence ({conf_float:.2f} < {CONFIDENCE_THRESHOLD}); manual review required",
                    severity,
                )
            )

    for entry in _selected_cpt4(state):
        code = entry.get("code", "?")
        confidence = entry.get("confidence")
        if confidence is None:
            continue
        try:
            conf_float = float(confidence)
        except (ValueError, TypeError):
            continue
        if conf_float < CONFIDENCE_THRESHOLD:
            severity = "critical" if conf_float < 0.60 else "warning"
            errors.append(
                _error(
                    f"cpt4:{code}",
                    f"CPT-4 code {code} has low confidence ({conf_float:.2f} < {CONFIDENCE_THRESHOLD}); manual review required",
                    severity,
                )
            )

    return errors


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------


def validation_agent(state: ClaimState) -> ClaimState:
    """
    LangGraph node: 3-level validation pipeline.

    Reads:  state['patient_name'], state['patient_dob'], state['patient_insurance_id']
            state['icd10_selected'], state['cpt4_selected'], state['extracted_entities']
    Writes: state['validation_errors']  — list of {field, message, severity} dicts
            state['validation_passed']  — True only when no critical errors remain
    """
    try:
        all_errors: list[dict[str, Any]] = []

        # Level 1 — deterministic rule checks (no API)
        l1_errors = _run_level1(state)
        all_errors.extend(l1_errors)
        logger.info("validation_agent Level 1: %d issue(s)", len(l1_errors))

        # Level 2 — GPT-4o clinical logic (one API call)
        l2_errors = _run_level2(state)
        all_errors.extend(l2_errors)
        logger.info("validation_agent Level 2: %d issue(s)", len(l2_errors))

        # Level 3 — confidence threshold check (no API)
        l3_errors = _run_level3(state)
        all_errors.extend(l3_errors)
        logger.info("validation_agent Level 3: %d issue(s)", len(l3_errors))

        # validation_passed = True  when zero CRITICAL errors (warnings are acceptable)
        # validation_passed = False only when there is at least one CRITICAL error
        has_critical = any(e.get("severity") == "critical" for e in all_errors)
        validation_passed = not has_critical

        warning_count = sum(1 for e in all_errors if e.get("severity") == "warning")
        logger.info(
            "validation_agent complete: %d total issue(s) "
            "(%d critical, %d warning), passed=%s",
            len(all_errors),
            sum(1 for e in all_errors if e.get("severity") == "critical"),
            warning_count,
            validation_passed,
        )

        return {
            **state,
            "validation_errors": all_errors,
            "validation_passed": validation_passed,
        }

    except Exception as e:
        logger.exception("validation_agent failed: %s", e)
        return {
            **state,
            "validation_errors": [
                _error("validation_agent", f"Validation agent internal error: {e}", "critical")
            ],
            "validation_passed": False,
        }

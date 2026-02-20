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
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langsmith import traceable

from orchestrator.state import ClaimState

logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.80"))


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
    ICD-10 codes must be at least 4 characters.
    3-char codes are category-level headers, not billable.
    """
    errors: list[dict] = []
    for entry in _selected_icd10(state):
        code = (entry.get("code") or "").strip()
        if code and len(code) <= 3:
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


@traceable(name="validation_agent_llm_clinical_logic")
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

        has_critical = any(e.get("severity") == "critical" for e in all_errors)
        validation_passed = not has_critical

        logger.info(
            "validation_agent complete: %d total issue(s), passed=%s",
            len(all_errors),
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

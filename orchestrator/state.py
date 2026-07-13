"""Claim state for the MediSuite AI Agent LangGraph workflow."""

from typing import Optional, TypedDict

# ---------------------------------------------------------------------------
# Selectable LLM models
# ---------------------------------------------------------------------------
# The user picks one of these on the Submit Claim page; the choice is carried
# through the pipeline via ClaimState["llm_model"] and read by each agent at
# call time. Note: gpt-5 / gpt-5.5 are reasoning models and do not accept the
# same params as gpt-4o (e.g. temperature) — see model_supports_temperature().

AVAILABLE_LLM_MODELS: tuple[str, ...] = ("gpt-4o", "gpt-5", "gpt-5.5")
DEFAULT_LLM_MODEL: str = "gpt-4o"


def resolve_llm_model(value: Optional[str]) -> str:
    """Return value if it is one of AVAILABLE_LLM_MODELS, else the default."""
    return value if value in AVAILABLE_LLM_MODELS else DEFAULT_LLM_MODEL


def model_supports_temperature(model: str) -> bool:
    """
    Only gpt-4o accepts a temperature parameter. The reasoning models
    (gpt-5 / gpt-5.5) reject it, so callers must omit temperature for them.
    """
    return model == "gpt-4o"


class ClaimState(TypedDict, total=False):
    """State passed between agents in the claims processing graph."""

    claim_id: str
    file_path: str  # optional; used by document_agent to extract raw_document_text
    patient_name: str
    patient_dob: str
    patient_insurance_id: str
    raw_document_text: str
    extracted_entities: dict  # diagnoses, procedures, medications, dates

    # LLM model selected per-request on the Submit Claim page. One of
    # AVAILABLE_LLM_MODELS; agents read this at call time instead of hardcoding.
    llm_model: str

    # Live pipeline progress stage, mirrored to the Claim DB row by each agent so
    # the frontend can poll GET /claims/{id}/status while the pipeline runs.
    # One of: document | coding | validation | claim | complete
    processing_stage: str

    # Optional patient fields submitted via the API form. Used ONLY as fallback
    # values when the Document Agent did not extract that field from the note.
    form_overrides: dict  # {patient_name, patient_dob, patient_insurance_id, patient_sex, patient_address, insurance_provider}

    # Stage 1 — raw semantic retrieval hits (ChromaDB / HuggingFace)
    icd10_candidates: list[dict]  # code, disease, category, score
    cpt4_candidates: list[dict]   # code, description, category, score

    # Stage 2 — GPT-4o reranked selections (coding_agent output)
    icd10_selected: dict  # {selected_codes: [{code, disease, confidence, reasoning, citation}], reasoning_chain}
    cpt4_selected: dict   # {selected_codes: [{code, description, confidence, reasoning, citation}], reasoning_chain}

    # validation_agent output
    validation_errors: list[dict]  # [{field, message, severity: critical|warning|info}]
    validation_passed: bool        # False if any critical errors exist
    revalidation_count: int        # incremented on each retry; capped at MAX_REVALIDATION_ATTEMPTS

    # claim_agent output
    claim_form_path: str    # path to generated ./claims/{claim_id}.json
    processing_status: str  # 'claim_generated' | 'claim_failed'
    requires_human_review: bool       # True when any code falls below CONFIDENCE_THRESHOLD
    low_confidence_codes: list[dict]  # codes that triggered human review

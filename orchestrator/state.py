"""Claim state for the MediSuite AI Agent LangGraph workflow."""

from typing import TypedDict


class ClaimState(TypedDict, total=False):
    """State passed between agents in the claims processing graph."""

    claim_id: str
    file_path: str  # optional; used by document_agent to extract raw_document_text
    patient_name: str
    patient_dob: str
    patient_insurance_id: str
    raw_document_text: str
    extracted_entities: dict  # diagnoses, procedures, medications, dates

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

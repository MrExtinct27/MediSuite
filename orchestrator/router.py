"""
Conditional edge functions for the MediSuite LangGraph workflow.
"""

from __future__ import annotations

import os

from orchestrator.state import ClaimState

MAX_REVALIDATION_ATTEMPTS: int = int(os.getenv("MAX_REVALIDATION_ATTEMPTS", "2"))


def route_after_validation(state: ClaimState) -> str:
    """
    Decide the next node after validation_agent.

    - validation_passed=True  → 'claim_agent'
    - validation_passed=False and revalidation_count < MAX_REVALIDATION_ATTEMPTS
                               → 'coding_agent'  (retry with same entities)
    - validation_passed=False and retries exhausted → 'claim_agent'
      (force-generate claim with warnings; processing_status is set to
       'claim_generated_with_warnings' by claim_agent reading validation_passed=False)
    """
    if state.get("validation_passed"):
        return "claim_agent"

    attempts: int = state.get("revalidation_count") or 0
    if attempts < MAX_REVALIDATION_ATTEMPTS:
        return "coding_agent"

    # Retries exhausted — always generate the claim rather than silently ending.
    return "claim_agent"

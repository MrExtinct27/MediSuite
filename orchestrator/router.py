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
    - validation_passed=False and retries exhausted → '__end__'
    """
    if state.get("validation_passed"):
        return "claim_agent"

    attempts: int = state.get("revalidation_count") or 0
    if attempts < MAX_REVALIDATION_ATTEMPTS:
        return "coding_agent"

    return "__end__"

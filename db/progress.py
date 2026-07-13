"""
Live pipeline progress helper.

Each agent calls set_processing_stage() as it begins so the Claim DB row always
reflects where the pipeline currently is. The frontend polls
GET /claims/{claim_id}/status (which just reads this column) once per second to
drive the live Pipeline Status tracker.

Best-effort by design: a failure to write progress must never break the pipeline,
so all DB errors are swallowed (and logged) rather than raised.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Ordered stages the pipeline moves through.
STAGES = ("document", "coding", "validation", "claim", "complete")


def set_processing_stage(
    claim_id: Optional[str],
    stage: str,
    status: Optional[str] = None,
) -> None:
    """
    Update a claim's processing_stage (and optionally status) in its own session.

    Uses a short-lived session so it is safe to call from any agent node without
    interfering with the request's DB session. Never raises into the pipeline.
    """
    if not claim_id:
        return
    try:
        from db.database import SessionLocal
        from db.models import Claim

        with SessionLocal() as session:
            claim = session.get(Claim, claim_id)
            if claim is None:
                # Row not created yet (or unknown id) — nothing to update.
                return
            claim.processing_stage = stage
            if status is not None:
                claim.status = status
            session.commit()
    except Exception as exc:  # never let progress tracking break the pipeline
        logger.warning(
            "Failed to set processing_stage=%s for claim %s: %s", stage, claim_id, exc
        )

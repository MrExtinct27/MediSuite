"""
MediSuite AI Agent — FastAPI application.

Startup sequence:
  1. Load .env
  2. Init DB (create tables)
  3. Ensure ChromaDB collections are populated (HuggingFace embeddings, no OpenAI)
  4. Include API routers
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from api.limiter import limiter
from api.routes import auth as auth_router
from api.routes import claims as claims_router
from db.database import init_db
from knowledge_base.embeddings import ensure_collections
from orchestrator.state import AVAILABLE_LLM_MODELS, DEFAULT_LLM_MODEL

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="MediSuite AI Agent",
    description="Automated ICD-10 / CPT-4 medical claims processing pipeline.",
    version="1.0.0",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event() -> None:
    # Log effective pipeline-loop guards so an env override is immediately visible.
    from orchestrator.router import MAX_REVALIDATION_ATTEMPTS
    logger.info(
        "MediSuite config: MAX_REVALIDATION_ATTEMPTS=%d, PIPELINE_TIMEOUT_SECONDS=%s",
        MAX_REVALIDATION_ATTEMPTS,
        os.getenv("PIPELINE_TIMEOUT_SECONDS", "300"),
    )

    logger.info("MediSuite startup: initialising database …")
    init_db()

    logger.info("MediSuite startup: ensuring ChromaDB collections …")
    # ChromaDB uses HuggingFace SentenceTransformer — no OpenAI call here
    ensure_collections()

    logger.info("MediSuite startup: ready.")


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


@app.get("/health", tags=["meta"])
def health() -> dict:
    """
    Liveness / readiness check.
    Reports embedding model (HuggingFace), the selectable LLM models (the default
    plus the full available list), and LangSmith status.
    No embedding or LLM calls are made here.
    """
    langsmith_enabled: bool = (
        os.getenv("LANGSMITH_TRACING", "").lower() == "true"
        and bool(os.getenv("LANGSMITH_API_KEY", "").strip())
    )
    return {
        "status": "ok",
        "embedding_model": os.getenv(
            "EMBEDDING_MODEL", "pritamdeka/S-PubMedBert-MS-MARCO"
        ),
        "default_llm_model": DEFAULT_LLM_MODEL,
        "available_llm_models": list(AVAILABLE_LLM_MODELS),
        "langsmith_enabled": langsmith_enabled,
    }


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(auth_router.router)
app.include_router(claims_router.router)
app.include_router(claims_router.cache_router)

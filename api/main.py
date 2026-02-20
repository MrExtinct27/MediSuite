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

from api.routes import claims as claims_router
from db.database import init_db
from knowledge_base.embeddings import ensure_collections

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event() -> None:
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
    Reports embedding model (HuggingFace), LLM model (GPT-4o), and LangSmith status.
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
        "llm_model": "gpt-4o",
        "langsmith_enabled": langsmith_enabled,
    }


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(claims_router.router)

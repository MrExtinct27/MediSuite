"""
ChromaDB setup with HuggingFace sentence-transformers embeddings.

Embedding model: pritamdeka/S-PubMedBert-MS-MARCO (biomedical domain).
Loaded via chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction.
Runs 100 % locally — no API key, no network call, completely free and offline.

Do NOT use OpenAI for any embedding operation.
OpenAI (GPT-4o) is only called by coding_agent.py during Stage 2 reranking.

Collections:
  - icd10_codes : {code, disease, category}  — embedded as '{code}: {disease} [{category}]'
  - cpt4_codes  : {code, description, category} — embedded as '{code}: {description} [{category}]'

Public API used by coding_agent.py:
  from knowledge_base.embeddings import search_icd10, search_cpt4

  search_icd10(query, n_results=20) — query is embedded locally by HuggingFace; no API call
  search_cpt4 (query, n_results=20) — same
"""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (reads from environment / .env)
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_PERSIST_DIR = str(_REPO_ROOT / "chroma_data")
_DEFAULT_EMBEDDING_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", _DEFAULT_PERSIST_DIR)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", _DEFAULT_EMBEDDING_MODEL)

ICD10_COLLECTION = "icd10_codes"
CPT4_COLLECTION = "cpt4_codes"

_ICD10_JSON = _REPO_ROOT / "knowledge_base" / "ICD10.json"
_CPT4_JSON = _REPO_ROOT / "knowledge_base" / "CPT4.json"

# ---------------------------------------------------------------------------
# Singleton ChromaDB client + embedding function
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _embedding_fn() -> SentenceTransformerEmbeddingFunction:
    """
    Return a cached HuggingFace SentenceTransformer embedding function.
    ChromaDB uses this to embed both documents (at ingest) and queries (at search time).
    No OpenAI API key is required or used here.
    """
    logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
    return SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def _chroma_client() -> chromadb.PersistentClient:
    """Return a cached persistent ChromaDB client."""
    Path(CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
    logger.info("ChromaDB persist dir: %s", CHROMA_PERSIST_DIR)
    return chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)


# ---------------------------------------------------------------------------
# Collection access helpers
# ---------------------------------------------------------------------------


def _get_collection(name: str) -> chromadb.Collection:
    return _chroma_client().get_or_create_collection(
        name=name,
        embedding_function=_embedding_fn(),
        metadata={"hnsw:space": "cosine"},
    )


# ---------------------------------------------------------------------------
# Ingestion helpers (idempotent — skip if already populated)
# ---------------------------------------------------------------------------

_BATCH_SIZE = 512  # safe upper limit for ChromaDB upsert


def _ingest_icd10() -> None:
    """Load ICD10.json into ChromaDB collection (idempotent)."""
    collection = _get_collection(ICD10_COLLECTION)
    if collection.count() > 0:
        logger.debug("ICD-10 collection already populated (%d docs).", collection.count())
        return

    logger.info("Ingesting ICD-10 codes from %s …", _ICD10_JSON)
    with open(_ICD10_JSON, encoding="utf-8") as f:
        records: list[dict[str, Any]] = json.load(f)

    _batch_upsert(
        collection=collection,
        records=records,
        doc_fn=lambda r: f"{r['code']}: {r['disease']} [{r.get('category', '')}]",
        meta_fn=lambda r: {"code": r["code"], "disease": r["disease"], "category": r.get("category", "")},
        id_fn=lambda r: r["code"],
    )
    logger.info("ICD-10 ingestion complete — %d docs.", collection.count())


def _ingest_cpt4() -> None:
    """Load CPT4.json into ChromaDB collection (idempotent)."""
    collection = _get_collection(CPT4_COLLECTION)
    if collection.count() > 0:
        logger.debug("CPT-4 collection already populated (%d docs).", collection.count())
        return

    logger.info("Ingesting CPT-4 codes from %s …", _CPT4_JSON)
    with open(_CPT4_JSON, encoding="utf-8") as f:
        records: list[dict[str, Any]] = json.load(f)

    _batch_upsert(
        collection=collection,
        records=records,
        doc_fn=lambda r: f"{r['code']}: {r.get('procedure', r.get('description', ''))} [{r.get('category', '')}]",
        meta_fn=lambda r: {
            "code": r["code"],
            "description": r.get("procedure", r.get("description", "")),
            "category": r.get("category", ""),
        },
        id_fn=lambda r: r["code"],
    )
    logger.info("CPT-4 ingestion complete — %d docs.", collection.count())


def _batch_upsert(
    collection: chromadb.Collection,
    records: list[dict],
    doc_fn: Any,
    meta_fn: Any,
    id_fn: Any,
) -> None:
    """Upsert records into a ChromaDB collection in safe-sized batches.
    Deduplicates by ID before upserting to avoid ChromaDB DuplicateIDError.
    """
    # Deduplicate by ID — keep last occurrence
    seen: dict = {}
    for r in records:
        seen[id_fn(r)] = r
    unique_records = list(seen.values())

    ids, docs, metas = [], [], []
    for r in unique_records:
        ids.append(id_fn(r))
        docs.append(doc_fn(r))
        metas.append(meta_fn(r))

        if len(ids) >= _BATCH_SIZE:
            collection.upsert(ids=ids, documents=docs, metadatas=metas)
            ids, docs, metas = [], [], []

    if ids:
        collection.upsert(ids=ids, documents=docs, metadatas=metas)


# ---------------------------------------------------------------------------
# Public API — call ensure_collections() once at startup
# ---------------------------------------------------------------------------


def ensure_collections() -> None:
    """Ingest ICD-10 and CPT-4 into ChromaDB if not already present."""
    _ingest_icd10()
    _ingest_cpt4()


def search_icd10(query: str, n_results: int = 20) -> list[dict[str, Any]]:
    """
    Semantic search over the ICD-10 ChromaDB collection.

    The query string is embedded locally by the HuggingFace SentenceTransformer model
    (pritamdeka/S-PubMedBert-MS-MARCO). No OpenAI API call is made here.
    This is Stage 1 retrieval — completely free and offline.

    Returns up to n_results dicts: {code, disease, category, score}.
    Score is cosine similarity (1.0 = identical, 0.0 = orthogonal).
    """
    if not query or not query.strip():
        return []
    try:
        collection = _get_collection(ICD10_COLLECTION)
        results = collection.query(query_texts=[query], n_results=min(n_results, collection.count() or n_results))
        return _format_results(results, key_map={"code": "code", "disease": "disease", "category": "category"})
    except Exception as e:
        logger.exception("ICD-10 search failed: %s", e)
        return []


def search_cpt4(query: str, n_results: int = 20) -> list[dict[str, Any]]:
    """
    Semantic search over the CPT-4 ChromaDB collection.

    The query string is embedded locally by the HuggingFace SentenceTransformer model
    (pritamdeka/S-PubMedBert-MS-MARCO). No OpenAI API call is made here.
    This is Stage 1 retrieval — completely free and offline.

    Returns up to n_results dicts: {code, description, category, score}.
    Score is cosine similarity (1.0 = identical, 0.0 = orthogonal).
    """
    if not query or not query.strip():
        return []
    try:
        collection = _get_collection(CPT4_COLLECTION)
        results = collection.query(query_texts=[query], n_results=min(n_results, collection.count() or n_results))
        return _format_results(results, key_map={"code": "code", "description": "description", "category": "category"})
    except Exception as e:
        logger.exception("CPT-4 search failed: %s", e)
        return []


def _format_results(raw: dict, key_map: dict[str, str]) -> list[dict[str, Any]]:
    """Convert ChromaDB query response into a flat list of result dicts with a score field."""
    out: list[dict[str, Any]] = []
    metas = raw.get("metadatas", [[]])[0]
    distances = raw.get("distances", [[]])[0]

    for meta, dist in zip(metas, distances):
        # ChromaDB cosine distance: 0 = identical → convert to similarity score
        score = round(1.0 - float(dist), 4)
        entry: dict[str, Any] = {"score": score}
        for result_key, meta_key in key_map.items():
            entry[result_key] = meta.get(meta_key, "")
        out.append(entry)

    return out

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    
    print("🔄 Starting ChromaDB ingestion...")
    print(f"   Persist dir : {CHROMA_PERSIST_DIR}")
    print(f"   Model       : {EMBEDDING_MODEL}")
    print()
    
    print("📥 Embedding ICD-10 codes (this may take a while on first run)...")
    _ingest_icd10()
    icd_count = _get_collection(ICD10_COLLECTION).count()
    print(f"✅ {icd_count} codes embedded into icd10_codes")
    print()
    
    print("📥 Embedding CPT-4 codes...")
    _ingest_cpt4()
    cpt_count = _get_collection(CPT4_COLLECTION).count()
    print(f"✅ {cpt_count} codes embedded into cpt4_codes")
    print()
    
    print("🔍 Running test search: 'type 2 diabetes with kidney complications'")
    results = search_icd10("type 2 diabetes with kidney complications", n_results=3)
    for r in results:
        print(f"   {r['code']} | {r['disease']} | score: {r['score']}")
    print()
    
    print("🔍 Running test search: 'office visit established patient'")
    results = search_cpt4("office visit established patient", n_results=3)
    for r in results:
        print(f"   {r['code']} | {r['description']} | score: {r['score']}")
    print()
    
    print("🎉 ChromaDB ready!")
"""
ChromaDB retrieval with HuggingFace Inference API embeddings.

Embedding model: pritamdeka/S-PubMedBert-MS-MARCO (biomedical domain, 768-dim).
Embeddings are computed by calling the HuggingFace Inference API
(huggingface_hub.InferenceClient, provider="hf-inference") instead of loading the
model locally. This removes the sentence-transformers / torch memory footprint for
deployment. The API returns vectors identical to the local model (verified: cosine
similarity 1.000000, same 768 dimensions), so the existing ChromaDB collections
remain valid and are NOT re-embedded.

Requires HF_TOKEN in the environment. Do NOT set HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE /
HF_DATASETS_OFFLINE — those block the network calls this module now depends on.

Do NOT use OpenAI for any embedding operation.
OpenAI (GPT-4o etc.) is only called by coding_agent.py during Stage 2 reranking.

Collections (queried, never re-embedded here):
  - icd10_codes : {code, disease, category}     — 71,724 vectors
  - cpt4_codes  : {code, description, category}  — 8,227 vectors

Public API used by coding_agent.py:
  from knowledge_base.embeddings import search_icd10, search_cpt4

  search_icd10(query, n_results=20) — query embedded via the HF Inference API
  search_cpt4 (query, n_results=20) — same
"""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import chromadb
import numpy as np
from huggingface_hub import InferenceClient

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).parent
_DEFAULT_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"
# Cosine distance space — must match how the collections were created so retrieval
# scores stay in the expected 0–1 similarity range (not the cloud L2 default).
_COSINE_CONFIG = {"hnsw": {"space": "cosine"}}


# ---------------------------------------------------------------------------
# HuggingFace Inference API embedding
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _inference_client() -> InferenceClient:
    """
    Return a cached HuggingFace InferenceClient. Created once at first use (not per
    call) so the underlying HTTP connection pool is reused across all queries.
    Requires HF_TOKEN; raises clearly if it is missing.
    """
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN is not set. The HuggingFace Inference API requires an API token; "
            "set HF_TOKEN in the environment (.env)."
        )
    return InferenceClient(provider="hf-inference", api_key=token)


def _embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed one or more texts via the HuggingFace Inference API and return a list of
    768-dim float vectors (one per input text).

    The Inference API's feature_extraction takes a single string, so a list is
    embedded one call per text (queries are always a single text; batch ingestion
    loops). Errors from the API — auth, rate limit (429), timeout, connection —
    are allowed to propagate: this function NEVER returns an empty vector on
    failure, because doing so would silently break retrieval.
    """
    client = _inference_client()
    model = os.getenv("EMBEDDING_MODEL", _DEFAULT_MODEL)

    vectors: list[list[float]] = []
    for text in texts:
        # feature_extraction raises on API error / rate limit / timeout — do not catch.
        raw = client.feature_extraction(text, model=model)

        arr = np.asarray(raw, dtype=np.float32)
        # Some endpoints return token-level output (seq_len, dim); mean-pool to a
        # single sentence embedding — matches the verified local-vs-API comparison.
        if arr.ndim > 1:
            arr = arr.mean(axis=0)

        if arr.ndim != 1 or arr.size == 0:
            raise RuntimeError(
                f"HuggingFace Inference API returned an unexpected embedding shape "
                f"{arr.shape} for model {model!r}; refusing to use an empty/invalid vector."
            )
        vectors.append(arr.tolist())

    return vectors


# ---------------------------------------------------------------------------
# ChromaDB client / collections
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _chroma_client() -> chromadb.api.ClientAPI:
    """
    Return a cached Chroma Cloud client. Created once (lru_cache) so the HTTP
    connection is reused across all queries — never per request.

    Fails loudly if credentials are missing or the connection fails: we never
    silently fall back to an empty/local store, consistent with the retrieval
    error-propagation policy.
    """
    missing = [
        var for var in ("CHROMA_API_KEY", "CHROMA_TENANT", "CHROMA_DATABASE")
        if not os.getenv(var, "").strip()
    ]
    if missing:
        raise RuntimeError(
            f"Chroma Cloud credentials missing: {', '.join(missing)}. "
            "Set them in the environment (.env)."
        )
    logger.info("Connecting to Chroma Cloud (database=%s)", os.environ["CHROMA_DATABASE"])
    return chromadb.CloudClient(
        api_key=os.environ["CHROMA_API_KEY"].strip(),
        tenant=os.environ["CHROMA_TENANT"].strip(),
        database=os.environ["CHROMA_DATABASE"].strip(),
    )


def _open_collection(name: str) -> chromadb.Collection:
    """
    Open an EXISTING (cloud) collection for querying. No embedding function is
    attached: queries pass explicit query_embeddings computed via the HF Inference
    API. Raises if the collection does not exist or the connection fails — never
    swallows the error.
    """
    return _chroma_client().get_collection(name=name)


def _batch_upsert(
    collection: chromadb.Collection,
    records: list[dict],
    doc_fn: Callable[[dict], str],
    meta_fn: Callable[[dict], dict],
    id_fn: Callable[[dict], str],
) -> None:
    """Upsert records into a ChromaDB collection in safe-sized batches.
    Deduplicates by ID before upserting to avoid ChromaDB DuplicateIDError.

    Embeddings are computed explicitly via the HF Inference API and passed to
    upsert (the collection has no embedding function), so documents are never
    embedded by a local model. NOTE: this only runs on a fresh ingest; the
    deployed collections are already populated and are never re-embedded.
    """
    seen: dict[str, dict] = {}
    for r in records:
        id_ = id_fn(r)
        if id_ not in seen:
            seen[id_] = r
    unique = list(seen.values())

    # Chroma Cloud caps upserts at 300 records per request — stay under it.
    batch_size = 250
    for i in range(0, len(unique), batch_size):
        batch = unique[i : i + batch_size]
        documents = [doc_fn(r) for r in batch]
        collection.upsert(
            ids=[id_fn(r) for r in batch],
            embeddings=_embed_texts(documents),
            documents=documents,
            metadatas=[meta_fn(r) for r in batch],
        )


def _ingest_icd10() -> None:
    """
    Ensure the ICD-10 collection is populated. On Chroma Cloud the collection
    already exists and is populated, so this detects that and skips — it never
    re-embeds on boot. Only a genuinely missing/empty collection is ingested.

    A connection/auth failure propagates (fail loudly); it is NOT mistaken for an
    absent collection, so we never fall through to re-embedding against the cloud.
    """
    try:
        existing = _open_collection("icd10_codes")
    except chromadb.errors.NotFoundError:
        existing = None  # genuinely absent — everything else propagates

    if existing is not None:
        count = existing.count()
        if count > 0:
            logger.info("ICD-10 collection already populated (%d docs) — skipping ingestion.", count)
            return

    col = existing or _chroma_client().get_or_create_collection(
        name="icd10_codes", configuration=_COSINE_CONFIG
    )
    path = _DATA_DIR / "ICD10.json"
    logger.info("Ingesting ICD-10 codes from %s …", path)
    with open(path, encoding="utf-8") as f:
        data: list[dict] = json.load(f)

    _batch_upsert(
        col,
        data,
        doc_fn=lambda r: (
            f"{r.get('code', '')}: {r.get('disease', '')} [{r.get('category', '')}]"
        ),
        meta_fn=lambda r: {
            "code": r.get("code", ""),
            "disease": r.get("disease", ""),
            "category": r.get("category", ""),
        },
        id_fn=lambda r: r["code"],
    )
    logger.info("ICD-10 ingestion complete — %d docs.", col.count())


def _ingest_cpt4() -> None:
    """
    Ensure the CPT-4 collection is populated. See _ingest_icd10 — on Chroma Cloud
    this detects the already-populated collection and skips (no re-embedding on
    boot); connection/auth failures propagate rather than triggering ingestion.
    """
    try:
        existing = _open_collection("cpt4_codes")
    except chromadb.errors.NotFoundError:
        existing = None

    if existing is not None:
        count = existing.count()
        if count > 0:
            logger.info("CPT-4 collection already populated (%d docs) — skipping ingestion.", count)
            return

    col = existing or _chroma_client().get_or_create_collection(
        name="cpt4_codes", configuration=_COSINE_CONFIG
    )
    path = _DATA_DIR / "CPT4.json"
    logger.info("Ingesting CPT-4 codes from %s …", path)
    with open(path, encoding="utf-8") as f:
        data: list[dict] = json.load(f)

    _batch_upsert(
        col,
        data,
        doc_fn=lambda r: (
            f"{r.get('code', '')}: "
            f"{r.get('description') or r.get('procedure', '')} "
            f"[{r.get('category', '')}]"
        ),
        meta_fn=lambda r: {
            "code": r.get("code", ""),
            "description": r.get("description") or r.get("procedure", ""),
            "category": r.get("category", ""),
        },
        id_fn=lambda r: r["code"],
    )
    logger.info("CPT-4 ingestion complete — %d docs.", col.count())


def ensure_collections() -> None:
    """
    Verify the Chroma Cloud ICD-10 and CPT-4 collections are present and populated.
    Since they are already populated in the cloud, this skips ingestion — it does
    NOT re-embed on every boot. Fails loudly on missing credentials / connection.
    """
    _ingest_icd10()
    _ingest_cpt4()


def _format_results(raw: dict, key_map: dict[str, str]) -> list[dict[str, Any]]:
    """Convert ChromaDB query response into a flat list of result dicts with a score field."""
    results = []
    metadatas = (raw.get("metadatas") or [[]])[0]
    distances = (raw.get("distances") or [[]])[0]
    for meta, dist in zip(metadatas, distances):
        entry: dict[str, Any] = {
            out_key: meta.get(src_key, "") for out_key, src_key in key_map.items()
        }
        entry["score"] = round(1.0 - float(dist), 6)
        results.append(entry)
    return results


def search_icd10(query: str, n_results: int = 20) -> list[dict[str, Any]]:
    """
    Semantic search over the ICD-10 ChromaDB collection.

    The query is embedded via the HuggingFace Inference API
    (pritamdeka/S-PubMedBert-MS-MARCO) and matched against the pre-embedded
    collection. This is Stage 1 retrieval — no OpenAI call is made here.

    Returns up to n_results dicts: {code, disease, category, score}.
    Score is cosine similarity (1.0 = identical, 0.0 = orthogonal).

    Errors (missing HF_TOKEN, API failure, rate limit, timeout) propagate to the
    caller rather than being swallowed into an empty result.
    """
    col = _open_collection("icd10_codes")
    query_vector = _embed_texts([query])[0]
    raw = col.query(query_embeddings=[query_vector], n_results=n_results)
    return _format_results(
        raw, {"code": "code", "disease": "disease", "category": "category"}
    )


def search_cpt4(query: str, n_results: int = 20) -> list[dict[str, Any]]:
    """
    Semantic search over the CPT-4 ChromaDB collection.

    The query is embedded via the HuggingFace Inference API
    (pritamdeka/S-PubMedBert-MS-MARCO) and matched against the pre-embedded
    collection. This is Stage 1 retrieval — no OpenAI call is made here.

    Returns up to n_results dicts: {code, description, category, score}.
    Score is cosine similarity (1.0 = identical, 0.0 = orthogonal).

    Errors (missing HF_TOKEN, API failure, rate limit, timeout) propagate to the
    caller rather than being swallowed into an empty result.
    """
    col = _open_collection("cpt4_codes")
    query_vector = _embed_texts([query])[0]
    raw = col.query(query_embeddings=[query_vector], n_results=n_results)
    return _format_results(
        raw, {"code": "code", "description": "description", "category": "category"}
    )


if __name__ == "__main__":
    # Load .env so CHROMA_* / EMBEDDING_MODEL / HF_TOKEN are set before any
    # cached client or inference client is constructed.
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    model = os.getenv("EMBEDDING_MODEL", _DEFAULT_MODEL)

    print("🔄 Verifying Chroma Cloud collections...")
    print("   Database    : " + os.getenv("CHROMA_DATABASE", "(unset)"))
    print("   Model       : " + model + " (HuggingFace Inference API)")

    print("📥 Ensuring ICD-10 codes are embedded (skips if already populated)...")
    _ingest_icd10()
    print(f"✅ {_open_collection('icd10_codes').count()} codes in icd10_codes")

    print("📥 Ensuring CPT-4 codes are embedded...")
    _ingest_cpt4()
    print(f"✅ {_open_collection('cpt4_codes').count()} codes in cpt4_codes")

    print("🔍 Running test search: 'type 2 diabetes with kidney complications'")
    for r in search_icd10("type 2 diabetes with kidney complications", n_results=3):
        print(f"   {r['code']} | {r['disease']} | score: {r['score']:.4f}")

    print("🔍 Running test search: 'office visit established patient'")
    for r in search_cpt4("office visit established patient", n_results=3):
        print(f"   {r['code']} | {r['description']} | score: {r['score']:.4f}")

    print("🎉 ChromaDB ready!")

"""
Coding Agent — LangGraph node.
2-stage pipeline: Semantic Retrieval (ChromaDB/HuggingFace) → LLM Reranking (GPT-4o).

Stage 1 — Semantic Retrieval  (NO API KEY NEEDED, completely free & offline)
  - Calls search_icd10() / search_cpt4() from knowledge_base/embeddings.py
  - Those functions embed the query locally using HuggingFace SentenceTransformer
    (pritamdeka/S-PubMedBert-MS-MARCO) via chromadb SentenceTransformerEmbeddingFunction
  - No OpenAI API call occurs during retrieval

Stage 2 — LLM Reranking  (OPENAI API CALL — requires OPENAI_API_KEY)
  - Passes the retrieved candidates to GPT-4o via langchain-openai ChatOpenAI
  - GPT-4o reasons over the candidates and selects the best codes
  - This is the only point in the coding pipeline that touches the OpenAI API
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langsmith import traceable

# search_icd10 / search_cpt4 embed queries locally via HuggingFace — no OpenAI call at retrieval time
from knowledge_base.embeddings import search_cpt4, search_icd10
from orchestrator.state import ClaimState

logger = logging.getLogger(__name__)

_N_RETRIEVAL_RESULTS = 20  # ChromaDB candidates per query
_MAX_SELECTED_CODES = 5


# ---------------------------------------------------------------------------
# Stage 1 — Semantic Retrieval (ChromaDB + HuggingFace embeddings)
# ---------------------------------------------------------------------------


def _build_retrieval_query(item: dict[str, Any]) -> str:
    """
    Combine description + source_text excerpt into a rich retrieval query,
    as specified: '{description} {source_text excerpt}'.
    """
    description = item.get("description", "") or ""
    source_text = item.get("source_text", "") or ""
    # Trim source_text to avoid overly long queries
    source_snippet = source_text[:300].strip()
    return f"{description} {source_snippet}".strip()


def _retrieve_icd10_candidates(entities: dict[str, Any]) -> list[dict]:
    """
    For each diagnosis in extracted_entities, run semantic search and
    collect unique ICD-10 candidates (deduped by code, keeping highest score).
    """
    diagnoses = entities.get("diagnoses", []) or []
    seen: dict[str, dict] = {}

    for diag in diagnoses:
        if not isinstance(diag, dict):
            continue
        query = _build_retrieval_query(diag)
        if not query:
            continue
        try:
            candidates = search_icd10(query, n_results=_N_RETRIEVAL_RESULTS)
            for c in candidates:
                code = c.get("code", "")
                if code not in seen or c.get("score", 0) > seen[code].get("score", 0):
                    seen[code] = c
        except Exception as e:
            logger.exception("ICD-10 retrieval error for diagnosis '%s': %s", diag.get("description", ""), e)

    return list(seen.values())


def _retrieve_cpt4_candidates(entities: dict[str, Any]) -> list[dict]:
    """
    For each procedure in extracted_entities, run semantic search and
    collect unique CPT-4 candidates (deduped by code, keeping highest score).
    """
    procedures = entities.get("procedures", []) or []
    seen: dict[str, dict] = {}

    for proc in procedures:
        if not isinstance(proc, dict):
            continue
        query = _build_retrieval_query(proc)
        if not query:
            continue
        try:
            candidates = search_cpt4(query, n_results=_N_RETRIEVAL_RESULTS)
            for c in candidates:
                code = c.get("code", "")
                if code not in seen or c.get("score", 0) > seen[code].get("score", 0):
                    seen[code] = c
        except Exception as e:
            logger.exception("CPT-4 retrieval error for procedure '%s': %s", proc.get("description", ""), e)

    return list(seen.values())


# ---------------------------------------------------------------------------
# Stage 2 — LLM Reranking (GPT-4o, reasoning only — no embeddings)
# ---------------------------------------------------------------------------

_ICD10_SYSTEM_PROMPT = """\
You are a certified medical coding specialist (CPC) with expertise in ICD-10 coding. \
Your task is to select the most accurate ICD-10 codes from the provided candidates \
based on the clinical documentation."""

_ICD10_USER_TEMPLATE = """\
You will receive:
1. The extracted clinical entities from the patient document
2. A list of candidate ICD-10 codes retrieved by semantic search

Your response MUST be valid JSON in this exact format:
{{
  "selected_codes": [
    {{
      "code": "exact ICD-10 code from candidates",
      "disease": "disease description",
      "confidence": 0.0-1.0,
      "reasoning": "step-by-step explanation of why this code applies",
      "citation": "exact text from the clinical document supporting this code"
    }}
  ],
  "reasoning_chain": "overall explanation of coding decisions made"
}}

Chain-of-thought rules:
Step 1: Identify the primary diagnosis from the clinical entities
Step 2: Review each candidate code description carefully
Step 3: Match code specificity to documentation (use most specific code supported)
Step 4: Check if secondary diagnoses need separate codes
Step 5: Assign confidence: >0.90 certain, 0.80-0.90 probable, <0.80 uncertain

Only select codes from the provided candidates list. Maximum {max_codes} codes.
If no candidate is appropriate, return an empty selected_codes array.

Clinical entities:
{entities_json}

ICD-10 candidates (code | disease | category | semantic score):
{candidates_text}

JSON only, no markdown:"""

_CPT4_SYSTEM_PROMPT = """\
You are a certified medical coding specialist (CPC) with expertise in CPT-4 procedure coding. \
Your task is to select the most accurate CPT-4 codes from the provided candidates \
based on the clinical documentation."""

_CPT4_USER_TEMPLATE = """\
You will receive:
1. The extracted clinical entities from the patient document
2. A list of candidate CPT-4 codes retrieved by semantic search

Your response MUST be valid JSON in this exact format:
{{
  "selected_codes": [
    {{
      "code": "exact CPT-4 code from candidates",
      "description": "procedure description",
      "confidence": 0.0-1.0,
      "reasoning": "step-by-step explanation of why this code applies",
      "citation": "exact text from the clinical document supporting this code"
    }}
  ],
  "reasoning_chain": "overall explanation of coding decisions made"
}}

Chain-of-thought rules:
Step 1: Identify the primary procedure from the clinical entities
Step 2: Review each candidate code description carefully
Step 3: Match code specificity to documentation (use most specific code supported)
Step 4: Check if additional procedures need separate codes
Step 5: Assign confidence: >0.90 certain, 0.80-0.90 probable, <0.80 uncertain

Only select codes from the provided candidates list. Maximum {max_codes} codes.
If no candidate is appropriate, return an empty selected_codes array.

Clinical entities:
{entities_json}

CPT-4 candidates (code | description | category | semantic score):
{candidates_text}

JSON only, no markdown:"""


def _candidates_to_text(candidates: list[dict], description_key: str) -> str:
    """Render candidates as a compact table string for the LLM prompt."""
    rows = []
    for c in candidates:
        score = c.get("score", 0.0)
        rows.append(
            f"  {c.get('code', '')} | {c.get(description_key, '')} | {c.get('category', '')} | {score:.4f}"
        )
    return "\n".join(rows) if rows else "  (none)"


def _parse_llm_json(content: str, stage: str) -> dict[str, Any]:
    """Strip markdown fences and parse JSON from LLM response."""
    import re

    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```\w*\n?", "", content)
        content = re.sub(r"\n?```\s*$", "", content)
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        logger.warning("%s reranking: LLM returned invalid JSON: %s", stage, e)
        return {"selected_codes": [], "reasoning_chain": ""}


@traceable(name="coding_agent_rerank_icd10")
def _rerank_icd10(entities: dict[str, Any], candidates: list[dict]) -> dict[str, Any]:
    """GPT-4o reranking for ICD-10 candidates. Reasoning/reranking only — no embeddings."""
    if not candidates:
        return {"selected_codes": [], "reasoning_chain": "No ICD-10 candidates to rerank."}

    candidates_text = _candidates_to_text(candidates, "disease")
    entities_json = json.dumps(entities, indent=2)

    user_content = _ICD10_USER_TEMPLATE.format(
        max_codes=_MAX_SELECTED_CODES,
        entities_json=entities_json,
        candidates_text=candidates_text,
    )

    try:
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        response = llm.invoke(
            [SystemMessage(content=_ICD10_SYSTEM_PROMPT), HumanMessage(content=user_content)]
        )
        content = response.content if hasattr(response, "content") else str(response)
        return _parse_llm_json(content, "ICD-10")
    except Exception as e:
        logger.exception("ICD-10 reranking LLM call failed: %s", e)
        return {"selected_codes": [], "reasoning_chain": ""}


@traceable(name="coding_agent_rerank_cpt4")
def _rerank_cpt4(entities: dict[str, Any], candidates: list[dict]) -> dict[str, Any]:
    """GPT-4o reranking for CPT-4 candidates. Reasoning/reranking only — no embeddings."""
    if not candidates:
        return {"selected_codes": [], "reasoning_chain": "No CPT-4 candidates to rerank."}

    candidates_text = _candidates_to_text(candidates, "description")
    entities_json = json.dumps(entities, indent=2)

    user_content = _CPT4_USER_TEMPLATE.format(
        max_codes=_MAX_SELECTED_CODES,
        entities_json=entities_json,
        candidates_text=candidates_text,
    )

    try:
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        response = llm.invoke(
            [SystemMessage(content=_CPT4_SYSTEM_PROMPT), HumanMessage(content=user_content)]
        )
        content = response.content if hasattr(response, "content") else str(response)
        return _parse_llm_json(content, "CPT-4")
    except Exception as e:
        logger.exception("CPT-4 reranking LLM call failed: %s", e)
        return {"selected_codes": [], "reasoning_chain": ""}


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------


def coding_agent(state: ClaimState) -> ClaimState:
    """
    LangGraph node: 2-stage RAG + LLM reranking for ICD-10 and CPT-4 codes.

    Reads:  state['extracted_entities']
    Writes: state['icd10_candidates']     — raw semantic hits (Stage 1)
            state['cpt4_candidates']      — raw semantic hits (Stage 1)
            state['icd10_selected']       — GPT-4o reranked codes (Stage 2)
            state['cpt4_selected']        — GPT-4o reranked codes (Stage 2)
    """
    try:
        entities: dict[str, Any] = state.get("extracted_entities") or {}

        # --- Stage 1: Semantic retrieval — HuggingFace embeddings, no API call ----------
        icd10_candidates = _retrieve_icd10_candidates(entities)
        cpt4_candidates = _retrieve_cpt4_candidates(entities)

        logger.info(
            "coding_agent Stage 1: %d ICD-10 candidates, %d CPT-4 candidates",
            len(icd10_candidates),
            len(cpt4_candidates),
        )

        # --- Stage 2: LLM reranking — GPT-4o via OpenAI API (first & only API call) ----
        icd10_result = _rerank_icd10(entities, icd10_candidates)
        cpt4_result = _rerank_cpt4(entities, cpt4_candidates)

        logger.info(
            "coding_agent Stage 2: %d ICD-10 selected, %d CPT-4 selected",
            len(icd10_result.get("selected_codes", [])),
            len(cpt4_result.get("selected_codes", [])),
        )

        return {
            **state,
            "icd10_candidates": icd10_candidates,
            "cpt4_candidates": cpt4_candidates,
            "icd10_selected": icd10_result,
            "cpt4_selected": cpt4_result,
        }

    except Exception as e:
        logger.exception("coding_agent failed: %s", e)
        return {
            **state,
            "icd10_candidates": state.get("icd10_candidates", []),
            "cpt4_candidates": state.get("cpt4_candidates", []),
            "icd10_selected": {"selected_codes": [], "reasoning_chain": ""},
            "cpt4_selected": {"selected_codes": [], "reasoning_chain": ""},
        }

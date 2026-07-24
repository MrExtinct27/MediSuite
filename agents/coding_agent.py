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

import asyncio
import hashlib
import json
import logging
import os
import time
from typing import Any

from langsmith import traceable
from openai import AsyncOpenAI

# search_icd10 / search_cpt4 embed queries locally via HuggingFace — no OpenAI call at retrieval time
from db.progress import set_processing_stage
from knowledge_base.embeddings import search_cpt4, search_icd10
from orchestrator.state import (
    ClaimState,
    model_supports_temperature,
    resolve_llm_model,
)

logger = logging.getLogger(__name__)

# Shared async client — one persistent connection pool for all reranking calls.
# Do NOT instantiate inside the rerank functions; that creates a new httpx client each time.
# The client is model-agnostic; the model string is chosen per request (see state["llm_model"]).
_async_client = AsyncOpenAI()


def _chat_completion_kwargs(model: str) -> dict:
    """
    Build the model-specific kwargs for a chat completion call. Temperature is
    only accepted by gpt-4o; the reasoning models (gpt-5 / gpt-5.5) reject it,
    so it is omitted for them.
    """
    kwargs: dict = {"model": model}
    if model_supports_temperature(model):
        kwargs["temperature"] = 0
    return kwargs

_N_RETRIEVAL_RESULTS = 20  # ChromaDB candidates per query
_MAX_SELECTED_CODES = 8   # raised from 5: primary dx + secondary dx + mechanism + place + activity

# ---------------------------------------------------------------------------
# PHI masking — used for LangSmith trace inputs only; never for agent logic
# ---------------------------------------------------------------------------

_PHI_FIELDS = [
    "patient_name", "patient_dob", "patient_insurance_id",
    "name", "dob", "insurance_id",
]


def mask_phi_for_logging(state: dict) -> dict:
    """Return a shallow copy of state with PHI fields and raw document replaced."""
    masked = state.copy()
    for field in _PHI_FIELDS:
        if masked.get(field):
            masked[field] = "[PHI-REDACTED]"
    if "raw_document_text" in masked:
        masked["raw_document_text"] = "[DOCUMENT-REDACTED]"
    return masked


def _mask_rerank_trace_inputs(inputs: dict) -> dict:
    """process_inputs hook for @traceable — redacts PHI from the entities dict."""
    if "entities" in inputs:
        return {**inputs, "entities": mask_phi_for_logging(inputs["entities"])}
    return inputs


# ---------------------------------------------------------------------------
# In-memory ChromaDB retrieval cache
# ---------------------------------------------------------------------------

_retrieval_cache: dict[str, list[dict]] = {}


def _get_cache_key(query: str, n_results: int, collection: str) -> str:
    return hashlib.md5(f"{collection}:{n_results}:{query}".encode()).hexdigest()


def _cached_search_icd10(query: str, n_results: int) -> list[dict]:
    key = _get_cache_key(query, n_results, "icd10")
    if key in _retrieval_cache:
        logger.info("Cache hit  [icd10] %.50s", query)
        return _retrieval_cache[key]
    results = search_icd10(query, n_results=n_results)
    _retrieval_cache[key] = results
    logger.info("Cache miss [icd10] %.50s", query)
    return results


def _cached_search_cpt4(query: str, n_results: int) -> list[dict]:
    key = _get_cache_key(query, n_results, "cpt4")
    if key in _retrieval_cache:
        logger.info("Cache hit  [cpt4] %.50s", query)
        return _retrieval_cache[key]
    results = search_cpt4(query, n_results=n_results)
    _retrieval_cache[key] = results
    logger.info("Cache miss [cpt4] %.50s", query)
    return results


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


# Keyword patterns used to build more specific CPT-4 retrieval queries.
# Tuples of (substring_to_match, expanded_search_terms).
_CPT4_QUERY_EXPANSIONS: list[tuple[str, str]] = [
    # Office / outpatient evaluation & management
    ("office visit",         "office outpatient visit evaluation management established"),
    ("outpatient visit",     "office outpatient visit evaluation management established"),
    ("established patient",  "office outpatient visit evaluation management established"),
    ("new patient",          "office outpatient visit evaluation management new patient"),
    ("office evaluation",    "office outpatient visit evaluation management established"),
    ("clinic visit",         "office outpatient visit evaluation management"),
    # Neurological / cognitive exams
    ("neurological exam",    "evaluation management office visit neurological examination"),
    ("neuro exam",           "evaluation management office visit neurological examination"),
    ("cognitive",            "evaluation management office visit cognitive assessment"),
    ("mental status",        "evaluation management office visit cognitive mental status"),
    # CT scans
    ("ct head",              "CT head brain without contrast"),
    ("ct scan head",         "CT head brain without contrast"),
    ("ct of head",           "CT head brain without contrast"),
    ("ct brain",             "CT head brain without contrast"),
    ("ct scan brain",        "CT head brain without contrast"),
    ("ct chest",             "CT thorax chest without contrast"),
    ("ct abdomen",           "CT abdomen pelvis without contrast"),
    ("ct pelvis",            "CT abdomen pelvis without contrast"),
    ("ct spine",             "CT spine cervical lumbar without contrast"),
    # MRI
    ("mri head",             "MRI brain head without contrast"),
    ("mri brain",            "MRI brain head without contrast"),
    ("mri spine",            "MRI spine cervical lumbar without contrast"),
    # Ultrasound / echo exams
    ("echo exam abdomen limited",   "echo exam abdomen limited ultrasound 76705"),
    ("echo exam abdomen",           "echo exam abdomen complete ultrasound 76700"),
    ("echo exam pelvis",            "echo exam pelvis ultrasound 76856"),
    ("echo exam renal",             "echo exam renal kidney ultrasound 76770"),
    ("echo exam thyroid",           "echo exam thyroid ultrasound 76536"),
    ("abdominal ultrasound",        "echo exam abdomen complete ultrasound 76700"),
    ("ultrasound abdomen",          "echo exam abdomen complete ultrasound 76700"),
    ("limited ultrasound",          "echo exam abdomen limited ultrasound 76705"),
    ("limited abdominal",           "echo exam abdomen limited ultrasound 76705"),
    ("single organ ultrasound",     "echo exam abdomen limited ultrasound 76705"),
    ("ultrasound pelvis",           "echo exam pelvis ultrasound 76856"),
    ("pelvic ultrasound",           "echo exam pelvis ultrasound 76856"),
    ("ultrasound renal",            "echo exam renal kidney ultrasound 76770"),
    ("kidney ultrasound",           "echo exam renal kidney ultrasound 76770"),
    ("ultrasound thyroid",          "echo exam thyroid ultrasound 76536"),
    ("ultrasound vascular",         "duplex scan vascular ultrasound 93971"),
    ("duplex scan",                 "duplex scan vascular ultrasound 93971"),
    ("ob ultrasound",               "ultrasound obstetric 76801"),
    ("obstetric ultrasound",        "ultrasound obstetric 76801"),
    ("echocardiogram",              "echocardiography transthoracic 93306"),
    ("cardiac echo",                "echocardiography transthoracic 93306"),
    ("transthoracic echo",          "echocardiography transthoracic 93306"),
    # Generic ultrasound fallback — after all specific patterns above
    ("ultrasound",                  "echo exam ultrasound diagnostic"),
    # X-rays
    ("chest x-ray",          "radiograph chest x-ray"),
    ("xray chest",           "radiograph chest x-ray"),
    # Lab / panels
    ("metabolic panel",      "comprehensive metabolic panel laboratory"),
    ("cmp",                  "comprehensive metabolic panel laboratory"),
    ("bmp",                  "basic metabolic panel laboratory"),
    ("cbc",                  "complete blood count laboratory"),
    ("blood count",          "complete blood count laboratory"),
    ("hemoglobin a1c",       "hemoglobin glycosylated A1c laboratory"),
    ("hba1c",                "hemoglobin glycosylated A1c laboratory"),
    ("a1c",                  "hemoglobin glycosylated A1c laboratory"),
    ("urinalysis",           "urinalysis laboratory"),
    # Emergency department visits (99281-99285 range, NOT critical care 99291)
    ("emergency department visit mild",              "emergency department visit evaluation management 99281"),
    ("emergency department visit low complexity",    "emergency department visit evaluation management 99282"),
    ("emergency department visit moderate",          "emergency department visit evaluation management 99283"),
    ("emergency department visit high complexity",   "emergency department visit evaluation management 99284"),
    ("emergency department visit severe",            "emergency department visit evaluation management 99285"),
    # Generic ED fallback — after the specific severity entries above
    ("emergency department visit",  "emergency department visit evaluation management"),
    ("emergency department",        "emergency department visit evaluation management"),
    ("emergency room",              "emergency department visit evaluation management"),
    # Critical care — only when explicitly documented
    ("critical care",               "critical care evaluation management 99291 99292"),
    ("critically ill",              "critical care evaluation management 99291 99292"),
    # Procedures
    ("nebulizer",            "nebulizer treatment respiratory"),
    ("ecg",                  "electrocardiogram ECG tracing interpretation"),
    ("ekg",                  "electrocardiogram ECG tracing interpretation"),
    ("spirometry",           "pulmonary function spirometry"),
    ("injection",            "therapeutic injection"),
]


def _build_cpt4_query(proc: dict[str, Any]) -> str:
    """
    Build a specific CPT-4 retrieval query from a procedure entity.

    Strategy (in priority order):
    1. Pattern-match the description against known CPT-4 query expansions so the
       HuggingFace embedding lands in the correct code cluster.
    2. Fall back to the generic description + source_text approach.

    For procedures, extracting the EXACT service type produces much better
    ChromaDB retrieval results:
      - 'Office visit established patient moderate complexity'
        → 'office outpatient visit evaluation management established'
      - 'CT scan of head'
        → 'CT head brain without contrast'
      - 'neurological examination'
        → 'evaluation management office visit neurological examination'
    """
    description = (proc.get("description", "") or "").strip()
    source_text  = (proc.get("source_text",  "") or "").strip()
    description_lower = description.lower()

    for substring, expanded in _CPT4_QUERY_EXPANSIONS:
        if substring in description_lower:
            # Append source snippet for additional context
            source_snippet = source_text[:200].strip()
            query = f"{expanded} {source_snippet}".strip()
            logger.debug("CPT-4 query expansion: '%s' → '%s'", description[:60], expanded)
            return query

    # No pattern matched — fall back to generic query
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
        # No per-item swallow: an embedding/vector-search failure (HF API outage,
        # rate limit, missing HF_TOKEN, timeout) is systemic, not specific to one
        # diagnosis. Let it propagate so the pipeline surfaces a real error instead
        # of silently producing empty candidates and a blank claim.
        candidates = _cached_search_icd10(query, _N_RETRIEVAL_RESULTS)
        for c in candidates:
            code = c.get("code", "")
            if code not in seen or c.get("score", 0) > seen[code].get("score", 0):
                seen[code] = c

    return list(seen.values())


def _retrieve_cpt4_candidates(entities: dict[str, Any]) -> list[dict]:
    """
    For each procedure in extracted_entities, collect CPT-4 candidates via one of two paths:

    Fast path — explicit CPT code:
      When the document_agent already captured a CPT code number in the procedure's "cpt_code"
      field (e.g. the plan section said "CPT 76705"), inject a synthetic candidate directly
      into the seen dict at score=1.0 and skip ChromaDB retrieval for that procedure.

    Standard path — ChromaDB semantic search:
      Build a specific query via _build_cpt4_query and run HuggingFace-embedded search.
    """
    procedures = entities.get("procedures", []) or []
    seen: dict[str, dict] = {}

    for proc in procedures:
        if not isinstance(proc, dict):
            continue

        explicit_code = (proc.get("cpt_code") or "").strip()
        description   = (proc.get("description", "") or "").strip()

        # --- Fast path: explicit CPT code bypasses ChromaDB ---
        if explicit_code:
            if explicit_code not in seen:
                seen[explicit_code] = {
                    "code":        explicit_code,
                    "description": description,
                    "category":    "explicit",
                    "score":       1.0,
                }
                logger.debug(
                    "CPT-4 fast path: explicit code %s for '%s' — skipping ChromaDB",
                    explicit_code, description[:60],
                )
            continue  # no ChromaDB search needed for this procedure

        # --- Standard path: semantic retrieval via HuggingFace embeddings ---
        query = _build_cpt4_query(proc)
        if not query:
            continue
        # No per-item swallow — see _retrieve_icd10_candidates: embedding/vector-search
        # failures are systemic and must surface, not degrade to empty candidates.
        candidates = _cached_search_cpt4(query, _N_RETRIEVAL_RESULTS)
        for c in candidates:
            code = c.get("code", "")
            if code not in seen or c.get("score", 0) > seen[code].get("score", 0):
                seen[code] = c

    return list(seen.values())


# ---------------------------------------------------------------------------
# Stage 2 — LLM Reranking (GPT-4o, reasoning only — no embeddings)
# ---------------------------------------------------------------------------

_ICD10_SYSTEM_PROMPT = """\
You are a certified medical coding specialist (CPC) with expertise in ICD-10 coding. \
Your task is to select the most accurate ICD-10 codes from the provided candidates \
based on the clinical documentation.

EXTERNAL CAUSE CODES — Required supplementary codes on all injury claims:
External cause codes (ICD-10 W/X/Y chapter) describe HOW, WHERE, and the ACTIVITY at the
time of injury. Always select them from the candidate list when the note documents an injury
with stated circumstances. Select one code per category when candidates are available.

HOW the injury occurred (mechanism):
- Falling object / struck by object     → W20.8XXA
- Unspecified fall                      → W19.XXXA
- Motor vehicle accident                → V-code from candidates
- Assault                               → X-code from candidates

WHERE it happened (Y92 place-of-occurrence codes):
- Home kitchen                          → Y92.010
- Home bedroom                          → Y92.011
- Home (unspecified)                    → Y92.009
- Workplace                             → Y92.2
- Street / highway                      → Y92.41
- School                                → Y92.21
- Sports / athletic area                → Y92.3

ACTIVITY at time of injury (Y93 codes):
- Cooking / baking                      → Y93.G3
- Sports / exercise                     → Y93 sport code from candidates
- Working / occupational                → Y93.89
- Leisure / recreation                  → Y93.E9

Selection rules for external cause codes:
- Only select external cause codes when the note EXPLICITLY documents an injury AND its circumstances.
- External cause codes are NEVER the primary diagnosis — always list them after the injury code.
- Do NOT select external cause codes for non-injury visits (chronic disease, routine follow-up).
- If no external cause candidate is in the list, do not invent one; note the gap in reasoning_chain.
- COMPLETENESS MANDATE: When injury circumstances are documented, you MUST assign all three
  external cause code types — mechanism (W/X/Y), place-of-occurrence (Y92), and activity (Y93) —
  if candidates for each are present. Never omit one type to stay under a code count limit.
- ALL secondary diagnoses (e.g. cervicalgia, headache) must be coded BEFORE applying any code
  count limit. Code the full clinical picture first, then trim only truly redundant codes.\
"""

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
      "reasoning": "authoritative explanation citing exact CMS guideline or clinical evidence supporting this code — never hedge if the code selection is guideline-compliant",
      "citation": "exact text from the clinical document supporting this code"
    }}
  ],
  "reasoning_chain": "overall explanation of coding decisions made"
}}

Chain-of-thought rules:
Step 1: Identify the primary diagnosis from the clinical entities
Step 2: Review each candidate code description carefully
Step 3: Match code specificity to documentation (use most specific code supported)
Step 4: Check if secondary diagnoses need separate codes. ALL secondary diagnoses
        (e.g. cervicalgia M542, headache, etc.) MUST be included before applying any code limit.
Step 5: EXTERNAL CAUSE CHECK — if the note documents an injury with circumstances:
        a) Select the injury mechanism code (W/X/Y) from candidates
        b) Select the place-of-occurrence code (Y92.xxx) from candidates if documented
        c) Select the activity code (Y93.xxx) from candidates if documented
        COMPLETENESS MANDATE: You MUST include all three types (a, b, c) when candidates
        are available — never omit Y93 activity or Y92 place codes to stay under the limit.
        External cause codes must follow the primary injury code in selected_codes order.
        Skip this step entirely for non-injury visits.

Coding confidence rules:
- When documentation states 'hypertension' without further specification, I10 (Essential hypertension) IS the correct and most specific code per CMS guidelines. State this confidently — never express uncertainty about I10 selection.
- When a diagnosis matches a candidate code exactly or near-exactly, assign confidence >= 0.90 and state the match clearly.
- Never say 'most likely' or 'given lack of specification' — if the code is correct per CMS guidelines, state it as fact.
- Only express uncertainty (confidence < 0.80) when the clinical documentation is genuinely ambiguous or incomplete.

Step 6: Assign confidence: >0.90 certain, 0.80-0.90 probable, <0.80 uncertain

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
based on the clinical documentation.

For procedure type identification, extract the EXACT service type from the documentation:
- "Office visit established patient moderate complexity"
  → search for and prefer: office outpatient visit established moderate (99213-99214)
- "CT scan of head"
  → search for and prefer: CT head brain without contrast (70450-70470)
- "neurological examination" or "neuro exam"
  → search for and prefer: evaluation management office visit (99202-99215)

Code selection rules:
- Always prefer E/M codes (99201-99215) for office visits over any procedure-specific code.
- Never select an unlisted procedure code (any code ending in 99, such as 99199, 93799, etc.) \
  when a specific code exists in the candidates list.
- When multiple E/M candidates exist, select the level that matches the documented complexity: \
  low=99202/99212, moderate=99203/99213/99214, high=99205/99215.
- When a specific imaging code exists (e.g. 70450 CT head), prefer it over a general radiology code.
- Always select imaging procedure codes separately from E/M codes. Never merge or omit an imaging
  procedure that is present in the candidate list.
- Ultrasound / echo exam codes: prefer 76700 (complete abdominal), 76705 (limited), 76856 (pelvis),
  76770 (renal), 76536 (thyroid) when the candidate list contains them.
- Candidates with category="explicit" were extracted directly from a CPT code stated in the note —
  always include them in selected_codes with confidence=1.0 and cite the source_text.

Emergency department complexity mapping:
- mild/minor                     → 99281
- low complexity                 → 99282
- moderate complexity            → 99283
- high complexity / urgent       → 99284
- severe / immediately life threatening → 99285

CRITICAL distinction — Emergency Department vs Critical Care:
- Only select 99291 (critical care, first hour) when the clinical note EXPLICITLY contains \
  the words "critical care" or "critically ill".
- A high-complexity or severe ED visit is 99284 or 99285, NOT 99291. \
  These are entirely different service types with different billing requirements.
- If the note says "high severity", "urgent", "immediately life threatening", or any similar \
  language WITHOUT the explicit phrase "critical care" or "critically ill", select 99284/99285, \
  never 99291."""

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
Step 2: Determine service type — is this an office visit/E&M, imaging, lab, or procedure?
Step 3: If E&M: prefer 99201-99215 range; match complexity level to documentation
Step 4: If imaging (CT/MRI/X-ray): select the most specific anatomic code available
Step 5: Review each candidate description carefully; eliminate unlisted codes (ending in 99)
        if any specific code is available in the candidate list
Step 6: Match code specificity to documentation (use most specific code supported)
Step 7: Check if additional procedures need separate codes
Step 8: Assign confidence: >0.90 certain, 0.80-0.90 probable, <0.80 uncertain

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


@traceable(name="coding_agent_rerank_icd10", process_inputs=_mask_rerank_trace_inputs)
async def _rerank_icd10(entities: dict[str, Any], candidates: list[dict], model: str) -> dict[str, Any]:
    """LLM reranking for ICD-10 candidates via AsyncOpenAI — no embeddings. Model chosen per request."""
    if not candidates:
        return {"selected_codes": [], "reasoning_chain": "No ICD-10 candidates to rerank."}

    user_content = _ICD10_USER_TEMPLATE.format(
        max_codes=_MAX_SELECTED_CODES,
        entities_json=json.dumps(entities, indent=2),
        candidates_text=_candidates_to_text(candidates, "disease"),
    )

    try:
        response = await _async_client.chat.completions.create(
            **_chat_completion_kwargs(model),
            messages=[
                {"role": "system", "content": _ICD10_SYSTEM_PROMPT},
                {"role": "user",   "content": user_content},
            ],
        )
        return _parse_llm_json(response.choices[0].message.content or "", "ICD-10")
    except Exception as e:
        logger.exception("ICD-10 reranking LLM call failed: %s", e)
        return {"selected_codes": [], "reasoning_chain": ""}


@traceable(name="coding_agent_rerank_cpt4", process_inputs=_mask_rerank_trace_inputs)
async def _rerank_cpt4(entities: dict[str, Any], candidates: list[dict], model: str) -> dict[str, Any]:
    """LLM reranking for CPT-4 candidates via AsyncOpenAI — no embeddings. Model chosen per request."""
    if not candidates:
        return {"selected_codes": [], "reasoning_chain": "No CPT-4 candidates to rerank."}

    user_content = _CPT4_USER_TEMPLATE.format(
        max_codes=_MAX_SELECTED_CODES,
        entities_json=json.dumps(entities, indent=2),
        candidates_text=_candidates_to_text(candidates, "description"),
    )

    try:
        response = await _async_client.chat.completions.create(
            **_chat_completion_kwargs(model),
            messages=[
                {"role": "system", "content": _CPT4_SYSTEM_PROMPT},
                {"role": "user",   "content": user_content},
            ],
        )
        return _parse_llm_json(response.choices[0].message.content or "", "CPT-4")
    except Exception as e:
        logger.exception("CPT-4 reranking LLM call failed: %s", e)
        return {"selected_codes": [], "reasoning_chain": ""}


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------


class RetrievalError(RuntimeError):
    """
    Raised when Stage 1 embedding / vector retrieval fails (e.g. HuggingFace
    Inference API outage, rate limit, timeout, or missing HF_TOKEN).

    This is surfaced rather than swallowed: without candidates the claim would be
    silently empty, so the request must fail with a clear error instead.
    """


async def coding_agent(state: ClaimState) -> ClaimState:
    """
    LangGraph node: 2-stage RAG + LLM reranking for ICD-10 and CPT-4 codes.

    Reads:  state['extracted_entities']
    Writes: state['icd10_candidates']     — raw semantic hits (Stage 1)
            state['cpt4_candidates']      — raw semantic hits (Stage 1)
            state['icd10_selected']       — GPT-4o reranked codes (Stage 2)
            state['cpt4_selected']        — GPT-4o reranked codes (Stage 2)
    """
    try:
        set_processing_stage(state.get("claim_id"), "coding")

        entities: dict[str, Any] = state.get("extracted_entities") or {}
        model = resolve_llm_model(state.get("llm_model"))

        # --- Stage 1: Semantic retrieval — both collections queried in parallel ----------
        # _retrieve_*_candidates are sync (blocking HF Inference API call + ChromaDB
        # query). asyncio.to_thread() offloads each blocking I/O call to the thread
        # pool so the two run concurrently.
        t1 = time.time()
        try:
            icd10_candidates, cpt4_candidates = await asyncio.gather(
                asyncio.to_thread(_retrieve_icd10_candidates, entities),
                asyncio.to_thread(_retrieve_cpt4_candidates, entities),
            )
        except Exception as e:
            # Embedding / vector-search failure is systemic (HF API down, rate
            # limited, no token). Do NOT degrade to an empty claim — raise so the
            # request fails with a clear, visible error.
            raise RetrievalError(f"Stage 1 retrieval failed (embedding/vector search): {e}") from e
        logger.info(
            "coding_agent Stage 1 (parallel retrieval): %d ICD-10, %d CPT-4 candidates — %.2fs",
            len(icd10_candidates),
            len(cpt4_candidates),
            time.time() - t1,
        )

        # --- Stage 2: LLM reranking — both GPT-4o calls run in parallel ----------------
        # _rerank_* are async and use await llm.ainvoke(), releasing the event loop
        # while waiting for HTTP responses, so asyncio.gather() genuinely overlaps them.
        t2 = time.time()
        icd10_result, cpt4_result = await asyncio.gather(
            _rerank_icd10(entities, icd10_candidates, model),
            _rerank_cpt4(entities, cpt4_candidates, model),
        )
        logger.info(
            "coding_agent Stage 2 (parallel reranking): %d ICD-10, %d CPT-4 selected — %.2fs",
            len(icd10_result.get("selected_codes", [])),
            len(cpt4_result.get("selected_codes", [])),
            time.time() - t2,
        )

        return {
            **state,
            "icd10_candidates": icd10_candidates,
            "cpt4_candidates": cpt4_candidates,
            "icd10_selected": icd10_result,
            "cpt4_selected": cpt4_result,
        }

    except RetrievalError:
        # Surface embedding/retrieval failures to the caller instead of returning
        # an empty (blank-claim) result.
        logger.exception("coding_agent: Stage 1 retrieval failed — surfacing error")
        raise
    except Exception as e:
        logger.exception("coding_agent failed: %s", e)
        return {
            **state,
            "icd10_candidates": state.get("icd10_candidates", []),
            "cpt4_candidates": state.get("cpt4_candidates", []),
            "icd10_selected": {"selected_codes": [], "reasoning_chain": ""},
            "cpt4_selected": {"selected_codes": [], "reasoning_chain": ""},
        }

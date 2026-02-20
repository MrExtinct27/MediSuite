"""
Document Agent — LangGraph node.
Extracts text from clinical documents and runs GPT-4o–based clinical entity extraction.
Uses HuggingFace/sentence-transformers for embeddings (not OpenAI). GPT-4o only for LLM reasoning.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langsmith import traceable

from orchestrator.state import ClaimState

logger = logging.getLogger(__name__)

# --- Step 1: Document extraction ---


def extract_text_from_file(file_path: str) -> str:
    """
    Extract and return cleaned, normalized text from a document file.
    Supports .pdf (including scanned/OCR), .docx, and .txt.
    """
    path = Path(file_path)
    if not path.exists():
        logger.warning("File not found: %s", file_path)
        return ""

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _extract_text_pdf(str(path))
    if suffix == ".docx":
        return _extract_text_docx(str(path))
    if suffix == ".txt":
        return _extract_text_plain(str(path))

    logger.warning("Unsupported file type: %s", suffix)
    return ""


def _extract_text_pdf(file_path: str) -> str:
    """Extract text from PDF; use PyMuPDF first, fall back to pdf2image + pytesseract for scanned PDFs."""
    try:
        doc = fitz.open(file_path)
        parts = []
        for page in doc:
            parts.append(page.get_text())
        doc.close()
        text = "\n".join(parts).strip()
        if not text or len(text) < 50:
            text = _extract_text_pdf_ocr(file_path)
        return _normalize_text(text)
    except Exception as e:
        logger.exception("PDF extraction failed: %s", e)
        try:
            return _normalize_text(_extract_text_pdf_ocr(file_path))
        except Exception as e2:
            logger.exception("PDF OCR fallback failed: %s", e2)
            return ""


def _extract_text_pdf_ocr(file_path: str) -> str:
    """Scanned PDF: render pages to images and run pytesseract OCR."""
    import pdf2image
    import pytesseract

    images = pdf2image.convert_from_path(file_path)
    parts = []
    for img in images:
        parts.append(pytesseract.image_to_string(img))
    return "\n".join(parts)


def _extract_text_docx(file_path: str) -> str:
    """Extract paragraphs from .docx using python-docx."""
    try:
        from docx import Document
        doc = Document(file_path)
        parts = [p.text for p in doc.paragraphs if p.text.strip()]
        return _normalize_text("\n".join(parts))
    except Exception as e:
        logger.exception("DOCX extraction failed: %s", e)
        return ""


def _extract_text_plain(file_path: str) -> str:
    """Read .txt file with utf-8, fallback to latin-1."""
    try:
        with open(file_path, encoding="utf-8") as f:
            return _normalize_text(f.read())
    except UnicodeDecodeError:
        with open(file_path, encoding="latin-1") as f:
            return _normalize_text(f.read())
    except Exception as e:
        logger.exception("TXT read failed: %s", e)
        return ""


def _normalize_text(text: str) -> str:
    """Clean and normalize extracted text."""
    if not text:
        return ""
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# --- Step 2: Clinical entity extraction (GPT-4o, langchain-openai) ---

EXTRACTION_SYSTEM_PROMPT = """You are a clinical information extraction specialist. Extract structured medical information from the clinical document. Return ONLY valid JSON."""

EXTRACTION_USER_TEMPLATE = """Extract the following fields from the document below. Return a single JSON object with exactly these keys (use null for missing values).

- "patient_name": full name or null
- "patient_dob": DOB in YYYY-MM-DD format or null
- "patient_insurance_id": insurance ID or null
- "diagnoses": array of objects, each with "description" (exact diagnosis text), "confidence" (0.0-1.0 float), "source_text" (verbatim excerpt)
- "procedures": array of objects, each with "description", "confidence", "source_text"
- "medications": list of medication strings mentioned
- "dates": object with "service_date", "admission_date", "discharge_date" (YYYY-MM-DD or null each)
- "provider_name": physician/provider name or null
- "facility_name": hospital/clinic name or null

Rules: Only extract what is explicitly stated. Set confidence < 0.7 for ambiguous extractions. Include source_text for every diagnosis and procedure. Return null for missing fields; never guess.

Document:
---
{text}
---
JSON only, no markdown:"""


@traceable(name="document_agent_extract_entities")
def _extract_entities_with_llm(text: str) -> dict[str, Any] | None:
    """Call GPT-4o via langchain-openai; return parsed JSON or None."""
    if not text or not text.strip():
        return _empty_entities()

    try:
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        messages = [
            SystemMessage(content=EXTRACTION_SYSTEM_PROMPT),
            HumanMessage(content=EXTRACTION_USER_TEMPLATE.format(text=text[:120_000])),
        ]
        response = llm.invoke(messages)
        content = response.content if hasattr(response, "content") else str(response)
        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```\w*\n?", "", content)
            content = re.sub(r"\n?```\s*$", "", content)
        return json.loads(content)
    except json.JSONDecodeError as e:
        logger.warning("LLM response was not valid JSON: %s", e)
        return _empty_entities()
    except Exception as e:
        logger.exception("LLM extraction failed: %s", e)
        return _empty_entities()


def _empty_entities() -> dict[str, Any]:
    return {
        "patient_name": None,
        "patient_dob": None,
        "patient_insurance_id": None,
        "diagnoses": [],
        "procedures": [],
        "medications": [],
        "dates": {"service_date": None, "admission_date": None, "discharge_date": None},
        "provider_name": None,
        "facility_name": None,
    }


# --- Step 3: Confidence scoring ---


def _apply_confidence_scoring(entities: dict[str, Any]) -> dict[str, Any]:
    """
    Enforce confidence rules: mark or downweight items below threshold.
    Keeps structure; low-confidence diagnoses/procedures remain but can be flagged.
    """
    # Optional: filter or flag items with confidence < 0.7
    for key in ("diagnoses", "procedures"):
        if key not in entities or not isinstance(entities[key], list):
            continue
        for item in entities[key]:
            if isinstance(item, dict) and "confidence" in item:
                c = item["confidence"]
                if not isinstance(c, (int, float)):
                    item["confidence"] = 0.5
                elif c > 1.0:
                    item["confidence"] = 1.0
                elif c < 0.0:
                    item["confidence"] = 0.0
    return entities


# --- LangGraph node ---


def document_agent(state: ClaimState) -> ClaimState:
    """
    LangGraph node: document extraction → GPT-4o entity extraction → confidence scoring.
    Reads state["raw_document_text"] or state["file_path"]; writes back raw_document_text and extracted_entities.
    """
    try:
        # Resolve input text
        raw_text = state.get("raw_document_text") or ""
        file_path = state.get("file_path")
        if isinstance(file_path, str) and (not raw_text or not raw_text.strip()):
            raw_text = extract_text_from_file(file_path)

        if not raw_text or not raw_text.strip():
            return {
                **state,
                "raw_document_text": raw_text,
                "extracted_entities": _empty_entities(),
            }

        # GPT-4o extraction (reasoning only; no embeddings)
        entities = _extract_entities_with_llm(raw_text)
        if entities is None:
            entities = _empty_entities()

        entities = _apply_confidence_scoring(entities)

        # Populate state fields used by downstream agents
        updates: ClaimState = {
            "raw_document_text": raw_text,
            "extracted_entities": entities,
        }
        if entities.get("patient_name") is not None:
            updates["patient_name"] = str(entities["patient_name"])
        if entities.get("patient_dob") is not None:
            updates["patient_dob"] = str(entities["patient_dob"])
        if entities.get("patient_insurance_id") is not None:
            updates["patient_insurance_id"] = str(entities["patient_insurance_id"])

        return {**state, **updates}
    except Exception as e:
        logger.exception("document_agent failed: %s", e)
        return {
            **state,
            "raw_document_text": state.get("raw_document_text", ""),
            "extracted_entities": _empty_entities(),
        }

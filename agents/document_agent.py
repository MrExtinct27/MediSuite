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

EXTRACTION_SYSTEM_PROMPT = """\
You are a clinical information extraction specialist. \
Extract structured medical information from the clinical document. Return ONLY valid JSON.

CRITICAL RULE — Evaluation & Management (E/M) Codes:
Every clinical note MUST be checked for an E/M service.
E/M services are ALWAYS billable and must ALWAYS be included as a procedure even if not \
explicitly stated in the note — they represent the visit itself.

E/M detection rules (apply in order, add a procedure entry for each match):
- If the note mentions "new patient" or the patient has never been seen before
  → add procedure: description="office outpatient visit new patient", source_text=<relevant excerpt>
- If the note mentions "established patient" or "follow-up" or "follow up"
  → add procedure: description="office outpatient visit established patient", source_text=<relevant excerpt>
- If the note mentions "emergency department", "ER", or "emergency room"
  → add procedure: description="emergency department visit", source_text=<relevant excerpt>
- If no visit type is specified but there is clearly a physician-patient encounter,
  default to: description="office outpatient visit established patient"

E/M complexity mapping (set on the procedure entry's description):
- "low complexity" or "straightforward" → append "low complexity" (maps to 99202/99212)
- "moderate complexity"                → append "moderate complexity" (maps to 99204/99213/99214)
- "high complexity"                    → append "high complexity" (maps to 99205/99215)
- Emergency department severity mapping (use EXACT labels below):
    mild/minor        → "emergency department visit mild" (99281)
    low complexity    → "emergency department visit low complexity" (99282)
    moderate          → "emergency department visit moderate complexity" (99283)
    high/urgent       → "emergency department visit high complexity" (99284)
    severe/life threatening → "emergency department visit severe" (99285)
  IMPORTANT: Only use the label "critical care" when the note EXPLICITLY contains the words
  "critical care" or "critically ill". A high-complexity or severe ED visit is 99284/99285,
  NOT critical care (99291). These are completely different service types.

NEVER skip E/M code extraction. It is required on every single claim.

EXTERNAL CAUSE CODES — Always check for these on injury claims:
External cause codes (ICD-10 W/X/Y chapter) describe HOW, WHERE, and WHAT ACTIVITY the
patient was performing when injured. They are required supplementary codes whenever the note
documents an injury with stated circumstances. Extract them as additional diagnosis entries.

HOW the injury occurred (mechanism — W/X/Y codes):
- Struck by / hit by falling object            → description="struck by falling object injury" (W20.8XXA)
- Fall (unspecified / from same level)          → description="fall unspecified injury" (W19.XXXA)
- Motor vehicle accident / MVA / collision      → description="motor vehicle accident injury" (V codes)
- Assault / struck by another person           → description="assault injury" (X codes)
- Burn / fire / hot substance                  → description="burn fire injury" (X00–X19)
- Poisoning / drug overdose                    → description="poisoning drug substance" (X40–X49)

WHERE it happened (place of occurrence — Y92 codes):
- Home kitchen                                 → description="place of occurrence home kitchen" (Y92.010)
- Home bedroom / sleeping area                 → description="place of occurrence home bedroom" (Y92.011)
- Home (unspecified room)                      → description="place of occurrence home" (Y92.009)
- Workplace / office                           → description="place of occurrence workplace" (Y92.2)
- Street or highway                            → description="place of occurrence street highway" (Y92.41)
- School                                       → description="place of occurrence school" (Y92.21)
- Sports / athletic area                       → description="place of occurrence sports area" (Y92.3)

ACTIVITY at time of injury (Y93 codes):
- Cooking / baking / food preparation          → description="activity cooking baking" (Y93.G3)
- Sports / exercise (specify sport if known)   → description="activity sports exercise" (Y93 sport codes)
- Working / occupational activity              → description="activity working occupational" (Y93.89)
- Leisure / recreation                         → description="activity leisure recreation" (Y93.E9)

Rules for external cause extraction:
- Only add external cause entries when the note EXPLICITLY describes an injury AND its circumstances.
- Add one entry per category (mechanism, place, activity) when documented.
- Set confidence = 0.85 when circumstances are clear; 0.70 when implied but not explicit.
- Do NOT add external cause entries for non-injury visits (chronic disease management, routine follow-up, etc.).

Imaging rule — combined studies:
When a CT scan or MRI covers multiple body regions (e.g. "abdomen AND pelvis", \
"chest/abdomen/pelvis"), extract it as ONE combined study, not as separate region entries.
Example: "CT abdomen and pelvis" → one procedure entry: \
description="CT abdomen pelvis combined without contrast".

IMAGING AND DIAGNOSTIC PROCEDURE RULES:
Always extract imaging procedures as their own separate entries, distinct from any surgical or
E/M procedures. Never omit an imaging procedure that is explicitly listed in the plan section.

Ultrasound / echo exam mapping (use EXACT description text below for best ChromaDB retrieval):
- "ultrasound abdomen" / "abdominal ultrasound" / "echo exam abdomen"
  → description="echo exam abdomen" (maps to CPT 76700)
- "limited ultrasound" / "single organ ultrasound" / "limited abdominal ultrasound"
  → description="echo exam abdomen limited" (maps to CPT 76705)
- "ultrasound pelvis" / "pelvic ultrasound"
  → description="echo exam pelvis" (maps to CPT 76856/76857)
- "ultrasound obstetric" / "OB ultrasound"
  → description="ultrasound obstetric" (maps to CPT 76801/76805)
- "ultrasound renal" / "kidney ultrasound"
  → description="echo exam renal kidney" (maps to CPT 76770)
- "ultrasound thyroid"
  → description="echo exam thyroid" (maps to CPT 76536)
- "ultrasound vascular" / "duplex scan"
  → description="duplex scan vascular ultrasound" (maps to CPT 93971)
- "echocardiogram" / "cardiac echo" / "transthoracic echo"
  → description="echocardiography transthoracic" (maps to CPT 93306)

Explicit CPT code rule:
If the plan section or procedure note lists a CPT code number explicitly \
(e.g. "CPT 76705" or "ordered: 76705"), capture it in a "cpt_code" field on the procedure \
entry so it can bypass ChromaDB retrieval entirely:
  {{ "description": "limited ultrasound abdomen", "cpt_code": "76705", \
     "confidence": 1.0, "source_text": "..." }}

IMPLIED PROCEDURE RULES — Infer these even when not explicitly stated:
Apply these rules after processing all explicit content. They govern procedures that are
clinically necessary and billable but may be described only by context or implication.

Rule 1 — Every clinical encounter has an E/M code:
Already covered by the CRITICAL RULE above. Reconfirm here: if no E/M procedure entry
has been added yet, add one now. Determine patient type (new vs established), setting
(office / ED / inpatient), and complexity from any contextual clues in the note.
Confidence = 0.90 when setting and patient type are clear; 0.75 when inferred from context.

Rule 2 — Every surgery implies anesthesia:
If the note documents any surgical procedure (appendectomy, laparoscopy, colectomy, hernia
repair, joint replacement, etc.) AND mentions "general anesthesia", "regional anesthesia",
"MAC", or "sedation", add a separate procedure entry for it:
  - "general anesthesia" → description="anesthesia general surgical procedure"
  - "regional anesthesia" / "spinal" / "epidural" → description="anesthesia regional"
  - "MAC" / "monitored anesthesia care" / "IV sedation" → description="anesthesia MAC sedation"
Set confidence = 0.95 when anesthesia type is documented; 0.70 when only the surgery is
documented (anesthesia is implied but not named — do NOT add entry if type is unknown).

Rule 3 — Diagnostic confirmation implies the confirming test:
If the note uses language that implies a diagnostic test produced the finding, extract that
test as a procedure even if it is only referenced in passing:
  - "ultrasound showed / revealed / demonstrated" → extract ultrasound using imaging rules above
  - "CT showed / CT revealed / CT demonstrated"   → extract CT scan using imaging rules above
  - "MRI showed / MRI revealed"                   → extract MRI using imaging rules above
  - "labs showed / bloodwork confirmed"            → extract the specific lab (CBC, CMP, etc.)
  - "X-ray showed"                                → extract radiograph
  - "biopsy confirmed" / "pathology showed"        → extract biopsy/pathology procedure
  - "EKG/ECG showed"                              → description="electrocardiogram ECG tracing"
Use the source phrase (e.g. "CT revealed appendicitis") verbatim as source_text.
Set confidence = 0.90 when the modality is unambiguous from context.

Rule 4 — Injury notes imply external cause codes (W/X/Y):
Already covered by the EXTERNAL CAUSE CODES section above. Reconfirm here: whenever the
note describes an injury with any circumstantial detail (mechanism, location, activity),
those W/X/Y codes MUST be extracted as diagnosis entries even if they were not written
explicitly by the provider. Apply the HOW / WHERE / ACTIVITY lookup tables above.

Implied procedure confidence scale:
- 1.00 — procedure explicitly named with CPT code in the note
- 0.95 — procedure clearly documented (e.g. "general anesthesia administered")
- 0.90 — strongly implied by a specific phrase ("CT revealed", "ultrasound showed")
- 0.75 — inferred from clinical context (E/M type from visit setting)
- 0.70 — indirectly implied; document the inference in source_text\
"""

EXTRACTION_USER_TEMPLATE = """\
Extract the following fields from the document below. \
Return a single JSON object with exactly these keys (use null for missing values).

- "patient_name": full name or null
- "patient_dob": DOB in YYYY-MM-DD format or null
- "patient_insurance_id": insurance ID or null
- "diagnoses": array of objects, each with:
    "description" (exact diagnosis text), "confidence" (0.0-1.0 float), "source_text" (verbatim excerpt)
- "procedures": array of objects, each with "description", "confidence", "source_text",
    and an OPTIONAL "cpt_code" field (string) when an explicit CPT number is stated in the note.
  IMPORTANT — always include an E/M visit procedure per the system prompt rules.
  For each procedure, extract the EXACT service type so it maps cleanly to a CPT code:
    · "Office visit established patient moderate complexity" (not just "office visit")
    · "CT head brain without contrast" (not just "CT scan")
    · "echo exam abdomen" for abdominal ultrasound (not just "ultrasound")
    · "neurological examination" → use "office outpatient visit established patient"
  Extract imaging procedures as separate entries from E/M and surgical procedures.
  Never skip an imaging procedure that appears in the plan section.
  When a CT/MRI covers multiple regions, combine them into ONE entry (see system prompt).
  If the note explicitly states a CPT code number (e.g. "76705"), set "cpt_code": "76705"
  on that procedure entry — do NOT omit procedures that have explicit CPT codes.
- "medications": list of medication strings mentioned
- "dates": object with "service_date", "admission_date", "discharge_date" (YYYY-MM-DD or null each)
- "provider_name": physician/provider name or null
- "facility_name": hospital/clinic name or null

Rules: Only extract what is explicitly stated. Set confidence < 0.7 for ambiguous extractions. \
Include source_text for every diagnosis and procedure. Return null for missing fields; never guess. \
The E/M visit procedure is the one exception — always include it even when not explicitly listed.

Document:
---
{text}
---
JSON only, no markdown:\
"""


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

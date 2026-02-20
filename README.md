# MediSuite AI Agent
### Agentic AI for Automated Medical Claims and Clinical Workflow Assistance
> Master's Project — CPSC 597/598 | California State University, Fullerton
> Student: Yash Pankaj Mahajan | Advisor: Dr. Kenneth Kung

---

## Overview
MediSuite AI Agent is a multi-agent AI system that automates the
end-to-end medical claims processing workflow. It uses LangGraph
to orchestrate four specialized AI agents that collaborate to extract
clinical information, assign medical codes, validate claims, and
generate CMS-1500 claim forms — with full explainability and
LangSmith observability.

---

## Architecture
Four LangGraph agents connected in a stateful pipeline:

```
Document Agent → Coding Agent → Validation Agent → Claim Agent
                      ↑               |
                      |_______________| (re-routes on validation failure)
```

- **Document Agent:** Extracts clinical entities from PDF/DOCX/TXT
  using GPT-4o with confidence scoring
- **Coding Agent:** 2-stage RAG pipeline — ChromaDB semantic search
  (HuggingFace) → GPT-4o reranking with chain-of-thought reasoning
- **Validation Agent:** 3-level validation (rules + LLM + anomaly detection)
  with conditional LangGraph routing
- **Claim Agent:** Generates structured CMS-1500 claim JSON with
  full explainability trails

---

## Tech Stack

### AI / Agent Layer
| Component | Role |
|-----------|------|
| LangGraph | Stateful multi-agent orchestration |
| GPT-4o | Clinical entity extraction + code reranking |
| ChromaDB | Vector database for ICD-10/CPT-4 semantic search |
| HuggingFace (`pritamdeka/S-PubMedBert-MS-MARCO`) | Biomedical embeddings |
| LangSmith | Agent observability and tracing |

### Backend
| Component | Role |
|-----------|------|
| FastAPI | REST API |
| SQLAlchemy + SQLite | Claim storage and audit logging |
| python-dotenv | Environment management |

### Document Processing
| Component | Role |
|-----------|------|
| PyMuPDF | PDF text extraction |
| Tesseract OCR | Scanned document support |
| python-docx | DOCX support |

---

## Project Structure

```
medisuite/
├── agents/
│   ├── document_agent.py      # Clinical NER + extraction
│   ├── coding_agent.py        # RAG + GPT-4o reranking
│   ├── validation_agent.py    # 3-level validation
│   └── claim_agent.py         # CMS-1500 generation
├── orchestrator/
│   ├── graph.py               # LangGraph StateGraph
│   ├── state.py               # ClaimState TypedDict
│   └── router.py              # Conditional edge logic
├── api/
│   ├── main.py                # FastAPI app
│   └── routes/claims.py       # API endpoints
├── db/
│   ├── models.py              # SQLAlchemy models
│   └── database.py            # DB connection
├── knowledge_base/
│   ├── embeddings.py          # ChromaDB + HF embeddings
│   ├── ICD10.json             # 70,000+ diagnosis codes
│   └── CPT4.json              # 10,000+ procedure codes
└── tests/
    └── sample_note.txt        # Test clinical note
```

---

## Phase 1 — Complete ✅
Core backend pipeline fully working end-to-end.

**What was built:**
- `ClaimState` TypedDict flowing through all 4 LangGraph agents
- ChromaDB populated with ICD-10 and CPT-4 code embeddings
  using `pritamdeka/S-PubMedBert-MS-MARCO` biomedical model
- Document Agent extracting structured clinical entities via GPT-4o
- Coding Agent: 2-stage RAG retrieval (top-20 candidates) +
  GPT-4o reranking with chain-of-thought and confidence scoring
- Validation Agent with 3 levels:
  - Level 1: CMS rule-based checks
  - Level 2: GPT-4o clinical logic validation
  - Level 3: Code co-occurrence anomaly detection
- Conditional LangGraph routing (re-codes if validation fails,
  routes to human review if confidence < 80%)
- Claim Agent generating CMS-1500 JSON with full explainability
- FastAPI with `/claims/process`, `/claims/{id}/status`,
  `/claims/{id}/explanation`, `/health` endpoints
- SQLite audit logging for every agent action
- LangSmith tracing on all LLM calls

**Sample output for a Type 2 Diabetes patient:**
```
ICD-10: E1121 (95%), I10 (90%), N18.3 (95%)
CPT-4:  99214 (90%), 80053 (89%), 83036 (91%)
Processing time: ~15 seconds end-to-end
```

---

## Phase 2 — In Progress 🔄
- Human-in-the-loop review queue
- Feedback loop from rejected claims
- HIPAA compliance + AES encryption
- A/B benchmarking (with RAG vs without RAG)
- Performance evaluation and thesis metrics

## Phase 3 — Planned 📋
- React + TypeScript frontend
- Claims dashboard with explainability UI
- Analytics with Recharts
- Docker deployment

---

## Setup & Installation

### Prerequisites
- Python 3.11+
- OpenAI API key
- LangSmith API key

### Installation
```bash
git clone https://github.com/MrExtinct27/MediSuite-Ai-Agent.git
cd MediSuite-Ai-Agent
python3.11 -m venv medisuite-env
source medisuite-env/bin/activate
pip install -r requirements.txt
```

### Configuration
```bash
cp .env.example .env
# Fill in your API keys in .env
```

### Run ChromaDB Embedding (first time only)
```bash
python -m knowledge_base.embeddings
```

### Initialize Database
```bash
python -m db.database
```

### Start the Server
```bash
python -m uvicorn api.main:app --reload --port 8000
```

### Test the Pipeline
```bash
curl -X POST http://localhost:8000/claims/process \
  -F "file=@tests/sample_note.txt" \
  -F "patient_name=John Smith" \
  -F "insurance_id=BCB123456"
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/claims/process` | Submit clinical document for processing |
| `GET` | `/claims/{id}/status` | Get claim processing status |
| `GET` | `/claims/{id}/explanation` | Get full reasoning trail |
| `GET` | `/claims/{id}/result` | Get complete claim JSON |
| `GET` | `/health` | System health check |

---

## Important Notes

### First Run
The HuggingFace biomedical embedding model (~438 MB) downloads
automatically on first run and caches locally.
Subsequent runs load from cache instantly.

### Never Commit
```
.env              # API keys
chroma_data/      # embeddings — regenerate with embeddings.py
medisuite-env/    # venv
*.db              # database files
claims/           # generated outputs
```

---

## Acknowledgements
- Original MediSuite concept: [Ali-Afifi/MediSuite-Ai-Agent](https://github.com/Ali-Afifi)
- Built upon and significantly extended for CSUF Master's Project
- LangGraph, LangSmith by [LangChain](https://www.langchain.com/)
- Biomedical embeddings: [pritamdeka/S-PubMedBert-MS-MARCO](https://huggingface.co/pritamdeka/S-PubMedBert-MS-MARCO)

---

## License
MIT License — see [LICENSE](LICENSE) file

# OncoBot: Agentic Cancer Research Assistant

OncoBot is an agentic AI system for oncology — it assists cancer researchers, clinicians, and patients with oncology inquiries using a **single-agent LangGraph architecture** with 8 tools, semantic caching, and conversational memory.

Deployed on AWS EC2 via GitHub Actions CI/CD.

---

## Architecture

```
User query → Cache check (ChromaDB cosine ≥ 0.92) → Cache hit? → Return instantly
                                    │
                               Cache miss
                                    │
         ┌────────── AGENT (gemini-3.1-flash-lite-preview) ──────────┐
         │  Single LangGraph node, picks tools from docstrings.       │
         │  ReAct loop: up to 5 tool iterations per turn.             │
         │                                                            │
         │  RAG & Search:                                             │
         │  ├── oncology_rag_search       (ChromaDB vector store)     │
         │  ├── fetch_pubmed_abstracts    (NCBI E-utilities API)      │
         │  ├── search_clinical_trials    (ClinicalTrials.gov v2)     │
         │  └── summarize_arxiv_paper     (arXiv API)                 │
         │                                                            │
         │  Vision:                                                    │
         │  └── analyze_medical_image     (Gemini Vision)             │
         │                                                            │
         │  ML Classification:                                        │
         │  ├── classify_breast_ultrasound (OncoScanBC, MobileNetV2)  │
         │  ├── classify_skin_lesion      (OncoScanSkin, MobileNetV2) │
         │  └── classify_cancer_type      (OncoTypeBC, MLP)           │
         └───────────────────────────────────────────────────────────┘
                                    │
                        Store response in cache
```

2–3 API calls per query. At 500 RPD free tier → ~170–250 queries/day.

### Data flow for ML tools

Uploaded files (images, CSVs) are stored in **session-level shared state** inside
`tools/onco_tools.py`. Tools read directly from this state — the LLM never needs
to pass raw binary data through tool arguments. The flow:

```
User upload → app.py reads bytes
            → onco_agent.py calls set_session_image() / set_session_csv()
            → LLM decides which tool to call (no data args needed)
            → Tool reads from _session_image_b64 / _session_genomic_csv
            → clear_session_data() after agent finishes
```

For images, the base64 is also sent as a proper multimodal content part
in the HumanMessage, so Gemini can visually see it and decide which
classifier to use.

---

## Key Features

- **Agentic Reasoning**: The LLM decides which tools to call based on the query. Factual questions trigger RAG; "latest research" triggers PubMed; trial queries trigger ClinicalTrials.gov.
- **Semantic Cache**: ChromaDB-backed response cache (cosine ≥ 0.92). Paraphrases resolve to the same entry. No API quota consumed on cache hits.
- **Rate-limit Handling**: Auto-retries on 429 errors with API-suggested delays, 60s cap.
- **Conversational Memory**: MemorySaver checkpointing per session thread. Follow-up questions work without repeating context.
- **Live External Data**: ClinicalTrials.gov, PubMed, arXiv — all free, no API keys needed.
- **Reasoning Transparency**: The UI shows every tool call, arguments, and observations.
- **Multilingual**: `paraphrase-multilingual-MiniLM-L12-v2` embeddings match any language against the English knowledge base.
- **Multimodal**: Image upload → Gemini Vision analysis or CNN classification.
- **ML Classification**: OncoScanBC (breast ultrasound, 3 classes), OncoScanSkin (dermoscopy, 7 classes), OncoTypeBC (gene expression → 5 TCGA cancer types).
- **Auto-updating Knowledge Base**: APScheduler re-indexes `knowledge_base/` every 30 minutes. Uses SHA-256 content hashing to detect new or modified files — no duplicates across container rebuilds.
- **LLM-as-Judge Evaluation**: Measures faithfulness and answer relevancy using Gemini as a judge.

---

## ML Models

| Model | Task | Architecture | Classes |
|---|---|---|---|
| OncoScanBC | Breast ultrasound classification | MobileNetV2 | benign, malignant, normal |
| OncoScanSkin | Skin lesion classification | MobileNetV2 | 7 HAM10000 classes |
| OncoTypeBC | Cancer type from gene expression | Custom MLP (20531→512→128→5) | BRCA, KIRC, LUAD, PRAD, COAD |

Model weights are stored in `models/` and loaded lazily on first use.
OncoTypeBC also requires `scaler.pkl` (StandardScaler) and `label_ecoder.pkl` (LabelEncoder).

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Google Gemini 3.1 Flash Lite Preview (500 req/day free tier) |
| Agent Framework | LangGraph 1.1.2 (single-agent, StateGraph) |
| Orchestration | LangChain 1.2.12 |
| Embeddings | HuggingFace `paraphrase-multilingual-MiniLM-L12-v2` |
| Vector DB | ChromaDB 1.5.5 (RAG collection + response cache collection) |
| ML Models | PyTorch 2.6.0 CPU (MobileNetV2, custom MLP) |
| External APIs | ClinicalTrials.gov v2, NCBI PubMed E-utilities, arXiv |
| App Framework | Streamlit 1.55.0 |
| Containerization | Docker (python:3.11-slim, non-root), Docker Compose |
| Cloud | AWS EC2 (t3.medium), AWS ECR |
| CI/CD | GitHub Actions (OIDC auth, Trivy vulnerability scanning) |

---

## Project Structure

```
gemini_rag_chatbot/
├── .github/workflows/deploy.yml
├── agent/
│   ├── __init__.py
│   ├── cache.py           # Semantic query-response cache (cosine ≥ 0.92)
│   ├── memory.py          # MemorySaver checkpointer
│   └── onco_agent.py      # Single-agent LangGraph graph + streaming
├── tools/
│   ├── __init__.py
│   ├── onco_tools.py      # RAG, vision, ML classifiers, session state
│   └── external_tools.py  # ClinicalTrials.gov, PubMed, arXiv
├── evaluation/
│   └── ragas_eval.py      # LLM-as-judge faithfulness/relevancy evaluation
├── knowledge_base/        # MedQuAD XML question-answer pairs
├── models/                # PyTorch model weights (.pth, .pkl)
├── app.py                 # Streamlit UI (centered layout, chat-focused)
├── updater.py             # Incremental knowledge base indexer
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## Local Development

```bash
git clone https://github.com/vyshnavnp/gemini-rag-chatbot.git
cd gemini-rag-chatbot
python -m venv .venv && .venv\Scripts\activate

pip install -r requirements.txt

mkdir .streamlit
echo 'GEMINI_API_KEY = "your-key-here"' > .streamlit/secrets.toml

python updater.py        # Build knowledge base (first time)
streamlit run app.py
```

Model weights are included in the repo (~60 MB). No extra download step needed.

---

## EC2 Deployment

GitHub Actions (`.github/workflows/deploy.yml`) builds, pushes to ECR, and deploys on every push to `main`.

### CI/CD Pipeline

```
push to main → Build image → Trivy scan (CRITICAL) → Push to ECR → Deploy to EC2
```

- **OIDC authentication** — no static AWS keys stored in GitHub. The runner assumes an IAM role via short-lived tokens.
- **Trivy vulnerability scan** — blocks the push if any CRITICAL CVE is found in the image.
- **Non-root container** — runs as `appuser` (UID 1000), with `no-new-privileges` and `cap_drop: ALL`.
- **EC2 Instance Role** — the EC2 instance uses an IAM instance role for ECR pull and EBS management (no keys on the server).

### Required GitHub Secrets

| Secret | Purpose |
|---|---|
| `AWS_ROLE_ARN` | IAM role ARN for OIDC (replaces static keys) |
| `AWS_REGION` | AWS region |
| `ECR_REGISTRY` | ECR registry URI |
| `ECR_REPOSITORY` | ECR repository name |
| `EC2_HOST` | EC2 public IP/hostname |
| `EC2_SSH_KEY` | SSH private key for EC2 |
| `GEMINI_API_KEY` | Google Gemini API key |

First deployment: `docker exec -it oncobot-container python updater.py`

### Security Hardening

- **Dockerfile**: Non-root `appuser` (UID 1000), build dependencies purged after install
- **Docker Compose**: `security_opt: no-new-privileges`, `cap_drop: ALL`
- **CI/CD**: OIDC (no long-lived AWS credentials), Trivy image scanning
- **EC2**: IAM instance role (no static keys on server), EBS auto-expansion
- **Dependencies**: All packages pinned to exact versions, PyTorch CPU-only build

---

## API Quota

The project uses `gemini-3.1-flash-lite-preview` (500 RPD free tier). The semantic cache serves repeated and paraphrased queries without consuming quota. On exhaustion, the agent retries with the API-suggested delay before showing a user-friendly error.

---

## Disclaimer

This tool is for educational and research purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified oncologist.

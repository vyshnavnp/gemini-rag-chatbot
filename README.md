# OncoBot: Agentic Cancer Research Assistant

OncoBot is an agentic AI system for oncology — it assists cancer researchers, clinicians, and patients with oncology inquiries using a **single-agent LangGraph architecture** with 9 tools, semantic caching, and conversational memory.

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
         │  Vision & Diagrams:                                        │
         │  ├── analyze_medical_image     (Gemini Vision)             │
         │  └── generate_pathway_diagram  (Gemini → Graphviz DOT)    │
         │                                                            │
         │  ML Classification:                                        │
         │  ├── classify_breast_ultrasound (OncoScanBC, MobileNetV2)  │
         │  ├── classify_skin_lesion      (OncoScanSkin, MobileNetV2) │
         │  └── classify_cancer_type          (OncoTypeBC, PyTorch)  │
         └───────────────────────────────────────────────────────────┘
                                    │
                        Store response in cache
```

2–3 API calls per query. At 500 RPD free tier → ~170–250 queries/day.

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
- **Biological Pathway Diagrams**: Graphviz DOT generation rendered live in the UI.
- **ML Classification**: OncoScanBC (breast ultrasound), OncoScanSkin (dermoscopy), OncoTypeBC (molecular subtyping).
- **Auto-updating Knowledge Base**: APScheduler re-indexes `knowledge_base/` every 30 minutes.

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Google Gemini 3.1 Flash Lite Preview (500 req/day free tier) |
| Agent Framework | LangGraph 1.0.x (single-agent, StateGraph) |
| Orchestration | LangChain |
| Embeddings | HuggingFace `paraphrase-multilingual-MiniLM-L12-v2` |
| Vector DB | ChromaDB (RAG + response cache) |
| ML Models | PyTorch MobileNetV2 (OncoScanBC, OncoScanSkin, OncoTypeBC) |
| External APIs | ClinicalTrials.gov, NCBI PubMed, arXiv |
| App Framework | Streamlit |
| Containerization | Docker, Docker Compose |
| Cloud | AWS EC2 (t3.medium), AWS ECR |
| CI/CD | GitHub Actions |

---

## Project Structure

```
gemini_rag_chatbot/
├── .github/workflows/deploy.yml
├── agent/
│   ├── __init__.py
│   ├── cache.py           # Semantic query-response cache
│   ├── memory.py          # MemorySaver checkpointer
│   └── onco_agent.py      # Single-agent LangGraph graph
├── tools/
│   ├── __init__.py
│   ├── onco_tools.py      # RAG, image analysis, diagrams, ML classifiers
│   └── external_tools.py  # ClinicalTrials.gov, PubMed, arXiv
├── evaluation/
│   └── ragas_eval.py      # RAGAS faithfulness/relevancy evaluation
├── knowledge_base/        # MedQuAD XMLs and arXiv PDFs
├── models/                # PyTorch model weights (not committed)
├── app.py                 # Streamlit UI
├── updater.py             # Incremental knowledge base indexer
├── test_agent.py          # End-to-end verification script
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
python test_agent.py     # Verify everything works
streamlit run app.py
```

---

## EC2 Deployment

GitHub Actions (`.github/workflows/deploy.yml`) builds, pushes to ECR, and deploys on every push to `main`.

Required GitHub Actions secrets:
```
AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION,
ECR_REGISTRY, ECR_REPOSITORY, EC2_HOST, EC2_SSH_KEY, GEMINI_API_KEY
```

First deployment: `docker exec -it oncobot-container python updater.py`

---

## API Quota

The project uses `gemini-3.1-flash-lite-preview` (500 RPD free tier). The semantic cache serves repeated and paraphrased queries without consuming quota. On exhaustion, the agent retries with the API-suggested delay before showing a user-friendly error.

---

## Disclaimer

This tool is for educational and research purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified oncologist.

# OncoBot: Agentic Cancer Research Assistant

OncoBot is a production-grade, containerized agentic AI system designed to assist cancer researchers, clinicians, and patients with oncology inquiries.

It was originally built as a standard RAG chatbot and has been upgraded to a full **LangGraph ReAct agent** with multi-tool reasoning, conversational memory, live external data sources, and a multi-agent supervisor architecture.

Deployed on AWS EC2 via a fully automated GitHub Actions CI/CD pipeline.

---

## Architecture

### v1 (original): Static RAG chain
```
User query → ChromaDB retriever (k=4) → Prompt template → Gemini LLM → Response
```

### v2 (current): LangGraph ReAct Agent
```
User query → Agent reasons → Decides which tools to call → Calls them → Synthesizes → Response
              |
              ├── get_sentiment_tone        (DistilBERT, local)
              ├── oncology_rag_search       (ChromaDB vector store)
              ├── fetch_pubmed_abstracts    (NCBI E-utilities API)
              ├── search_clinical_trials    (ClinicalTrials.gov API v2)
              ├── generate_pathway_diagram  (Gemini → Graphviz DOT)
              ├── analyze_medical_image     (Gemini Vision)
              └── summarize_arxiv_paper     (arXiv API)
```

The agent runs a Thought → Action → Observation loop (ReAct pattern) until it has a complete answer. It uses `MemorySaver` checkpointing to maintain conversation history per session, so follow-up questions work without repeating context.

A **multi-agent supervisor** (`agent/supervisor.py`) is also implemented as an optional alternative. It routes queries to specialist sub-agents: a Research Agent (RAG + PubMed + arXiv), a Clinical Agent (trials + treatment info), and a Support Agent (empathetic patient responses). It is not active by default — swap it in by changing one line in `app.py`.

---

## Key Features

**Agentic Reasoning**
The LLM decides which tools to call and in what order based on the query. A factual question about cisplatin will trigger RAG search. "Latest research on CAR-T" will also trigger PubMed. "Are there trials for stage 4 lung cancer?" will trigger ClinicalTrials.gov. The agent chains these calls as needed.

**Conversational Memory**
Each browser session gets a unique thread ID. LangGraph's `MemorySaver` stores the full message history for that thread, so the agent can answer follow-up questions with context from earlier in the conversation.

**Live External Data**
- ClinicalTrials.gov REST API v2: searches recruiting trials by condition and phase. No API key required.
- NCBI PubMed E-utilities: fetches recent peer-reviewed abstracts. No API key required.
- arXiv API: looks up specific papers by ID on demand.

**Reasoning Transparency**
The right panel of the UI shows a collapsible "Agent Reasoning" section that lists every tool call, its arguments, and the observation returned — so users can see exactly how the agent arrived at its answer.

**Sentiment-Aware Responses**
DistilBERT classifies each query as distressed or clinical. The agent's system prompt instructs it to lead with empathy before information when distress is detected.

**Multilingual**
The `paraphrase-multilingual-MiniLM-L12-v2` embedding model allows queries in any language to match against the English knowledge base. The LLM responds in the language the user wrote in.

**Multimodal**
Users can upload a medical scan or diagram. The image is base64-encoded and passed to Gemini Vision via the `analyze_medical_image` tool.

**Biological Pathway Diagrams**
The `generate_pathway_diagram` tool prompts Gemini to produce Graphviz DOT code, which is rendered live in the UI with `st.graphviz_chart`.

**Auto-updating Knowledge Base**
APScheduler runs `updater.py` every 30 minutes in a background thread. It checks file modification times and only re-indexes changed files, so adding a new PDF or XML to `knowledge_base/` is picked up automatically without restarting the container.

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Google Gemini 2.5 Flash |
| Agent Framework | LangGraph 0.2.x (ReAct + StateGraph) |
| Orchestration | LangChain |
| Embeddings | HuggingFace `paraphrase-multilingual-MiniLM-L12-v2` |
| Vector DB | ChromaDB (persistent, Docker volume) |
| Sentiment | Transformers DistilBERT |
| External APIs | ClinicalTrials.gov, NCBI PubMed, arXiv |
| App Framework | Streamlit |
| Containerization | Docker, Docker Compose |
| Cloud | AWS EC2 (t3.medium), AWS ECR |
| CI/CD | GitHub Actions |

---

## Project Structure

```
gemini_rag_chatbot/
├── .github/
│   └── workflows/
│       └── deploy.yml          # GitHub Actions: build -> push ECR -> SSH deploy EC2
├── agent/
│   ├── __init__.py
│   ├── memory.py               # MemorySaver checkpointer, thread-scoped session memory
│   ├── onco_agent.py           # LangGraph ReAct agent (default)
│   └── supervisor.py           # Multi-agent supervisor with specialist routing (optional)
├── tools/
│   ├── __init__.py
│   ├── onco_tools.py           # RAG search, image analysis, diagram gen, sentiment
│   └── external_tools.py       # ClinicalTrials.gov, PubMed, arXiv tools
├── knowledge_base/             # MedQuAD XMLs and arXiv PDFs (mounted as Docker volume)
├── app.py                      # Streamlit UI: chat, visualization panel, reasoning panel
├── updater.py                  # Incremental knowledge base indexer
├── fetch_cancer_data.py        # arXiv scraper to populate knowledge_base/
├── test_agent.py               # 12-step verification script (run before deploying)
├── check_models.py             # Lists available Gemini models for your API key
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## Local Development

**Prerequisites:** Python 3.11, a Gemini API key.

```bash
# 1. Clone and create venv
git clone https://github.com/vyshnavnp/gemini-rag-chatbot.git
cd gemini-rag-chatbot
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your API key
mkdir .streamlit
echo 'GEMINI_API_KEY = "your-key-here"' > .streamlit/secrets.toml

# 4. Build the knowledge base (first time only)
python updater.py

# 5. Run the verification suite
python test_agent.py

# 6. Start the app
streamlit run app.py
```

---

## EC2 Deployment

The GitHub Actions workflow in `.github/workflows/deploy.yml` handles deployment automatically on every push to `main`.

It:
1. Builds the Docker image
2. Pushes it to AWS ECR
3. SSHs into EC2, pulls the new image, and restarts the container via `docker compose up -d`

Required GitHub Actions secrets:
```
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_REGION
ECR_REGISTRY
ECR_REPOSITORY
EC2_HOST
EC2_SSH_KEY
GEMINI_API_KEY
```

On first deployment, after the container starts, build the knowledge base inside the container:
```bash
docker exec -it oncobot-container python updater.py
```

After that, the background scheduler handles updates automatically every 30 minutes.

---

## Switching to Multi-Agent Mode

To use the supervisor with specialist agents instead of the single ReAct agent, change one line in `app.py`:

```python
# In the load_agent() function, replace:
return build_agent()

# With:
from agent.supervisor import build_supervisor
return build_supervisor()
```

Everything else (UI, memory, tool output parsing) stays the same.

---

## Disclaimer

This tool is for educational and research purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified oncologist.

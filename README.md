# 🎗️ OncoBot: Intelligent Cancer Research Assistant

**OncoBot** is a production-grade, containerized AI agent designed to assist researchers and patients with oncology inquiries. It utilizes **RAG (Retrieval-Augmented Generation)** to provide grounded, expert answers based on a curated library of medical PDFs and research papers.

This project demonstrates a full-stack AI implementation featuring **Multimodal analysis**, **Sentiment-aware responses**, **Cross-lingual search**, and a fully automated **CI/CD pipeline on AWS**.

---

## 🚀 Key Features

### 🧠 Advanced RAG Architecture
- **Curated Knowledge:** Answers are strictly grounded in a local vector database of arXiv oncology papers and MedQuAD XML data.
- **Dynamic Updates:** A background "Librarian" scheduler watches the `knowledge_base` folder and automatically indexes new PDFs/XMLs/CSVs without downtime.

### 👁️ Multimodal & Visual
- **Vision Support:** Users can upload medical diagrams or scans, and the bot analyzes them using **Google Gemini Vision** models.
- **Biological Pathway Visualization:** Generates real-time **Graphviz flowcharts** to visualize complex cellular processes (e.g., T-Cell activation, Metastasis pathways).

### 🌍 Global & Empathetic
- **Polyglot:** Supports cross-lingual search. Ask in **Spanish, Hindi, or French**, and the bot searches English medical records to answer in your native language.
- **Sentiment Analysis:** Uses a **BERT-based model** to detect user distress. The bot automatically adjusts its tone from "Clinical/Objective" to "Empathetic/Reassuring" based on the user's emotional state.

---

## 🛠️ Tech Stack

### AI & NLP
- **LLM:** Google Gemini 2.5 Flash / Pro
- **Embeddings:** HuggingFace `paraphrase-multilingual-MiniLM-L12-v2`
- **Orchestration:** LangChain, LangDetect
- **Vector DB:** ChromaDB (Persistent)
- **Sentiment:** Transformers (DistilBERT)

### Backend & DevOps
- **App Framework:** Streamlit
- **Containerization:** Docker & Docker Compose
- **Cloud Provider:** AWS EC2 (t3.medium) & AWS ECR (Private Registry)
- **CI/CD:** GitHub Actions (Automated Build -> Push to ECR -> SSH Deploy)

---

## 📂 Project Structure

```bash
OncoBot/
├── .github/workflows/   # CI/CD Pipeline (deploy.yml)
├── knowledge_base/      # Folder for PDFs, XMLs, CSVs
├── chroma_db/           # Vector Database storage
├── app.py               # Main Chat Interface & Logic
├── updater.py           # Indexing & Parsing Logic
├── fetch_cancer_data.py # Script to scrape arXiv papers
├── Dockerfile           # Docker configuration
└── docker-compose.yml   # Container orchestration

import os
import json
import time
import hashlib
import xml.etree.ElementTree as ET
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
# --- Config ---
# Paths are absolute so the script works correctly whether run as
#   python updater.py              (from any directory)
#   or from the APScheduler/thread inside the Streamlit process.
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(_PROJECT_ROOT, "chroma_db")
DATA_PATH = os.path.join(_PROJECT_ROOT, "knowledge_base")
# Store metadata inside chroma_db/ so it persists on the bind-mounted volume.
# This prevents duplicate re-indexing after container rebuilds.
METADATA_FILE = os.path.join(CHROMA_PATH, "index_metadata.json")
# Must match _COLLECTION_NAME in tools/onco_tools.py so that the indexer and
# retriever always read from and write to the same ChromaDB collection.
COLLECTION_NAME = "langchain"
# Multilingual Model (Task 6)
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def load_metadata():
    try:
        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    except (json.JSONDecodeError, OSError):
        return {}

def save_metadata(metadata):
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)

def file_hash(filepath):
    """SHA-256 hash of file contents. Reliable across Docker rebuilds
    (unlike mtime, which changes on every git checkout / COPY)."""
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()

def process_pdf(filepath):
    try:
        loader = PyPDFLoader(filepath)
        docs = loader.load()
        # Larger overlap for medical context continuity
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return splitter.split_documents(docs)
    except Exception as e:
        print(f"Error reading PDF {filepath}: {e}")
        return []

def process_xml(filepath):
    documents = []
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()
        focus = root.find("Focus")
        topic = focus.text if focus is not None else "Medical Topic"
        
        for qa in root.findall(".//QAPair"):
            q = qa.find("Question").text
            a = qa.find("Answer").text
            if q and a:
                # Add explicit "Medical Context" metadata
                text = f"Oncology Topic: {topic}\nQuestion: {q}\nAnswer: {a}"
                documents.append(Document(page_content=text, metadata={"source": filepath}))
    except (ET.ParseError, OSError) as e:
        print(f"Error parsing XML {filepath}: {e}")
    return documents

def update_knowledge_base():
    print(f"\n[{time.strftime('%X')}] 🧬 OncoBot Knowledge Update Started...")
    indexed_data = load_metadata()
    files_to_index = []
    
    for filename in os.listdir(DATA_PATH):
        if filename.endswith((".pdf", ".xml")):
            filepath = os.path.join(DATA_PATH, filename)
            fhash = file_hash(filepath)
            if indexed_data.get(filepath) != fhash:
                files_to_index.append(filepath)

    if not files_to_index:
        return

    print(f"📚 Integrating {len(files_to_index)} new oncology resources.")
    
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings,
                collection_name=COLLECTION_NAME)
    
    for filepath in files_to_index:
        print(f"-> Processing {os.path.basename(filepath)}...")
        chunks = process_pdf(filepath) if filepath.endswith(".pdf") else process_xml(filepath)
        if chunks:
            db.add_documents(chunks)
            indexed_data[filepath] = file_hash(filepath)

    # ChromaDB with PersistentClient auto-persists; no manual persist() needed.
    save_metadata(indexed_data)
    print("✅ Medical Database Updated.")

if __name__ == "__main__":
    update_knowledge_base()
import os
import json
import time
import xml.etree.ElementTree as ET
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
# --- Config ---
CHROMA_PATH = "chroma_db"
DATA_PATH = "knowledge_base"
METADATA_FILE = "index_metadata.json"
# Multilingual Model (Task 6)
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def load_metadata():
    try:
        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, 'r') as f:
                return json.load(f)
        return {}
    except:
        return {}

def save_metadata(metadata):
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=4)

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
    except:
        pass
    return documents

def update_knowledge_base():
    print(f"\n[{time.strftime('%X')}] 🧬 OncoBot Knowledge Update Started...")
    indexed_data = load_metadata()
    files_to_index = []
    
    for filename in os.listdir(DATA_PATH):
        if filename.endswith((".pdf", ".xml")):
            filepath = os.path.join(DATA_PATH, filename)
            mtime = os.path.getmtime(filepath)
            if filepath not in indexed_data or (mtime - indexed_data.get(filepath, 0.0)) > 1:
                files_to_index.append(filepath)

    if not files_to_index:
        return

    print(f"📚 Integrating {len(files_to_index)} new oncology resources.")
    
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    
    for filepath in files_to_index:
        print(f"-> Processing {os.path.basename(filepath)}...")
        chunks = process_pdf(filepath) if filepath.endswith(".pdf") else process_xml(filepath)
        if chunks:
            db.add_documents(chunks)
            indexed_data[filepath] = os.path.getmtime(filepath)

    db.persist()
    save_metadata(indexed_data)
    print("✅ Medical Database Updated.")

if __name__ == "__main__":
    update_knowledge_base()
import streamlit as st
import os
import base64
from PIL import Image
from apscheduler.schedulers.background import BackgroundScheduler
from langdetect import detect
from transformers import pipeline

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage

from updater import update_knowledge_base

# 1. Try getting the key from Docker/Environment Variable first
API_KEY = os.getenv("GEMINI_API_KEY")

# 2. If not found, look for local secrets.toml (Local Development)
if not API_KEY:
    try:
        API_KEY = st.secrets["GEMINI_API_KEY"]
    except Exception: # <--- CHANGED: Catch ALL errors (including StreamlitSecretNotFoundError)
        st.error("Missing GEMINI_API_KEY. Set it as an Env Var or in secrets.toml")
        st.stop()

CHROMA_PATH = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
GENERATOR_MODEL = "gemini-2.5-flash-preview-09-2025"

# --- Initialization ---
@st.cache_resource
def load_system():
    # Task 5: Sentiment Analysis
    sentiment_pipe = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    
    # Task 1 & 6: Multilingual Vector Store
    if not os.path.exists(CHROMA_PATH):
        return None, None, None
        
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    llm = ChatGoogleGenerativeAI(model=GENERATOR_MODEL, api_key=API_KEY)
    
    return sentiment_pipe, retriever, llm

# --- Background Updater ---
if 'scheduler_started' not in st.session_state:
    st.session_state['scheduler_started'] = False

def start_scheduler():
    if not st.session_state['scheduler_started']:
        scheduler = BackgroundScheduler()
        scheduler.add_job(update_knowledge_base, 'interval', minutes=30)
        scheduler.start()
        st.session_state['scheduler_started'] = True

# --- Main UI ---
st.set_page_config(layout="wide", page_title="OncoBot AI", page_icon="🎗️")

st.title("🎗️ OncoBot: Intelligent Cancer Research Assistant")
st.caption("Specialized in Oncology, Treatment Pathways, and Patient Support. (Not a replacement for a doctor).")

start_scheduler()
sentiment_pipe, retriever, llm = load_system()

if not retriever:
    st.error("Knowledge Base is empty. Please run 'python updater.py' first.")
    st.stop()

# --- Sidebar ---
st.sidebar.header("Patient/Researcher Tools")
uploaded_file = st.sidebar.file_uploader("Upload Scan/Diagram (Research Use Only)", type=["jpg", "png"])
if uploaded_file:
    st.sidebar.image(uploaded_file, caption="Analyzing Visual Data...", use_container_width=True)
    uploaded_file.seek(0)

st.sidebar.markdown("---")
lang_display = st.sidebar.empty()
sent_display = st.sidebar.empty()

# --- Logic ---
col1, col2 = st.columns([3, 2]) # Wider chat, smaller graph

with col1:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about immunotherapy, specific carcinomas, or side effects..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Consulting Oncology Database..."):
                response_text = ""
                graph_code = None

                # --- 1. Vision Mode ---
                if uploaded_file:
                    try:
                        vision_llm = ChatGoogleGenerativeAI(model=GENERATOR_MODEL, api_key=API_KEY)
                        image_bytes = uploaded_file.getvalue()
                        b64 = base64.b64encode(image_bytes).decode()
                        msg = HumanMessage(content=[
                            {"type": "text", "text": f"Context: This is a medical image related to cancer/oncology. Question: {prompt}"},
                            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{b64}"}
                        ])
                        response_text = vision_llm.invoke([msg]).content
                    except Exception as e:
                        response_text = f"Error interpreting image: {e}"

                # --- 2. Text Expert Mode ---
                else:
                    # A. Language Detection
                    try:
                        lang_code = detect(prompt)
                    except:
                        lang_code = "en"
                    lang_display.info(f"🌐 Language: **{lang_code.upper()}**")

                    # B. Sentiment Analysis
                    sent_res = sentiment_pipe(prompt)[0]
                    sentiment = sent_res['label']
                    score = sent_res['score']
                    
                    if sentiment == "NEGATIVE":
                        sent_display.error(f"❤️ Emotional Distress Detected ({score:.2f})")
                        tone_prompt = "The user appears distressed or worried. Be extremely empathetic, reassuring, and gentle. Use phrases like 'I understand this is difficult'."
                    else:
                        sent_display.success(f"⚕️ Clinical Tone ({score:.2f})")
                        tone_prompt = "Be professional, precise, and objective."

                    # C. The "Curated" Prompt
                    # UPDATED INSTRUCTION 3: FORCE VERTICAL LAYOUT
                    template = f"""
                    You are OncoBot, an AI specialized ONLY in Cancer and Oncology.
                    
                    USER METADATA:
                    - Language: {lang_code}
                    - Sentiment: {sentiment} ({tone_prompt})
                    
                    INSTRUCTIONS:
                    1. DOMAIN CHECK: If the user asks about anything NOT related to cancer/biology, REFUSE politely.
                    2. KNOWLEDGE: Use the provided Context to answer.
                    3. VISUALIZATION: If user asks to "visualize", "map", or "show flow", generate a GRAPHVIZ DOT block (```dot ... ```). 
                       CRITICAL: Inside the dot block, start with 'digraph G {{{{ rankdir=TB; ... }}}}' to ensure the graph is VERTICAL (Top-to-Bottom). Do NOT use Left-to-Right.
                    4. ENTITIES: List "Key Medical Terms" found.
                    5. SAFETY: Include disclaimer: "This is AI, not a doctor."
                    6. OUTPUT: Answer in {lang_code}.

                    CONTEXT:
                    {{context}}

                    QUESTION:
                    {{question}}
                    """

                    rag_chain = (
                        {"context": retriever, "question": RunnablePassthrough()}
                        | ChatPromptTemplate.from_template(template)
                        | llm
                    )
                    
                    full_response = rag_chain.invoke(prompt).content
                    
                    # D. Graph Extraction
                    if "```dot" in full_response:
                        parts = full_response.split("```dot")
                        response_text = parts[0]
                        graph_code = parts[1].split("```")[0]
                    else:
                        response_text = full_response

                st.markdown(response_text)
                if graph_code:
                    st.session_state['last_graph'] = graph_code
        
        st.session_state.messages.append({"role": "assistant", "content": response_text})

with col2:
    st.subheader("🧬 Biological Pathways")
    if 'last_graph' in st.session_state:
        # UPDATED: Added use_container_width=True to make it fill the column
        st.graphviz_chart(st.session_state['last_graph'], use_container_width=True)
        st.caption("Visual representation of the medical concept generated from the text.")
    else:
        st.info("Example commands:\n- 'Visualize the metastasis pathway'\n- 'Show a diagram of T-Cell activation'\n- 'Map the side effects of Chemotherapy'")
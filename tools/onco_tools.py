# tools/onco_tools.py
#
# This file wraps the core capabilities of the original OncoBot into
# LangChain @tool functions. The agent in agent/onco_agent.py imports
# these and decides when to call them based on the user query.
#
# Each tool is a plain Python function decorated with @tool.
# The docstring is critical -- LangChain uses it as the tool description
# that the LLM reads to decide whether to call the tool.

import os
import base64
from typing import Optional

from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline

# These constants mirror what is set in app.py so there is one source of truth.
CHROMA_PATH = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
GENERATOR_MODEL = "gemini-2.5-flash"

# ---------------------------------------------------------------------------
# Module-level singletons
# These are loaded once per process. Streamlit's @st.cache_resource handles
# caching at the app layer, but these tools are also called from inside the
# agent which runs outside of the Streamlit resource cache. Lazy-loading each
# time would be very slow, so we cache them at the module level instead.
# ---------------------------------------------------------------------------
_retriever = None
_sentiment_pipe = None


def _get_retriever():
    """
    Load the ChromaDB retriever once and reuse it.
    Returns None if the chroma_db directory does not exist yet.
    """
    global _retriever
    if _retriever is None:
        if not os.path.exists(CHROMA_PATH):
            return None
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        vector_store = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings
        )
        # k=6 gives the agent slightly more context than the old fixed k=4.
        # The agent can decide how much of that context to use.
        _retriever = vector_store.as_retriever(search_kwargs={"k": 6})
    return _retriever


def _get_sentiment_pipe():
    """Load the DistilBERT sentiment pipeline once and reuse it."""
    global _sentiment_pipe
    if _sentiment_pipe is None:
        _sentiment_pipe = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
    return _sentiment_pipe


# ---------------------------------------------------------------------------
# Tool 1: RAG Search
# ---------------------------------------------------------------------------

@tool
def oncology_rag_search(query: str) -> str:
    """
    Search the local oncology knowledge base (ChromaDB vector store) for
    information related to the given query.

    Use this tool whenever the user asks a factual question about:
    - cancer types, stages, or symptoms
    - treatment options (chemotherapy, immunotherapy, radiation, surgery)
    - drug names, mechanisms, or side effects
    - oncology research, clinical context, or patient support topics

    The knowledge base contains MedQuAD XML question-answer pairs and
    arXiv oncology research paper PDFs. Returns up to 6 relevant passages
    as a single combined string.

    Args:
        query: The search query in any language. The multilingual embedding
               model will match it against the English knowledge base.

    Returns:
        A string containing the concatenated relevant passages, each
        prefixed with its source filename.
    """
    retriever = _get_retriever()
    if retriever is None:
        return (
            "Knowledge base is not available. "
            "Run 'python updater.py' to build it first."
        )

    docs = retriever.invoke(query)

    if not docs:
        return "No relevant oncology information found for this query."

    # Format each retrieved chunk with its source so the LLM can cite it.
    results = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown source")
        source_name = os.path.basename(source)
        results.append(f"[Source {i}: {source_name}]\n{doc.page_content}")

    return "\n\n---\n\n".join(results)


# ---------------------------------------------------------------------------
# Tool 2: Medical Image Analysis (Vision)
# ---------------------------------------------------------------------------

@tool
def analyze_medical_image(question: str, image_b64: str) -> str:
    """
    Analyze a medical image (scan, diagram, or pathology slide) using
    Google Gemini's vision capability.

    Use this tool when the user has uploaded an image and is asking a
    question about it. The image must already be base64-encoded before
    passing it to this tool.

    Args:
        question: The user's question about the image.
        image_b64: The base64-encoded image bytes as a string (no data URI
                   prefix -- just the raw base64 string).

    Returns:
        A string with Gemini's interpretation of the medical image in the
        context of the user's question.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "GEMINI_API_KEY is not set. Cannot perform image analysis."

    vision_llm = ChatGoogleGenerativeAI(model=GENERATOR_MODEL, api_key=api_key)

    message = HumanMessage(content=[
        {
            "type": "text",
            "text": (
                "You are an oncology image analysis assistant. "
                "Analyze this medical image in the context of cancer research. "
                f"Question: {question}"
            )
        },
        {
            "type": "image_url",
            "image_url": f"data:image/jpeg;base64,{image_b64}"
        }
    ])

    try:
        response = vision_llm.invoke([message])
        return response.content
    except Exception as e:
        return f"Image analysis failed: {str(e)}"


# ---------------------------------------------------------------------------
# Tool 3: Biological Pathway Diagram Generation
# ---------------------------------------------------------------------------

@tool
def generate_pathway_diagram(topic: str) -> str:
    """
    Generate a Graphviz DOT language diagram for a biological or clinical
    pathway related to oncology.

    Use this tool when the user asks to:
    - visualize a pathway (e.g., "show me the metastasis pathway")
    - draw a diagram (e.g., "diagram of T-cell activation")
    - map a process (e.g., "map chemotherapy side effects")
    - show a flowchart of any cancer-related biological process

    The output is a raw Graphviz DOT string. The Streamlit app renders
    this with st.graphviz_chart(). The diagram uses top-to-bottom layout
    (rankdir=TB) for readability.

    Args:
        topic: A plain English description of the pathway or process to
               visualize (e.g., "PD-1/PD-L1 checkpoint inhibition pathway").

    Returns:
        A Graphviz DOT format string, or an error message if generation
        fails. The string does NOT include the triple-backtick fences --
        just the raw DOT content starting with 'digraph'.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "GEMINI_API_KEY is not set. Cannot generate diagram."

    llm = ChatGoogleGenerativeAI(model=GENERATOR_MODEL, api_key=api_key)

    diagram_prompt = f"""
You are a biomedical visualization expert.
Generate ONLY a valid Graphviz DOT language diagram for the following oncology topic.

Rules:
- Start with: digraph G {{ rankdir=TB;
- Use descriptive node labels in double quotes
- Use -> for directed edges
- End with }}
- Output ONLY the DOT code, no explanation, no markdown fences, no extra text

Topic: {topic}
"""

    try:
        response = llm.invoke(diagram_prompt)
        dot_content = response.content.strip()

        # Strip markdown code fences if the model added them anyway.
        if "```" in dot_content:
            lines = dot_content.split("\n")
            dot_content = "\n".join(
                line for line in lines
                if not line.strip().startswith("```")
            )

        return dot_content.strip()
    except Exception as e:
        return f"Diagram generation failed: {str(e)}"


# ---------------------------------------------------------------------------
# Tool 4: Sentiment Tone Analysis
# ---------------------------------------------------------------------------

@tool
def get_sentiment_tone(text: str) -> str:
    """
    Analyze the emotional tone of the user's message to determine whether
    the response should use an empathetic or clinical/professional tone.

    Use this tool at the start of any conversation turn to understand the
    user's emotional state before crafting a response -- especially if the
    query contains words suggesting fear, worry, diagnosis, or personal
    medical situations.

    Args:
        text: The user's raw query text.

    Returns:
        A short string describing the detected tone and how the response
        should be adjusted. Examples:
        - "NEGATIVE (score: 0.97) - Use empathetic tone. Acknowledge distress."
        - "POSITIVE (score: 0.88) - Use clinical/professional tone."
    """
    pipe = _get_sentiment_pipe()
    result = pipe(text[:512])[0]  # Truncate to 512 tokens, model limit.

    label = result["label"]
    score = result["score"]

    if label == "NEGATIVE":
        return (
            f"NEGATIVE (score: {score:.2f}) - "
            "User appears distressed or worried. "
            "Use an empathetic and reassuring tone. "
            "Acknowledge that this is difficult before giving medical information."
        )
    else:
        return (
            f"POSITIVE (score: {score:.2f}) - "
            "User appears calm or is asking in a clinical/research context. "
            "Use a precise, professional, objective tone."
        )

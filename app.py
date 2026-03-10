# app.py
#
# OncoBot - Agentic Oncology Research Assistant
#
# This is the main Streamlit application. It wires together:
#   - The LangGraph ReAct agent (agent/onco_agent.py)
#   - The background knowledge base updater (updater.py)
#   - The Streamlit UI: chat, sidebar, visualization panel, reasoning panel
#
# What changed from v1:
#   Before: user query -> retriever -> prompt -> LLM -> response (one shot, no reasoning)
#   After:  user query -> agent reasons -> calls tools as needed -> synthesizes response
#
# The agent handles sentiment analysis, RAG search, PubMed, ClinicalTrials,
# and image analysis as individual tool calls. The UI now shows which tools
# were used and the agent's reasoning steps in a collapsible panel.

import streamlit as st
import os
import uuid
import base64
from apscheduler.schedulers.background import BackgroundScheduler

from updater import update_knowledge_base
from agent.supervisor import build_supervisor, run_supervisor, stream_supervisor
from agent.cache import cache_size, clear_cache

# ---------------------------------------------------------------------------
# API Key Setup
# ---------------------------------------------------------------------------
# Check environment variable first (Docker/EC2 sets this via docker-compose).
# Fall back to .streamlit/secrets.toml for local development.

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        st.error(
            "GEMINI_API_KEY not found. "
            "Set it as an environment variable or add it to .streamlit/secrets.toml"
        )
        st.stop()

# Set the key in the environment so the agent and tool modules can read it
# with os.getenv() without needing to pass it around explicitly.
os.environ["GEMINI_API_KEY"] = api_key

CHROMA_PATH = "chroma_db"

# ---------------------------------------------------------------------------
# Cached resource: Agent
# ---------------------------------------------------------------------------
# build_agent() loads the LLM, registers all tools, and compiles the
# LangGraph state machine. This is expensive so we cache it once per
# Streamlit process using @st.cache_resource.

@st.cache_resource
def load_agent():
    """
    Load and compile the 5-role multi-agent supervisor graph.

    Returns the compiled graph, or stops the app if the API key is missing.
    The agents handle missing knowledge base gracefully via the RAG tool.
    """
    try:
        return build_supervisor()
    except EnvironmentError as e:
        st.error(str(e))
        st.stop()

# ---------------------------------------------------------------------------
# Background Updater
# ---------------------------------------------------------------------------
# APScheduler runs update_knowledge_base() every 30 minutes in a background
# thread. It checks file modification times (mtime) and only re-indexes
# files that changed. This matches the behavior from the original app.

def _start_scheduler():
    """
    Start the APScheduler background job to re-index the knowledge base
    every 30 minutes. Guards against being started more than once in the
    same Streamlit session.
    """
    if not st.session_state.get("scheduler_started", False):
        scheduler = BackgroundScheduler()
        scheduler.add_job(update_knowledge_base, "interval", minutes=30)
        scheduler.start()
        st.session_state["scheduler_started"] = True

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------

st.set_page_config(
    layout="wide",
    page_title="OncoBot AI",
    page_icon="ribbon"
)

st.title("OncoBot: Intelligent Cancer Research Assistant")
st.caption(
    "Specialized in Oncology, Treatment Pathways, and Patient Support. "
    "Not a replacement for a doctor."
)

# Run the scheduler and load the agent on every page load.
# Both are guarded against re-initialization.
_start_scheduler()
agent_graph = load_agent()

# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
# Each browser session gets a unique thread_id. LangGraph uses this to
# store and retrieve the conversation history from the MemorySaver checkpointer.
# This means the agent remembers what was said earlier in the same session.

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = str(uuid.uuid4())

if "messages" not in st.session_state:
    # Chat history for the Streamlit UI display (role + content pairs).
    st.session_state["messages"] = []

if "last_graph_dot" not in st.session_state:
    # Stores the most recent Graphviz DOT string for the visualization panel.
    st.session_state["last_graph_dot"] = None

if "last_reasoning_steps" not in st.session_state:
    # Stores the agent's reasoning steps from the last turn for the
    # transparency panel (collapsible expander in the right column).
    st.session_state["last_reasoning_steps"] = []

if "last_tools_used" not in st.session_state:
    # List of tool names called in the last turn, shown as badges in the UI.
    st.session_state["last_tools_used"] = []

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.header("Patient / Researcher Tools")

uploaded_file = st.sidebar.file_uploader(
    "Upload Scan or Diagram (Research Use Only)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    st.sidebar.image(
        uploaded_file,
        caption="Image will be analyzed by the agent.",
        use_container_width=True
    )
    # Reset the file pointer after preview so we can read it again later.
    uploaded_file.seek(0)

st.sidebar.markdown("---")
st.sidebar.markdown("**Session ID**")
st.sidebar.code(st.session_state["thread_id"][:8] + "...", language=None)

st.sidebar.markdown("---")
st.sidebar.markdown("**Response Cache**")
st.sidebar.caption(
    f"{cache_size()} response(s) cached. "
    "Cached answers are served instantly without using API quota."
)
if st.sidebar.button("Clear response cache"):
    deleted = clear_cache()
    st.sidebar.success(f"Cleared {deleted} cached response(s).")

st.sidebar.markdown("---")
st.sidebar.markdown("**Response Quality (RAGAS)**")
st.sidebar.caption(
    "Measures faithfulness and answer relevance against the RAG context. "
    "Uses Gemini as an LLM judge — costs 1-2 extra API requests."
)
if st.sidebar.button("Evaluate last response"):
    _msgs = st.session_state.get("messages", [])
    _last_q = next((m["content"] for m in reversed(_msgs) if m["role"] == "user"), None)
    _last_a = next((m["content"] for m in reversed(_msgs) if m["role"] == "assistant"), None)
    if _last_q and _last_a:
        with st.sidebar:
            with st.spinner("Running RAGAS evaluation..."):
                from evaluation.ragas_eval import evaluate_last_response
                _scores = evaluate_last_response(_last_q, _last_a)
        if "error" in _scores:
            st.sidebar.warning(f"Evaluation failed: {_scores['error']}")
        else:
            for _metric, _score in _scores.items():
                st.sidebar.metric(
                    label=_metric.replace("_", " ").title(),
                    value=f"{_score:.3f}",
                )
    else:
        st.sidebar.info("Send a message first to enable evaluation.")

st.sidebar.markdown("**Example prompts**")
st.sidebar.markdown(
    "- What are the side effects of pembrolizumab?\n"
    "- Show clinical trials for stage 4 lung cancer\n"
    "- Visualize the PD-1 checkpoint pathway\n"
    "- Latest research on CAR-T cell therapy\n"
    "- I am scared, I was just diagnosed with breast cancer"
)

# ---------------------------------------------------------------------------
# Main Layout: Chat (left) | Visualization + Reasoning (right)
# ---------------------------------------------------------------------------

col_chat, col_panel = st.columns([3, 2])

# --- Left column: Chat ---
with col_chat:
    # Replay existing chat history on each render.
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # The main input box.
    prompt = st.chat_input(
        "Ask about immunotherapy, specific carcinomas, clinical trials, or side effects..."
    )

    if prompt:
        # Show the user's message immediately.
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Prepare the image for the agent if one was uploaded.
        image_b64 = None
        if uploaded_file:
            image_bytes = uploaded_file.getvalue()
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        # Stream the agent response token-by-token so the UI feels fast.
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            status_placeholder   = st.empty()
            streamed_text = ""
            final_result = {
                "response": "",
                "graph_dot": None,
                "steps": [],
                "tools_used": [],
                "cache_hit": False,
            }

            for _event in stream_supervisor(
                agent_graph=agent_graph,
                user_message=prompt,
                thread_id=st.session_state["thread_id"],
                image_b64=image_b64,
            ):
                if _event["type"] == "status":
                    # Show which node is currently active.
                    status_placeholder.caption(f"*{_event['content']}*")
                elif _event["type"] == "token":
                    # Append token and re-render with a blinking cursor effect.
                    streamed_text += _event["content"]
                    response_placeholder.markdown(streamed_text + " ▮")
                elif _event["type"] == "done":
                    final_result = _event
                    status_placeholder.empty()

            # Cache hits deliver the full response inside the 'done' event.
            response_text = (
                final_result["response"]
                if final_result.get("cache_hit")
                else streamed_text
            )
            # Final render without cursor.
            response_placeholder.markdown(response_text)

            # Show cache hit notice or tools used as inline badges.
            if final_result.get("cache_hit"):
                similarity_pct = int(final_result.get("cache_similarity", 1.0) * 100)
                st.caption(f"Served from cache (similarity: {similarity_pct}%) — no API quota used.")
            elif final_result.get("tools_used"):
                tools_str = "  |  ".join(final_result["tools_used"])
                st.caption(f"Tools used: {tools_str}")

        # Save the assistant response to chat history.
        st.session_state["messages"].append({
            "role": "assistant",
            "content": response_text
        })

        # Persist data for the right panel.
        if final_result["graph_dot"]:
            st.session_state["last_graph_dot"] = final_result["graph_dot"]
        st.session_state["last_reasoning_steps"] = final_result["steps"]
        st.session_state["last_tools_used"] = final_result["tools_used"]

# --- Right column: Visualization + Reasoning Panel ---
with col_panel:

    # Section 1: Biological pathway diagram
    st.subheader("Biological Pathways")
    if st.session_state["last_graph_dot"]:
        st.graphviz_chart(
            st.session_state["last_graph_dot"],
            use_container_width=True
        )
        st.caption("Diagram generated from the agent's response.")
    else:
        st.info(
            "No diagram yet. Try asking:\n"
            "- Visualize the metastasis pathway\n"
            "- Show a diagram of T-cell activation\n"
            "- Map the side effects of chemotherapy"
        )

    st.markdown("---")

    # Section 2: Agent reasoning transparency
    # This shows the user exactly how the agent arrived at its answer:
    # which tools it called, what arguments it passed, and what they returned.
    st.subheader("Agent Reasoning")
    if st.session_state["last_reasoning_steps"]:
        with st.expander("Show reasoning steps", expanded=False):
            for i, step in enumerate(st.session_state["last_reasoning_steps"], start=1):
                step_type = step["type"]
                step_content = step["content"]

                if step_type == "tool_call":
                    st.markdown(f"**Step {i}: Tool Call**")
                    st.code(step_content, language="text")
                elif step_type == "observation":
                    st.markdown(f"**Step {i}: Observation**")
                    st.code(step_content, language="text")
                elif step_type == "final_answer":
                    st.markdown(f"**Step {i}: Final Answer (preview)**")
                    st.code(step_content, language="text")

                st.markdown("---")
    else:
        st.info("Agent reasoning steps will appear here after you send a message.")

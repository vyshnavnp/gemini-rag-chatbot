# app.py — OncoBot Streamlit application.

import streamlit as st
import os
import uuid
import base64
from apscheduler.schedulers.background import BackgroundScheduler

from updater import update_knowledge_base
from agent.onco_agent import build_agent, stream_agent
from agent.cache import cache_size, clear_cache

# ---------------------------------------------------------------------------
# API Key
# ---------------------------------------------------------------------------

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

os.environ["GEMINI_API_KEY"] = api_key

CHROMA_PATH = "chroma_db"


@st.cache_resource
def load_agent():
    try:
        return build_agent()
    except EnvironmentError as e:
        st.error(str(e))
        st.stop()


def _start_scheduler():
    if not st.session_state.get("scheduler_started", False):
        import threading
        threading.Thread(target=update_knowledge_base, daemon=True).start()
        scheduler = BackgroundScheduler()
        scheduler.add_job(update_knowledge_base, "interval", minutes=30)
        scheduler.start()
        st.session_state["scheduler_started"] = True


st.set_page_config(
    page_title="OncoBot AI",
    page_icon="ribbon",
    layout="centered",
)

_start_scheduler()
agent_graph = load_agent()

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "last_reasoning_steps" not in st.session_state:
    st.session_state["last_reasoning_steps"] = []
if "last_tools_used" not in st.session_state:
    st.session_state["last_tools_used"] = []

# ---------------------------------------------------------------------------
# Sidebar — minimal: uploads + developer tools tucked away
# ---------------------------------------------------------------------------

st.sidebar.markdown("### Upload Files")
st.sidebar.caption("Medical Image (scan, dermoscopy)")
uploaded_file = st.sidebar.file_uploader(
    "Medical image (scan, dermoscopy, diagram)",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
    help="Upload a breast ultrasound, skin lesion, or other medical image for AI analysis.",
)
if uploaded_file:
    st.sidebar.image(uploaded_file, use_container_width=True)
    uploaded_file.seek(0)

st.sidebar.caption("Gene Expression CSV (OncoTypeBC)")
uploaded_csv = st.sidebar.file_uploader(
    "Gene expression CSV",
    type=["csv"],
    label_visibility="collapsed",
    help="Upload a CSV of gene expression features for cancer type classification (OncoTypeBC).",
)
if uploaded_csv:
    st.sidebar.success(f"{uploaded_csv.name} loaded", icon="\u2705")

st.sidebar.markdown("---")

with st.sidebar.expander("Developer Tools", expanded=False):
    st.caption(f"Session: `{st.session_state['thread_id'][:8]}...`")
    st.caption(f"Cache: {cache_size()} response(s)")
    if st.button("Clear cache", use_container_width=True):
        deleted = clear_cache()
        st.success(f"Cleared {deleted} response(s).")
    st.markdown("---")
    st.caption("**RAGAS Evaluation**")
    if st.button("Evaluate last response", use_container_width=True):
        _msgs = st.session_state.get("messages", [])
        _last_q = next((m["content"] for m in reversed(_msgs) if m["role"] == "user"), None)
        _last_a = next((m["content"] for m in reversed(_msgs) if m["role"] == "assistant"), None)
        if _last_q and _last_a:
            with st.spinner("Running RAGAS evaluation..."):
                from evaluation.ragas_eval import evaluate_last_response
                _scores = evaluate_last_response(_last_q, _last_a)
            if "error" in _scores:
                st.warning(f"Evaluation failed: {_scores['error']}")
            else:
                cols = st.columns(len(_scores))
                for col, (_metric, _score) in zip(cols, _scores.items()):
                    col.metric(
                        label=_metric.replace("_", " ").title(),
                        value=f"{_score:.2f}",
                    )
        else:
            st.info("Send a message first.")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown(
    "<h2 style='text-align:center;margin-bottom:0'>OncoBot</h2>"
    "<p style='text-align:center;color:gray;margin-top:0'>"
    "AI Cancer Research Assistant &mdash; not a replacement for a doctor.</p>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Example prompt pills (shown only when conversation is empty)
# ---------------------------------------------------------------------------

if not st.session_state["messages"]:
    _examples = [
        "Side effects of pembrolizumab",
        "Clinical trials for stage 4 lung cancer",
        "What is immunotherapy",
        "Latest research on CAR-T therapy",
    ]
    _cols = st.columns(len(_examples))
    for _col, _ex in zip(_cols, _examples):
        if _col.button(_ex, use_container_width=True):
            st.session_state["_prefill"] = _ex
            st.rerun()

# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Show reasoning inline after the last assistant message if present
if st.session_state["last_reasoning_steps"]:
    with st.expander("Agent Reasoning", expanded=False):
        for i, step in enumerate(st.session_state["last_reasoning_steps"], start=1):
            step_type = step["type"]
            step_content = step["content"]
            label = {"tool_call": "Tool Call", "observation": "Observation", "agent_response": "Response"}.get(step_type, step_type)
            st.markdown(f"**Step {i}: {label}**")
            st.code(step_content, language="text")

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------

_prefill = st.session_state.pop("_prefill", None)
prompt = st.chat_input(
    "Ask about cancer types, treatments, clinical trials, or upload an image..."
)
if _prefill:
    prompt = _prefill

if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    image_b64 = None
    if uploaded_file:
        image_bytes = uploaded_file.getvalue()
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    genomic_csv = None
    if uploaded_csv:
        genomic_csv = uploaded_csv.getvalue().decode("utf-8")

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        status_placeholder = st.empty()
        streamed_text = ""
        final_result = {
            "response": "",
            "steps": [],
            "tools_used": [],
            "cache_hit": False,
        }

        for _event in stream_agent(
            agent_graph=agent_graph,
            user_message=prompt,
            thread_id=st.session_state["thread_id"],
            image_b64=image_b64,
            genomic_csv=genomic_csv,
        ):
            if _event["type"] == "status":
                status_placeholder.caption(f"*{_event['content']}*")
            elif _event["type"] == "token":
                streamed_text += _event["content"]
                response_placeholder.markdown(streamed_text + " \u25ae")
            elif _event["type"] == "done":
                final_result = _event
                status_placeholder.empty()

        response_text = (
            final_result["response"]
            if final_result.get("cache_hit")
            else streamed_text
        )
        response_placeholder.markdown(response_text)

        if final_result.get("cache_hit"):
            similarity_pct = int(final_result.get("cache_similarity", 1.0) * 100)
            st.caption(f"Cached ({similarity_pct}% match) — no API used")
        elif final_result.get("tools_used"):
            tools_str = " \u2192 ".join(final_result["tools_used"])
            st.caption(f"Tools: {tools_str}")

    st.session_state["messages"].append({
        "role": "assistant",
        "content": response_text,
    })

    st.session_state["last_reasoning_steps"] = final_result["steps"]
    st.session_state["last_tools_used"] = final_result["tools_used"]
    st.rerun()

# agent/onco_agent.py — Single-agent LangGraph architecture for OncoBot.

import os
import re
import time
from typing import Annotated

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, AIMessage, ToolMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from tools.onco_tools import (
    oncology_rag_search,
    analyze_medical_image,
    classify_breast_ultrasound,
    classify_skin_lesion,
    classify_cancer_type,
)
from tools.external_tools import (
    search_clinical_trials,
    fetch_pubmed_abstracts,
    summarize_arxiv_paper,
)
from agent.memory import get_checkpointer

MODEL = "gemini-3.1-flash-lite-preview"
THINKING_OFF = {"thinking_config": {"thinking_mode": "DISABLED"}}

# Max tool call rounds before forcing a final answer.
_MAX_TOOL_ITERATIONS = 5
# Max retries on 429 rate-limit errors per LLM call.
_MAX_RETRIES = 3

SYSTEM_PROMPT = """
You are OncoBot, an AI assistant specialized exclusively in oncology and cancer medicine.

TOOLS — use the docstrings to decide which to call:
- oncology_rag_search:              Always call first for factual oncology questions.
- fetch_pubmed_abstracts:           Call when user asks for "latest" or "recent" research.
- search_clinical_trials:           Call for clinical trial queries.
- summarize_arxiv_paper:            Call when user mentions a specific arXiv paper ID.
- analyze_medical_image:            General image analysis with Gemini Vision.
                                    The uploaded image is accessed automatically.
- classify_breast_ultrasound:       Breast ultrasound image → benign/malignant/normal.
                                    Use when user uploads a breast ultrasound scan.
                                    The uploaded image is accessed automatically.
- classify_skin_lesion:             Skin lesion image → 7-class lesion classification.
                                    Use when user uploads a skin/dermoscopy image.
                                    The uploaded image is accessed automatically.
- classify_cancer_type:             Gene expression CSV → cancer type classification
                                    (BRCA, KIRC, LUAD, PRAD, COAD).
                                    Use when user has uploaded gene expression data.
                                    The uploaded CSV is accessed automatically.

IMPORTANT: Image and CSV data are injected into tools automatically from the
user's upload. Do NOT attempt to pass raw image or CSV data as tool arguments.
Just call the appropriate tool — it will access the uploaded file.

RULES:
- If the question is unrelated to cancer or oncology, politely decline.
- Cite source document names from RAG results in your response.
- Respond in the same language the user wrote in.

TONE:
- If the user sounds distressed (scared, worried, diagnosed, afraid), lead with empathy
  and compassion before providing information. Use plain language, avoid jargon.
- Otherwise be clear and professional.

SAFETY:
- End every medical information response with:
  "This information is provided for educational purposes only and is not a substitute
  for professional medical advice. Please consult your oncologist."
"""


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def _safe_content(content) -> str:
    """Safely extract a plain string from an AIMessage.content (handles None, str, list)."""
    if not content:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict)
        )
    return str(content)


def build_agent():
    """Build and compile the single-agent LangGraph graph."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. "
            "Set it as an environment variable or in .streamlit/secrets.toml"
        )

    llm = ChatGoogleGenerativeAI(
        model=MODEL,
        api_key=api_key,
        temperature=0.3,
        model_kwargs=THINKING_OFF,
    )

    all_tools = [
        oncology_rag_search,
        fetch_pubmed_abstracts,
        search_clinical_trials,
        analyze_medical_image,
        summarize_arxiv_paper,
        classify_breast_ultrasound,
        classify_skin_lesion,
        classify_cancer_type,
    ]
    llm_with_tools = llm.bind_tools(all_tools)
    tool_executor = {t.name: t for t in all_tools}

    def agent_node(state: AgentState) -> dict:
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        new_messages = []

        for _ in range(_MAX_TOOL_ITERATIONS):
            for attempt in range(_MAX_RETRIES):
                try:
                    response = llm_with_tools.invoke(messages + new_messages)
                    break
                except Exception as exc:
                    if "429" in str(exc) and attempt < _MAX_RETRIES - 1:
                        match = re.search(r"retry.*?(\d+)", str(exc), re.I)
                        wait = int(match.group(1)) + 2 if match else (30 * (2 ** attempt))
                        time.sleep(min(wait, 60))
                        continue
                    raise

            new_messages.append(response)
            if not response.tool_calls:
                break

            for tc in response.tool_calls:
                tool_fn = tool_executor.get(tc["name"])
                if tool_fn:
                    try:
                        tool_result = tool_fn.invoke(tc["args"])
                    except Exception as e:
                        tool_result = f"Tool {tc['name']} failed: {str(e)}"
                else:
                    tool_result = f"Unknown tool: {tc['name']}"

                new_messages.append(
                    ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tc["id"],
                        name=tc["name"],
                    )
                )

        return {"messages": new_messages}

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_edge(START, "agent")
    graph.add_edge("agent", END)

    checkpointer = get_checkpointer()
    return graph.compile(checkpointer=checkpointer)


def _parse_final_state(final_state) -> dict:
    """Extract response, steps, and tools from final LangGraph state."""
    response_text = ""
    steps = []
    tools_used = []

    for msg in final_state["messages"]:
        if isinstance(msg, AIMessage):
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    tools_used.append(tc["name"])
                    steps.append({
                        "type": "tool_call",
                        "content": f"Calling tool: {tc['name']}\nArguments: {tc['args']}",
                    })
            else:
                content = _safe_content(msg.content)
                steps.append({
                    "type": "agent_response",
                    "content": (content[:200] + "...") if len(content) > 200 else content,
                })
                response_text = content

        elif isinstance(msg, ToolMessage):
            tool_output = msg.content[:300] + "..." if len(msg.content) > 300 else msg.content
            steps.append({
                "type": "observation",
                "content": f"Tool '{msg.name}' returned:\n{tool_output}",
            })

    if not response_text:
        response_text = "I'm sorry, I could not generate a response. Please try again."

    # Deduplicate tool names while preserving call order.
    seen: set = set()
    unique_tools = []
    for t in tools_used:
        if t not in seen:
            seen.add(t)
            unique_tools.append(t)

    return {
        "response": response_text,
        "steps": steps,
        "tools_used": unique_tools,
        "cache_hit": False,
    }


def _build_input_message(user_message: str, image_b64: str = None, genomic_csv: str = None) -> HumanMessage:
    """Build the HumanMessage.

    - Image is sent as a proper multimodal content part so Gemini can see it.
    - CSV is referenced (not inlined) since tools access it from shared state.
    - Raw data is NOT dumped into the text; tools read it via session state.
    """
    text = user_message
    if genomic_csv:
        text += "\n\n[A gene expression CSV file has been uploaded for analysis.]"
    if image_b64:
        return HumanMessage(content=[
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_b64}"},
        ])
    return HumanMessage(content=text)


def run_agent(agent_graph, user_message: str, thread_id: str, image_b64: str = None, genomic_csv: str = None) -> dict:
    """Run one synchronous agent turn. Returns result dict."""
    from agent.memory import make_run_config
    from agent.cache import get_cached_response, store_response
    from tools.onco_tools import set_session_image, set_session_csv, clear_session_data

    if not image_b64 and not genomic_csv:
        cached = get_cached_response(user_message)
        if cached is not None:
            return cached

    # Make uploaded data available to tools via shared session state.
    set_session_image(image_b64)
    set_session_csv(genomic_csv)

    input_message = _build_input_message(user_message, image_b64, genomic_csv)
    config = make_run_config(thread_id)

    try:
        for attempt in range(_MAX_RETRIES):
            try:
                final_state = agent_graph.invoke(
                    {"messages": [input_message]},
                    config=config,
                )
                break
            except Exception as exc:
                error_str = str(exc)
                if "429" in error_str and attempt < _MAX_RETRIES - 1:
                    match = re.search(r"retry.*?(\d+)\.?\d*\s*s", error_str, re.I)
                    wait = int(match.group(1)) + 2 if match else (30 * (2 ** attempt))
                    time.sleep(min(wait, 60))
                    continue
                raise
    finally:
        clear_session_data()

    result = _parse_final_state(final_state)

    if not image_b64 and not genomic_csv:
        store_response(user_message, result)

    return result


def stream_agent(
    agent_graph,
    user_message: str,
    thread_id: str,
    image_b64: str = None,
    genomic_csv: str = None,
):
    """Stream one agent turn. Yields status/token/done event dicts."""
    from agent.memory import make_run_config
    from agent.cache import get_cached_response, store_response
    from tools.onco_tools import set_session_image, set_session_csv, clear_session_data

    # Cache check.
    if not image_b64 and not genomic_csv:
        cached = get_cached_response(user_message)
        if cached is not None:
            yield {"type": "done", **cached, "cache_hit": True}
            return

    # Make uploaded data available to tools via shared session state.
    set_session_image(image_b64)
    set_session_csv(genomic_csv)

    input_message = _build_input_message(user_message, image_b64, genomic_csv)
    config = make_run_config(thread_id)

    full_response = ""
    steps: list = []
    tools_used: list = []
    status_emitted = False

    try:
        for update in agent_graph.stream(
            {"messages": [input_message]},
            config=config,
            stream_mode="updates",
        ):
            for node_name, node_output in update.items():
                if not status_emitted:
                    yield {"type": "status", "content": "Thinking..."}
                    status_emitted = True

                for msg in node_output.get("messages", []):
                    # Tool calls → transparency panel.
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                            args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
                            if name:
                                tools_used.append(name)
                                steps.append({
                                    "type": "tool_call",
                                    "content": f"Calling tool: {name}\nArguments: {args}",
                                })
                                yield {"type": "status", "content": f"Using {name}..."}

                    # Tool results → transparency panel.
                    if isinstance(msg, ToolMessage):
                        raw = msg.content if isinstance(msg.content, str) else _safe_content(msg.content)
                        tool_output = raw[:300] + "..." if len(raw) > 300 else raw
                        steps.append({
                            "type": "observation",
                            "content": f"Tool '{msg.name}' returned:\n{tool_output}",
                        })

                    # Final AI response (no tool calls = final answer).
                    if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
                        text = _safe_content(msg.content).strip()
                        if text:
                            full_response = text
                            yield {"type": "token", "content": text}

    except Exception as exc:
        error_str = str(exc)
        if "429" in error_str or "quota" in error_str.lower():
            msg = (
                "The Gemini API daily quota has been reached. "
                "gemini-3.1-flash-lite-preview allows 500 requests/day on the free "
                "tier; with caching enabled most repeated queries use no quota. "
                "Please wait a few minutes and try again."
            )
        else:
            msg = f"Agent encountered an error: {error_str}"
        full_response = msg
        yield {"type": "token", "content": msg}

    # Fallback: if streaming produced nothing, read the final checkpoint.
    if not full_response:
        try:
            final_state = agent_graph.get_state(config)
            if final_state and final_state.values:
                for msg in reversed(final_state.values.get("messages", [])):
                    if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
                        text = _safe_content(msg.content).strip()
                        if text and len(text) > 20:
                            full_response = text
                            yield {"type": "token", "content": text}
                            break
        except Exception:
            pass

    if not full_response:
        full_response = "I'm sorry, I could not generate a response. Please try again."
        yield {"type": "token", "content": full_response}

    # Deduplicate tools.
    seen: set = set()
    unique_tools: list = []
    for t in tools_used:
        if t not in seen:
            seen.add(t)
            unique_tools.append(t)

    result = {
        "response": full_response,
        "steps": steps,
        "tools_used": unique_tools,
        "cache_hit": False,
    }

    if not image_b64 and not genomic_csv:
        store_response(user_message, result)

    clear_session_data()
    yield {"type": "done", **result}

# agent/onco_agent.py
#
# Fallback single-model ReAct agent.
#
# The MAIN application uses the 5-role multi-agent supervisor in
# agent/supervisor.py. This file is retained as a simpler fallback
# (useful for development, debugging, or environments where the full
# supervisor is overkill).
#
# To swap this back in: edit app.py's load_agent() to call build_agent()
# and run_agent() instead of build_supervisor() / run_supervisor().
#
# Architecture (single model, manual tool loop, no ToolNode):
#   User query → gemini-3.1-flash-lite-preview (thinking disabled)
#              → decides which tools to call
#              → manual executor loop (avoids thought_signature gRPC crash)
#              → final answer

import os
import re
import time
from typing import Annotated

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from tools.onco_tools import (
    oncology_rag_search,
    analyze_medical_image,
    generate_pathway_diagram,
)
from tools.external_tools import (
    search_clinical_trials,
    fetch_pubmed_abstracts,
    summarize_arxiv_paper,
)
from agent.memory import get_checkpointer

# Single model — matches the supervisor for consistency.
MODEL = "gemini-3.1-flash-lite-preview"
THINKING_OFF = {"thinking_config": {"thinking_mode": "DISABLED"}}

SYSTEM_PROMPT = """
You are OncoBot, an AI assistant specialized exclusively in oncology and cancer medicine.

Tools available to you:
1. oncology_rag_search    - Search the local knowledge base for factual oncology information.
2. fetch_pubmed_abstracts - Use for "latest research" or recent study requests.
3. search_clinical_trials - Use for clinical trial queries.
4. generate_pathway_diagram - Use when the user wants a visual diagram or flowchart.
5. analyze_medical_image  - Use only when an image has been provided.
6. summarize_arxiv_paper  - Use when the user mentions a specific arXiv paper ID.

Rules:
- DOMAIN: If the question is unrelated to cancer or oncology, politely decline.
- CITATIONS: Cite source document names from RAG results in your response.
- SAFETY: End every medical information response with:
  "This information is provided for educational purposes only and is not a substitute
  for professional medical advice. Please consult your oncologist."
- TONE: If the user sounds distressed, lead with empathy before medical information.
- LANGUAGE: Respond in the same language the user wrote in.
- VISUALIZATION: Wrap DOT code in triple backticks with the 'dot' tag: ```dot ... ```
"""


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def build_agent():
    """
    Build the fallback single-model agent.

    Uses gemini-3.1-flash-lite-preview with thinking disabled and a manual
    tool loop. Not used by default — see agent/supervisor.py.

    Returns:
        Compiled LangGraph StateGraph with MemorySaver checkpointing.

    Raises:
        EnvironmentError: If GEMINI_API_KEY is not set.
    """
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
        generate_pathway_diagram,
        analyze_medical_image,
        summarize_arxiv_paper,
    ]
    llm_with_tools = llm.bind_tools(all_tools)
    tool_executor = {t.name: t for t in all_tools}

    def agent_node(state: AgentState) -> dict:
        from langchain_core.messages import ToolMessage

        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        new_messages = []

        MAX_TOOL_ITERATIONS = 5
        MAX_RETRIES = 3

        for _ in range(MAX_TOOL_ITERATIONS):
            for attempt in range(MAX_RETRIES):
                try:
                    response = llm_with_tools.invoke(messages + new_messages)
                    break
                except Exception as exc:
                    if "429" in str(exc) and attempt < MAX_RETRIES - 1:
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


def run_agent(agent_graph, user_message: str, thread_id: str, image_b64: str = None) -> dict:
    """
    Run one turn of the fallback single-model agent.
    Same return interface as run_supervisor() in agent/supervisor.py.
    """
    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
    from agent.memory import make_run_config
    from agent.cache import get_cached_response, store_response

    if not image_b64:
        cached = get_cached_response(user_message)
        if cached is not None:
            return cached

    if image_b64:
        input_message = HumanMessage(content=[
            {"type": "text", "text": f"{user_message}\n\n[IMAGE_DATA_BASE64]: {image_b64}"}
        ])
    else:
        input_message = HumanMessage(content=user_message)

    config = make_run_config(thread_id)

    MAX_RETRIES = 3
    for attempt in range(MAX_RETRIES):
        try:
            final_state = agent_graph.invoke(
                {"messages": [input_message]},
                config=config,
            )
            break
        except Exception as exc:
            error_str = str(exc)
            if "429" in error_str and attempt < MAX_RETRIES - 1:
                match = re.search(r"retry.*?(\d+)\.?\d*\s*s", error_str, re.I)
                wait = int(match.group(1)) + 2 if match else (30 * (2 ** attempt))
                time.sleep(min(wait, 60))
                continue
            raise

    response_text = ""
    graph_dot = None
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
                content = (
                    msg.content if isinstance(msg.content, str)
                    else " ".join(p.get("text", "") for p in msg.content if isinstance(p, dict))
                )
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

    if "```dot" in response_text:
        parts = response_text.split("```dot")
        response_text = parts[0].strip()
        raw_dot = parts[1].split("```")[0].strip()
        graph_dot = raw_dot

    seen: set = set()
    unique_tools = []
    for t in tools_used:
        if t not in seen:
            seen.add(t)
            unique_tools.append(t)

    result = {
        "response": response_text,
        "graph_dot": graph_dot,
        "steps": steps,
        "tools_used": unique_tools,
        "cache_hit": False,
    }

    if not image_b64:
        store_response(user_message, result)

    return result

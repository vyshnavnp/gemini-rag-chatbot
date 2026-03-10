# agent/onco_agent.py
#
# This is the core of the agentic upgrade. It replaces the simple one-shot
# rag_chain from the original app.py with a LangGraph ReAct agent.
#
# What changed from the old approach:
#   OLD: query -> retriever(k=4) -> prompt template -> LLM -> response
#   NEW: query -> agent reasons -> calls one or more tools -> synthesizes -> response
#
# The ReAct (Reasoning + Acting) pattern works like this:
#   1. The LLM receives the user query and sees the list of available tools.
#   2. It produces a "Thought" and decides which tool to call (if any).
#   3. The tool runs and returns an "Observation".
#   4. The LLM receives the observation and thinks again.
#   5. This loop continues until the LLM produces a Final Answer.
#
# LangGraph represents this as a state machine (graph):
#
#   [START] --> [agent node] --> (tool call?) --> [tool node] --> [agent node]
#                                     |
#                                  (no tool call)
#                                     |
#                                  [END]
#
# The "agent node" is the LLM with tool-calling. The "tool node" executes
# whichever tool the LLM requested and returns the result back to the LLM.
# LangGraph handles the state (message list) automatically between each step.

import os
import re
import time
from typing import Annotated

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from tools.onco_tools import (
    oncology_rag_search,
    analyze_medical_image,
    generate_pathway_diagram,
    get_sentiment_tone,
)
from tools.external_tools import (
    search_clinical_trials,
    fetch_pubmed_abstracts,
    summarize_arxiv_paper,
)
from agent.memory import get_checkpointer

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
# This is what the agent "is" at all times. It sets the persona, domain
# restriction, tool usage guidelines, and safety disclaimer.
# It is injected as the first message in every conversation.

PLANNER_PROMPT = """
You are OncoBot's research planner, specialized exclusively in oncology and cancer medicine.

Your ONLY job is to gather information by calling tools. You do NOT write the final answer
to the user — a dedicated summarizer model will do that after you finish.

Your role:
- Decide which tools to call based on the user's query.
- Call them in the right order.
- Once you have enough information, output a brief internal research summary (2-3 sentences
  max) that the summarizer can use. Do NOT write a full user-facing response.

You have access to the following tools:

1. get_sentiment_tone     - Call this ONLY when the query sounds emotional or personal
                            (mentions fear, worry, diagnosis news, prognosis, grief).
                            Skip it for factual, research, or clinical queries.
2. oncology_rag_search    - Search the local knowledge base for factual oncology information.
3. fetch_pubmed_abstracts - Use this when the user asks for "latest research" or recent studies.
4. search_clinical_trials - Use this when the user asks about available trials.
5. generate_pathway_diagram - Use this when the user wants a visual diagram or flowchart.
6. analyze_medical_image  - Use this only when an image has been provided by the user.
7. summarize_arxiv_paper  - Use this when the user mentions a specific arXiv paper ID.

Domain rule: if the question is completely unrelated to cancer or oncology, output
"OUT_OF_DOMAIN" and nothing else.
"""

SUMMARIZER_PROMPT = """
You are OncoBot, an AI assistant specialized in oncology and cancer medicine.
You are the final step in a two-model pipeline. The planner model has already called
all necessary tools and gathered the research. Your job is to write the final
user-facing response based on the full conversation history and tool results above.

Rules you must follow:
- DOMAIN: If the planner indicated "OUT_OF_DOMAIN", politely decline and redirect
  the user to ask an oncology-related question. Do not answer off-topic queries.
- CITATIONS: If oncology_rag_search results are present in the context, mention
  the source document names in your response.
- SAFETY: Always end any medical information response with:
  "This information is provided for educational purposes only and is not a substitute
  for professional medical advice. Please consult your oncologist."
- TONE: If the sentiment tool returned NEGATIVE, lead with empathy and acknowledgement
  of the user's distress before providing medical information.
- LANGUAGE: Respond in the same language the user wrote in.
- VISUALIZATION: If a pathway diagram was generated, return the DOT code in your
  response wrapped in triple backticks with the 'dot' language tag: ```dot ... ```
- FORMAT: Write a clear, well-structured response. Use bullet points or numbered
  lists where appropriate. Do not repeat the tool call logs — just the final answer.
"""


# ---------------------------------------------------------------------------
# Agent state
# ---------------------------------------------------------------------------
# LangGraph uses a typed state dict to pass information between graph nodes.
# The 'messages' field is the conversation history. The add_messages reducer
# is a LangGraph built-in that appends new messages instead of overwriting.

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_agent():
    """
    Build and compile the two-model LangGraph agent graph.

    Architecture:
        User query
            ↓
        planner_node  (gemini-2.5-flash, bound with all tools)
            ↓  calls tools as needed, loops back after each tool result
        tool_node     (LangGraph ToolNode executes the requested tool)
            ↓  returns to planner_node with observation
        planner_node  (decides: call more tools, or done?)
            ↓  done (no more tool calls)
        summarizer_node  (gemini-3.1-flash-lite-preview, NO tools)
            ↓  reads full conversation + all tool results, writes final answer
        END

    Why two models:
        - gemini-2.5-flash handles all tool-calling. Tool calling on Gemini 3.x
          preview models triggers a thought_signature error because their
          thinking mode generates internal tokens that the current LangChain
          gRPC transport does not echo back correctly.
        - gemini-3.1-flash-lite-preview never calls tools, so it has no
          thought_signature issue, and its higher free-tier quota (500 RPD vs
          20 RPD) applies to the synthesis step only.

    Returns:
        A compiled LangGraph StateGraph with MemorySaver checkpointing.

    Raises:
        EnvironmentError: If GEMINI_API_KEY is not set.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. "
            "Set it as an environment variable or in .streamlit/secrets.toml"
        )

    # --- Model 1: Planner (gemini-2.5-flash) ---
    # Handles all tool calls. Low temperature for deterministic tool selection.
    planner_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        api_key=api_key,
        temperature=0.2,
    )

    # --- Model 2: Summarizer (gemini-3.1-flash-lite-preview) ---
    # Only synthesizes text from tool results. Never calls tools.
    # Slightly higher temperature for more natural prose.
    summarizer_llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite-preview",
        api_key=api_key,
        temperature=0.4,
    )

    all_tools = [
        get_sentiment_tone,
        oncology_rag_search,
        fetch_pubmed_abstracts,
        search_clinical_trials,
        generate_pathway_diagram,
        analyze_medical_image,
        summarize_arxiv_paper,
    ]
    planner_with_tools = planner_llm.bind_tools(all_tools)

    # --- Node: planner ---
    def planner_node(state: AgentState) -> dict:
        """
        Runs gemini-2.5-flash with tools bound.

        Reads the current conversation history and either:
        - Returns an AIMessage with tool_calls (triggers the tool node), or
        - Returns a plain AIMessage (research complete; triggers summarizer).

        The planner's final plain message is a brief internal summary of what
        was found -- it is NOT shown to the user directly.
        """
        messages = [SystemMessage(content=PLANNER_PROMPT)] + state["messages"]
        response = planner_with_tools.invoke(messages)
        return {"messages": [response]}

    # --- Node: tools ---
    # LangGraph prebuilt: reads tool_calls from the last AIMessage, executes
    # the requested tool functions, and appends ToolMessage results to state.
    tool_node = ToolNode(tools=all_tools)

    # --- Node: summarizer ---
    def summarizer_node(state: AgentState) -> dict:
        """
        Runs gemini-3.1-flash-lite-preview WITHOUT any tools bound.

        Receives the full message history (user query + all tool observations +
        planner's research summary) and writes the final user-facing answer.

        Because no tools are bound, this model never triggers a tool call and
        therefore never encounters the thought_signature error.
        """
        messages = [SystemMessage(content=SUMMARIZER_PROMPT)] + state["messages"]
        response = summarizer_llm.invoke(messages)
        return {"messages": [response]}

    # --- Routing condition ---
    def route_planner(state: AgentState) -> str:
        """
        After the planner node runs, decide where to go next:
          - If the last message has tool_calls  -> execute them ("tools")
          - If the last message has no tool_calls -> research done ("summarizer")
        """
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return "summarizer"

    # --- Graph assembly ---
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner_node)
    graph.add_node("tools", tool_node)
    graph.add_node("summarizer", summarizer_node)

    graph.add_edge(START, "planner")
    graph.add_conditional_edges("planner", route_planner)
    graph.add_edge("tools", "planner")   # Tool results loop back to planner
    graph.add_edge("summarizer", END)    # Summarizer output is always final

    checkpointer = get_checkpointer()
    return graph.compile(checkpointer=checkpointer)


def run_agent(agent_graph, user_message: str, thread_id: str, image_b64: str = None) -> dict:
    """
    Run one turn of the agent conversation.

    This is the main entry point called by app.py on each user message.
    It handles injecting the image into the message if one was uploaded,
    invokes the compiled agent graph, and returns parsed results for the UI.

    Args:
        agent_graph: The compiled LangGraph graph returned by build_agent().
        user_message: The user's text query.
        thread_id: The session identifier for memory continuity.
                   Pass st.session_state["thread_id"] from the app.
        image_b64:  Optional base64-encoded image string. If provided, the
                    message is augmented to indicate an image was uploaded,
                    and analyze_medical_image can use it.

    Returns:
        A dict with the following keys:
            "response"     : str  - The final text response for the chat UI.
            "graph_dot"    : str or None - DOT code for visualization, if any.
            "steps"        : list[dict] - The reasoning steps for the
                             transparency panel. Each step has "type" and
                             "content" keys.
            "tools_used"   : list[str] - Names of tools that were called.
    """
    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
    from agent.memory import make_run_config
    from agent.cache import get_cached_response, store_response

    # ---------------------------------------------------------------------------
    # Cache check: skip the LLM entirely if we have a similar previous answer.
    # Images always bypass the cache since the answer depends on visual content.
    # ---------------------------------------------------------------------------
    if not image_b64:
        cached = get_cached_response(user_message)
        if cached is not None:
            return cached

    # If an image was uploaded, include the base64 string in the message
    # so the agent can pass it to analyze_medical_image if needed.
    if image_b64:
        input_message = HumanMessage(content=[
            {
                "type": "text",
                "text": (
                    f"{user_message}\n\n"
                    f"[IMAGE_DATA_BASE64]: {image_b64}"
                )
            }
        ])
    else:
        input_message = HumanMessage(content=user_message)

    config = make_run_config(thread_id)

    # ---------------------------------------------------------------------------
    # Invoke with automatic retry on 429 rate-limit responses.
    # The free tier allows 20 requests/day. When the limit is hit, the API
    # returns a 429 with a retry_delay field. We honour that delay (up to
    # 60 s) and retry up to 3 times before propagating the error.
    # ---------------------------------------------------------------------------
    MAX_RETRIES = 3
    for attempt in range(MAX_RETRIES):
        try:
            final_state = agent_graph.invoke(
                {"messages": [input_message]},
                config=config
            )
            break  # Success — exit the retry loop.
        except Exception as exc:
            error_str = str(exc)
            # Check for a 429 quota error from the Gemini API.
            if "429" in error_str and attempt < MAX_RETRIES - 1:
                # Extract the suggested retry delay from the error message if
                # present, otherwise fall back to 30 s exponential backoff.
                match = re.search(r"retry.*?(\d+)\.?\d*\s*s", error_str, re.I)
                wait = int(match.group(1)) + 2 if match else (30 * (2 ** attempt))
                wait = min(wait, 60)  # Cap at 60 s to avoid hanging the UI.
                time.sleep(wait)
                continue
            # For any other error (or final retry exhausted), re-raise so
            # app.py can catch it and display a clean message.
            raise

    # --- Parse the output ---
    # Walk through all messages produced in this turn to extract:
    # - The final text response (the summarizer's message — always the last AIMessage
    #   without tool_calls, produced by gemini-3.1-flash-lite-preview)
    # - Any DOT diagram code embedded in the summarizer's response
    # - Reasoning steps for the transparency panel
    # - Which tools were called by the planner
    #
    # Message ordering in the final state:
    #   HumanMessage (user query)
    #   AIMessage with tool_calls  (planner calls tools)  ← labelled "planner"
    #   ToolMessage(s)             (tool results)          ← labelled "observation"
    #   ... (more planner + tool rounds as needed) ...
    #   AIMessage without tool_calls (planner research summary) ← labelled "planner_summary"
    #   AIMessage without tool_calls (summarizer final answer)  ← this becomes response_text

    response_text = ""
    graph_dot = None
    steps = []
    tools_used = []

    # Collect all AIMessages without tool_calls in order. The second-to-last is
    # the planner's research summary; the last is the summarizer's final answer.
    plain_ai_messages = []

    for msg in final_state["messages"]:
        if isinstance(msg, AIMessage):
            if msg.tool_calls:
                # Planner is calling tools — record each call for the UI panel.
                for tc in msg.tool_calls:
                    tool_name = tc["name"]
                    tools_used.append(tool_name)
                    steps.append({
                        "type": "tool_call",
                        "content": f"Calling tool: {tool_name}\nArguments: {tc['args']}"
                    })
            else:
                plain_ai_messages.append(msg)

        elif isinstance(msg, ToolMessage):
            tool_output = msg.content[:300] + "..." if len(msg.content) > 300 else msg.content
            steps.append({
                "type": "observation",
                "content": f"Tool '{msg.name}' returned:\n{tool_output}"
            })

    # Post-loop: extract planner research summary and summarizer final answer.
    # plain_ai_messages list (AIMessages without tool_calls), in order:
    #   [-2]  = planner's internal research summary (shown in transparency panel)
    #   [-1]  = summarizer's final user-facing answer  (shown in chat)
    if len(plain_ai_messages) >= 2:
        planner_raw = plain_ai_messages[-2].content
        if not isinstance(planner_raw, str):
            planner_raw = " ".join(
                p.get("text", "") for p in planner_raw if isinstance(p, dict)
            )
        steps.append({
            "type": "planner_summary",
            "content": (planner_raw[:200] + "...") if len(planner_raw) > 200 else planner_raw,
        })

    if plain_ai_messages:
        last_content = plain_ai_messages[-1].content
        response_text = (
            last_content
            if isinstance(last_content, str)
            else " ".join(p.get("text", "") for p in last_content if isinstance(p, dict))
        )
        steps.append({
            "type": "final_answer",
            "content": (response_text[:200] + "...") if len(response_text) > 200 else response_text,
        })
    else:
        response_text = "I'm sorry, I could not generate a response. Please try again."

    # Extract DOT diagram code if the response contains a ```dot block.
    if "```dot" in response_text:
        parts = response_text.split("```dot")
        response_text = parts[0].strip()
        raw_dot = parts[1].split("```")[0].strip()
        graph_dot = raw_dot

    # Remove duplicate tool names while preserving order.
    seen = set()
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

    # Persist this response so future similar queries can be served from cache.
    if not image_b64:
        store_response(user_message, result)

    return result

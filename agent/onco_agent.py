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
from langgraph.prebuilt import ToolNode, tools_condition
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

SYSTEM_PROMPT = """
You are OncoBot, an AI assistant specialized exclusively in oncology and cancer medicine.

Your role is to assist cancer patients, caregivers, and medical researchers with:
- Understanding cancer types, stages, and symptoms
- Explaining treatment options: chemotherapy, immunotherapy, radiation, targeted therapy, surgery
- Describing drug mechanisms, side effects, and interactions
- Summarizing clinical trial opportunities
- Explaining biological pathways and tumor biology
- Providing emotional support and clear explanations when users are distressed

You have access to the following tools. Use them in order of preference:

1. get_sentiment_tone     - Call this ONLY when the query sounds emotional or personal
                            (mentions fear, worry, diagnosis news, prognosis, grief).
                            Skip it for factual, research, or clinical queries.
2. oncology_rag_search    - Search the local knowledge base for factual oncology information.
3. fetch_pubmed_abstracts - Use this when the user asks for "latest research" or recent studies.
4. search_clinical_trials - Use this when the user asks about available trials.
5. generate_pathway_diagram - Use this when the user wants a visual diagram or flowchart.
6. analyze_medical_image  - Use this only when an image has been provided by the user.
7. summarize_arxiv_paper  - Use this when the user mentions a specific arXiv paper ID.

Rules you must follow:
- DOMAIN: If the question is not related to cancer, oncology, or closely related medicine,
  politely decline and redirect the user to ask an oncology-related question.
- CITATIONS: When you use oncology_rag_search, mention the source names in your response.
- SAFETY: Always end any medical information response with:
  "This information is provided for educational purposes only and is not a substitute
  for professional medical advice. Please consult your oncologist."
- TONE: When a query sounds emotional or personal, call get_sentiment_tone first and adjust
  your response accordingly. For clinical, research, or factual queries, use a professional
  tone without calling get_sentiment_tone.
- LANGUAGE: Always respond in the same language the user wrote in.
- VISUALIZATION: If you generate a pathway diagram, return the DOT code wrapped in
  triple backticks with the 'dot' language tag: ```dot ... ```
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
    Build and compile the LangGraph ReAct agent graph.

    This function should be called once at app startup (Streamlit caches it
    with @st.cache_resource). Calling it multiple times is harmless but
    wasteful since it loads the LLM and wires up the full state graph.

    The returned compiled graph has the same interface as a LangChain
    runnable: call .invoke(input, config) or .stream(input, config).

    Returns:
        A compiled LangGraph StateGraph (CompiledGraph object) with
        MemorySaver-based checkpointing enabled.

    Raises:
        EnvironmentError: If GEMINI_API_KEY is not set.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. "
            "Set it as an environment variable or in .streamlit/secrets.toml"
        )

    # The LLM needs to support tool calling. Gemini 2.5 Flash supports it.
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        api_key=api_key,
        temperature=0.3,   # Low temperature for factual medical responses.
    )

    # Register all tools with the LLM. This makes the LLM aware of what
    # tools exist and what their arguments are (from the @tool docstrings).
    all_tools = [
        get_sentiment_tone,
        oncology_rag_search,
        fetch_pubmed_abstracts,
        search_clinical_trials,
        generate_pathway_diagram,
        analyze_medical_image,
        summarize_arxiv_paper,
    ]
    llm_with_tools = llm.bind_tools(all_tools)

    # --- Node definitions ---

    def agent_node(state: AgentState) -> dict:
        """
        The agent node runs the LLM with the current message history.
        It either produces a final text response or requests a tool call.

        The system prompt is prepended to every call to ensure the LLM
        always operates within its oncology domain constraints.

        Args:
            state: Current graph state containing the message history.

        Returns:
            A dict with 'messages' containing the LLM's response message.
            If the LLM wants to call a tool, the response will be an
            AIMessage with tool_calls populated.
        """
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    # ToolNode is a LangGraph prebuilt that:
    # 1. Reads the tool_calls from the last AIMessage in state["messages"]
    # 2. Executes each requested tool function with the given arguments
    # 3. Returns ToolMessage(s) with the results
    tool_node = ToolNode(tools=all_tools)

    # --- Graph assembly ---

    graph = StateGraph(AgentState)

    # Add the two nodes
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    # The graph starts at the agent node
    graph.add_edge(START, "agent")

    # After the agent node, use 'tools_condition' to decide:
    #   - If the last message has tool_calls -> go to "tools"
    #   - If not (final answer) -> go to END
    graph.add_conditional_edges("agent", tools_condition)

    # After tools run, always go back to the agent so it can reason
    # about the tool output and either call more tools or give a final answer.
    graph.add_edge("tools", "agent")

    # Compile with memory checkpointing enabled.
    checkpointer = get_checkpointer()
    compiled_graph = graph.compile(checkpointer=checkpointer)

    return compiled_graph


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
    # - The final text response (last AIMessage with no tool_calls)
    # - Any DOT diagram code
    # - Reasoning steps for the transparency panel
    # - Which tools were called

    response_text = ""
    graph_dot = None
    steps = []
    tools_used = []

    for msg in final_state["messages"]:
        if isinstance(msg, AIMessage):
            # If this AIMessage has tool calls, it is a "thinking" step.
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_name = tc["name"]
                    tool_args = tc["args"]
                    tools_used.append(tool_name)
                    steps.append({
                        "type": "tool_call",
                        "content": f"Calling tool: {tool_name}\nArguments: {tool_args}"
                    })
            else:
                # This is the final response.
                raw = msg.content if isinstance(msg.content, str) else str(msg.content)
                steps.append({
                    "type": "final_answer",
                    "content": raw[:200] + "..." if len(raw) > 200 else raw
                })
                response_text = raw

        elif isinstance(msg, ToolMessage):
            # Record the tool's output as an observation step.
            tool_output = msg.content[:300] + "..." if len(msg.content) > 300 else msg.content
            steps.append({
                "type": "observation",
                "content": f"Tool '{msg.name}' returned:\n{tool_output}"
            })

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

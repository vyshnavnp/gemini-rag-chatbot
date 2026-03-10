# agent/supervisor.py
#
# Full 5-role multi-agent system built on LangGraph.
#
# Roles:
#   1. PLANNER      - Decomposes the user query into an ordered task list
#   2. COORDINATOR  - Routes tasks to the right specialist; declares FINISH
#   3. TOOL-USERS   - Three specialists (research, clinical, support), each
#                     with their own tool set and system prompt
#   4. EXECUTOR     - Manual tool loop inside each specialist (no ToolNode,
#                     avoids thought_signature issues with Gemini 3.x models)
#   5. CRITIC       - Reviews the final answer for safety, accuracy, citations
#
# Graph flow:
#   START → planner → coordinator → specialist → coordinator (loop)
#                                 → FINISH → critic
#                                            → END            (approved)
#                                            → coordinator    (revise, max 2x)
#
# Single model throughout: gemini-3.1-flash-lite-preview with thinking disabled.
# thinking_mode DISABLED prevents the thought_signature gRPC error that would
# otherwise crash when this model calls tools through LangChain.
# Free tier quota: 500 requests/day (vs 20 RPD for gemini-2.5-flash).

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
    generate_pathway_diagram,
)
from tools.external_tools import (
    search_clinical_trials,
    fetch_pubmed_abstracts,
    summarize_arxiv_paper,
)
from agent.memory import get_checkpointer

# ---------------------------------------------------------------------------
# Single model constant shared across all 5 roles.
# ---------------------------------------------------------------------------
MODEL = "gemini-3.1-flash-lite-preview"
THINKING_OFF = {"thinking_config": {"thinking_mode": "DISABLED"}}
# Planner, Coordinator, and Critic do NOT call tools, so the thought_signature
# gRPC issue cannot occur there.  We omit a thinking_config for those nodes,
# letting the model use its native reasoning capability (effectively AUTO).
# Only tool-using specialist nodes must keep thinking DISABLED.


def _safe_content(content) -> str:
    """
    Safely extract a plain string from an AIMessage.content.

    Gemini (and some other providers) return content as either a plain ``str``
    or a list of typed parts, e.g. ``[{"type": "text", "text": "..."}]``.
    Calling ``.strip()`` on the raw value crashes with
    ``'list' object has no attribute 'strip'`` when the list form is returned.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict)
        )
    return str(content)


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
# All five roles operate on the same shared state dict. LangGraph's
# add_messages reducer appends new messages rather than overwriting.

class OncoBotState(TypedDict):
    messages: Annotated[list, add_messages]  # Full conversation history
    next_agent: str       # Coordinator's routing decision
    plan: str             # Planner's task decomposition (not in messages)
    critic_feedback: str  # "APPROVED" or "REVISE: <reason>"
    revision_count: int   # Guard against infinite critic→revise loops


# ---------------------------------------------------------------------------
# Role 1: PLANNER prompt
# ---------------------------------------------------------------------------

PLANNER_PROMPT = """
You are the OncoBot Planner. Analyze the user's query and produce a brief ordered
task plan for the coordinator. Output ONLY the plan — nothing else.

Available specialists:
- research_agent: factual oncology information, PubMed searches, arXiv papers,
  biological pathway diagrams
- clinical_agent: clinical trials, treatment options, medical image analysis
- support_agent: emotional or distressed patients, compassionate patient-facing responses

Output format (use exactly this structure):
1. <specialist_name>: <what to do>
2. <specialist_name>: <what to do>   ← only include if a second specialist is needed

If the query is completely unrelated to oncology or cancer medicine,
output exactly: OUT_OF_DOMAIN
"""

# ---------------------------------------------------------------------------
# Role 2: COORDINATOR prompt
# ---------------------------------------------------------------------------

COORDINATOR_PROMPT = """
You are the OncoBot Coordinator. You read the task plan and the current conversation
history, then route to the appropriate specialist or declare the work done.

Task plan: {plan}
Critic feedback: {critic_feedback}

Rules:
- Route one specialist at a time, following the plan order.
- Once all planned tasks have specialist responses, output FINISH.
- If critic feedback says REVISE, route to the specialist best able to fix the issue.
- Never route the same specialist more than twice in one turn.
- Respond with ONLY one of: research_agent, clinical_agent, support_agent, FINISH
"""

# ---------------------------------------------------------------------------
# Role 3: Specialist system prompts (TOOL-USER role)
# ---------------------------------------------------------------------------

RESEARCH_AGENT_PROMPT = """
You are the Research Specialist for OncoBot. Find and synthesize factual oncology
information using your tools.

Tools available:
- oncology_rag_search: search the local knowledge base (MedQuAD + arXiv papers)
- fetch_pubmed_abstracts: find recent peer-reviewed research on PubMed
- summarize_arxiv_paper: retrieve a specific arXiv paper by ID
- generate_pathway_diagram: create a biological pathway visualization in DOT format

Strategy:
1. Always call oncology_rag_search first for factual questions.
2. Also call fetch_pubmed_abstracts if the user asks for "latest" or "recent" research.
3. Always cite the source filenames from the RAG results in your response.
4. If a diagram is requested, call generate_pathway_diagram and wrap the output
   in triple backticks with the 'dot' tag: ```dot ... ```
"""

CLINICAL_AGENT_PROMPT = """
You are the Clinical Specialist for OncoBot. Help users understand treatments and trials.

Tools available:
- search_clinical_trials: find recruiting trials on ClinicalTrials.gov
- oncology_rag_search: search the knowledge base for treatment information
- analyze_medical_image: analyze an uploaded scan or pathology slide with Gemini Vision
- generate_pathway_diagram: visualize treatment pathways

Strategy:
1. For treatment questions: call oncology_rag_search first, then search_clinical_trials.
2. For medical image questions: call analyze_medical_image with the question and image.
3. Always mention standard of care before experimental options.
4. Always remind users to consult their oncologist.
"""

SUPPORT_AGENT_PROMPT = """
You are the Patient Support Specialist for OncoBot. Provide compassionate, clear
responses to patients and caregivers who may be anxious or overwhelmed.

Tools available:
- oncology_rag_search: find patient-friendly information
- search_clinical_trials: help patients understand their options

Strategy:
1. ALWAYS lead with empathy. Acknowledge feelings before providing information.
2. If the query contains words like "scared", "worried", "diagnosed", "afraid",
   or describes a personal medical situation — open with warmth and compassion.
3. Use plain language. Avoid heavy medical jargon.
4. After empathetic acknowledgement, provide clear, helpful information.
5. End with encouragement and a reminder that their medical team can best guide them.
"""

# ---------------------------------------------------------------------------
# Role 5: CRITIC prompt
# ---------------------------------------------------------------------------

CRITIC_PROMPT = """
You are the OncoBot Critic. Review the most recent specialist response against
these quality checks before it reaches the user.

Checks:
1. SAFETY DISCLAIMER: Does it end with a note about "educational purposes only" and
   advising the user to consult their oncologist?
   (Required for any medical information response.)
2. CITATIONS: If oncology_rag_search was called, are source filenames cited?
3. TONE: If the original query was from a distressed patient, is the response empathetic?
4. ACCURACY: Does the response actually address the user's question?
5. COMPLETENESS: Are all parts of the user's question answered?

If ALL checks pass:
  Output the final polished response for the user. You may lightly edit for clarity
  and flow, but do not change factual content or remove citations/disclaimers.

If ANY check fails:
  Output EXACTLY: REVISE: <one sentence describing what is missing or wrong>
  Nothing else.
"""

# ---------------------------------------------------------------------------
# Role 4: EXECUTOR — specialist manual tool loop builder
# ---------------------------------------------------------------------------

_MAX_TOOL_ITERATIONS = 5
_MAX_RETRIES = 3


def _build_specialist(llm, tools: list, system_prompt: str, agent_name: str):
    """
    Build a specialist LangGraph node (combines Role 3 TOOL-USER + Role 4 EXECUTOR).

    Each specialist:
      - Binds its own specific tool set to the LLM
      - Runs a manual ReAct loop instead of using LangGraph's ToolNode
        (avoids the thought_signature gRPC crash with Gemini 3.x models)
      - Retries automatically on 429 rate-limit errors
      - Appends all messages (tool calls, results, final answer) to shared state

    Args:
        llm: The ChatGoogleGenerativeAI instance (tools will be bound to it).
        tools: List of @tool-decorated functions available to this specialist.
        system_prompt: The specialist's role, tools, and strategy instructions.
        agent_name: Display name used in LangGraph traces and step logs.

    Returns:
        A function (state: OncoBotState) -> dict suitable for graph.add_node().
    """
    llm_with_tools = llm.bind_tools(tools)
    tool_executor = {t.name: t for t in tools}

    def specialist_node(state: OncoBotState) -> dict:
        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        new_messages = []

        for _ in range(_MAX_TOOL_ITERATIONS):
            # Invoke LLM with 429 retry (Role 4: Executor).
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

            # Execute each tool call manually (Role 4: Executor).
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

    specialist_node.__name__ = agent_name
    return specialist_node


# ---------------------------------------------------------------------------
# Role 1: Planner node builder
# ---------------------------------------------------------------------------

def _build_planner_node(llm):
    """
    Build the planner node (Role 1).

    Reads the user query and produces a task decomposition stored in
    state['plan']. The plan is NOT added to state['messages'] so it
    doesn't appear in the conversation history seen by specialists.
    """
    def planner_node(state: OncoBotState) -> dict:
        messages = [SystemMessage(content=PLANNER_PROMPT)] + state["messages"]
        response = llm.invoke(messages)
        return {"plan": _safe_content(response.content).strip()}
    return planner_node


# ---------------------------------------------------------------------------
# Role 2: Coordinator node builder + routing
# ---------------------------------------------------------------------------

def _build_coordinator_node(llm):
    """
    Build the coordinator node (Role 2).

    Reads the task plan and conversation history, then routes to the
    appropriate specialist or declares FINISH. The coordinator's routing
    decision is stored in state['next_agent'] — not in messages.
    """
    def coordinator_node(state: OncoBotState) -> dict:
        plan = state.get("plan", "")
        critic_feedback = state.get("critic_feedback", "None")
        prompt = COORDINATOR_PROMPT.format(plan=plan, critic_feedback=critic_feedback)
        messages = [SystemMessage(content=prompt)] + state["messages"]
        response = llm.invoke(messages)

        decision = _safe_content(response.content).strip().lower()
        if "research" in decision:
            next_agent = "research_agent"
        elif "clinical" in decision:
            next_agent = "clinical_agent"
        elif "support" in decision:
            next_agent = "support_agent"
        else:
            next_agent = "FINISH"

        return {"next_agent": next_agent}
    return coordinator_node


def _route_from_coordinator(state: OncoBotState) -> str:
    """
    Edge condition after the coordinator node.
    Routes to a specialist name, or to "critic" when FINISH is declared.
    """
    next_agent = state.get("next_agent", "FINISH")
    if next_agent == "FINISH":
        return "critic"
    return next_agent


# ---------------------------------------------------------------------------
# Role 5: Critic node builder + routing
# ---------------------------------------------------------------------------

def _build_critic_node(llm):
    """
    Build the critic node (Role 5).

    Reviews the latest specialist response. If all quality checks pass,
    outputs the polished final answer (this becomes the response shown
    to the user). If a check fails, outputs "REVISE: <reason>" which
    routes back to the coordinator for a revision cycle (max 2 times).
    """
    def critic_node(state: OncoBotState) -> dict:
        messages = [SystemMessage(content=CRITIC_PROMPT)] + state["messages"]
        response = llm.invoke(messages)
        content = _safe_content(response.content).strip()

        if content.upper().startswith("REVISE"):
            return {
                "messages": [response],
                "critic_feedback": content,
                "revision_count": state.get("revision_count", 0) + 1,
            }
        else:
            # Critic approved — its output IS the final user-facing answer.
            return {
                "messages": [response],
                "critic_feedback": "APPROVED",
            }
    return critic_node


def _route_from_critic(state: OncoBotState) -> str:
    """
    Edge condition after the critic node.
    Sends back to coordinator for revision (up to 2 times) or ends the graph.
    """
    feedback = state.get("critic_feedback", "APPROVED")
    revision_count = state.get("revision_count", 0)
    if feedback.upper().startswith("REVISE") and revision_count < 2:
        return "coordinator"
    return END


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------

def build_supervisor():
    """
    Build and compile the full 5-role multi-agent LangGraph graph.

    Graph structure:
        START → planner → coordinator → research_agent  → coordinator (loop)
                                      → clinical_agent  → coordinator (loop)
                                      → support_agent   → coordinator (loop)
                                      → (FINISH) → critic
                                                   → END          (approved)
                                                   → coordinator  (revise, max 2x)

    Returns:
        A compiled LangGraph StateGraph with MemorySaver checkpointing.

    Raises:
        EnvironmentError: If GEMINI_API_KEY is not set.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY is not set.")

    # Thinking-capable LLM: Planner, Coordinator, Critic — no tools bound,
    # so thought_signature is never echoed back and thinking is safe to use.
    thinker_llm = ChatGoogleGenerativeAI(
        model=MODEL,
        api_key=api_key,
        temperature=0.3,
        # No thinking_config: model applies its native reasoning (AUTO).
    )

    # Tool-user LLM: specialists — thinking MUST be disabled to prevent the
    # thought_signature gRPC crash that occurs when a thinking model issues
    # tool calls through LangChain (LangChain does not echo the signature).
    tool_llm = ChatGoogleGenerativeAI(
        model=MODEL,
        api_key=api_key,
        temperature=0.3,
        model_kwargs=THINKING_OFF,
    )

    planner_node     = _build_planner_node(thinker_llm)
    coordinator_node = _build_coordinator_node(thinker_llm)
    critic_node      = _build_critic_node(thinker_llm)

    research_node = _build_specialist(
        llm=tool_llm,
        tools=[oncology_rag_search, fetch_pubmed_abstracts, summarize_arxiv_paper, generate_pathway_diagram],
        system_prompt=RESEARCH_AGENT_PROMPT,
        agent_name="research_agent",
    )
    clinical_node = _build_specialist(
        llm=tool_llm,
        tools=[search_clinical_trials, oncology_rag_search, analyze_medical_image, generate_pathway_diagram],
        system_prompt=CLINICAL_AGENT_PROMPT,
        agent_name="clinical_agent",
    )
    support_node = _build_specialist(
        llm=tool_llm,
        tools=[oncology_rag_search, search_clinical_trials],
        system_prompt=SUPPORT_AGENT_PROMPT,
        agent_name="support_agent",
    )

    graph = StateGraph(OncoBotState)

    graph.add_node("planner",     planner_node)
    graph.add_node("coordinator", coordinator_node)
    graph.add_node("research_agent", research_node)
    graph.add_node("clinical_agent", clinical_node)
    graph.add_node("support_agent",  support_node)
    graph.add_node("critic",      critic_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "coordinator")

    graph.add_conditional_edges(
        "coordinator",
        _route_from_coordinator,
        {
            "research_agent": "research_agent",
            "clinical_agent": "clinical_agent",
            "support_agent": "support_agent",
            "critic": "critic",
        }
    )

    # Each specialist loops back to the coordinator after finishing.
    graph.add_edge("research_agent", "coordinator")
    graph.add_edge("clinical_agent", "coordinator")
    graph.add_edge("support_agent",  "coordinator")

    graph.add_conditional_edges(
        "critic",
        _route_from_critic,
        {
            "coordinator": "coordinator",
            END: END,
        }
    )

    checkpointer = get_checkpointer()
    return graph.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# run_supervisor — invocation + output parsing
# ---------------------------------------------------------------------------

def run_supervisor(
    agent_graph,
    user_message: str,
    thread_id: str,
    image_b64: str = None,
) -> dict:
    """
    Run one turn of the 5-role supervisor conversation.

    Drop-in replacement for run_agent() in onco_agent.py — app.py calls
    this with the same arguments and receives the same dict structure.

    Args:
        agent_graph: Compiled LangGraph graph from build_supervisor().
        user_message: The user's text query.
        thread_id: Session ID for MemorySaver continuity.
        image_b64: Optional base64-encoded image string.

    Returns:
        Dict with keys: response, graph_dot, steps, tools_used, cache_hit.
    """
    from agent.memory import make_run_config
    from agent.cache import get_cached_response, store_response

    # Semantic cache check — skip all LLM calls if we have a similar answer.
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

    # Invoke with top-level 429 retry (specialist nodes also retry internally).
    MAX_RETRIES = 3
    for attempt in range(MAX_RETRIES):
        try:
            final_state = agent_graph.invoke(
                {
                    "messages": [input_message],
                    "plan": "",
                    "next_agent": "",
                    "critic_feedback": "",
                    "revision_count": 0,
                },
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

    # --- Parse output ---
    # Walk all state messages to build:
    #   response_text  — the critic's final approved answer (last AIMessage)
    #   steps          — reasoning transparency panel entries
    #   tools_used     — tool names for the badge strip
    #   graph_dot      — Graphviz DOT code if any diagram was generated

    response_text = ""
    graph_dot = None
    steps = []
    tools_used = []

    for msg in final_state["messages"]:
        if isinstance(msg, AIMessage):
            if msg.tool_calls:
                # Specialist called a tool — record for the transparency panel.
                for tc in msg.tool_calls:
                    tools_used.append(tc["name"])
                    steps.append({
                        "type": "tool_call",
                        "content": f"Calling tool: {tc['name']}\nArguments: {tc['args']}",
                    })
            else:
                # Non-tool AIMessage: specialist final answer or critic output.
                content = _safe_content(msg.content)
                steps.append({
                    "type": "agent_response",
                    "content": (content[:200] + "...") if len(content) > 200 else content,
                })
                # Always update so the last one (critic's output) becomes the answer.
                response_text = content

        elif isinstance(msg, ToolMessage):
            tool_output = msg.content[:300] + "..." if len(msg.content) > 300 else msg.content
            steps.append({
                "type": "observation",
                "content": f"Tool '{msg.name}' returned:\n{tool_output}",
            })

    if not response_text:
        response_text = "I'm sorry, I could not generate a response. Please try again."

    # Extract DOT diagram code if embedded in the response.
    if "```dot" in response_text:
        parts = response_text.split("```dot")
        response_text = parts[0].strip()
        raw_dot = parts[1].split("```")[0].strip()
        graph_dot = raw_dot

    # Deduplicate tool names while preserving call order.
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


# ---------------------------------------------------------------------------
# stream_supervisor — streaming invocation for real-time Streamlit output
# ---------------------------------------------------------------------------

_STREAM_STATUS_LABELS = {
    "planner":        "Analyzing query...",
    "coordinator":    "Planning next step...",
    "research_agent": "Searching oncology literature...",
    "clinical_agent": "Checking clinical data...",
    "support_agent":  "Preparing compassionate response...",
    "critic":         "Reviewing response quality...",
}


def stream_supervisor(
    agent_graph,
    user_message: str,
    thread_id: str,
    image_b64: str = None,
):
    """
    Stream one turn of the 5-role supervisor conversation.

    Yields dicts with these shapes:
      {"type": "status",  "content": str}
          Progress label as each node activates (display as a status caption).
      {"type": "token",   "content": str}
          One token chunk from the critic's final output. Accumulate these to
          build the full response_text.
      {"type": "done",    "response": str, "graph_dot": str|None,
                          "steps": list,   "tools_used": list,
                          "cache_hit": bool}
          Always the final event; carries full result metadata for panel updates.

    For cache hits a single "done" event is yielded immediately (no tokens).
    """
    from agent.memory import make_run_config
    from agent.cache import get_cached_response, store_response

    # Semantic cache check — skip graph invocation if a similar answer exists.
    if not image_b64:
        cached = get_cached_response(user_message)
        if cached is not None:
            yield {"type": "done", **cached, "cache_hit": True}
            return

    if image_b64:
        input_message = HumanMessage(content=[
            {"type": "text", "text": f"{user_message}\n\n[IMAGE_DATA_BASE64]: {image_b64}"}
        ])
    else:
        input_message = HumanMessage(content=user_message)

    config = make_run_config(thread_id)

    full_response = ""
    graph_dot = None
    steps: list = []
    tools_used: list = []
    status_shown: set = set()

    try:
        for chunk, metadata in agent_graph.stream(
            {
                "messages": [input_message],
                "plan": "",
                "next_agent": "",
                "critic_feedback": "",
                "revision_count": 0,
            },
            config=config,
            stream_mode="messages",
        ):
            node = metadata.get("langgraph_node", "")

            # Emit a progress label once per node activation.
            if node and node not in status_shown:
                status_shown.add(node)
                label = _STREAM_STATUS_LABELS.get(node, f"{node} working...")
                yield {"type": "status", "content": label}

            # Track tool calls in the transparency panel.
            if hasattr(chunk, "tool_calls") and chunk.tool_calls:
                for tc in chunk.tool_calls:
                    name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                    args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
                    if name:
                        tools_used.append(name)
                        steps.append({
                            "type": "tool_call",
                            "content": f"Calling tool: {name}\nArguments: {args}",
                        })

            # Track tool results in the transparency panel.
            if isinstance(chunk, ToolMessage):
                raw = _safe_content(chunk.content) if not isinstance(chunk.content, str) else chunk.content
                tool_output = raw[:300] + "..." if len(raw) > 300 else raw
                steps.append({
                    "type": "observation",
                    "content": f"Tool '{chunk.name}' returned:\n{tool_output}",
                })

            # Stream critic tokens directly to the user.
            if node == "critic":
                content = getattr(chunk, "content", None)
                if content:
                    text = _safe_content(content)
                    if text:
                        full_response += text
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

    if not full_response:
        full_response = "I'm sorry, I could not generate a response. Please try again."
        yield {"type": "token", "content": full_response}

    # Extract DOT diagram code if embedded in the critic's response.
    if "```dot" in full_response:
        parts = full_response.split("```dot")
        full_response = parts[0].strip()
        raw_dot = parts[1].split("```")[0].strip()
        graph_dot = raw_dot

    # Deduplicate tool names while preserving call order.
    seen: set = set()
    unique_tools: list = []
    for t in tools_used:
        if t not in seen:
            seen.add(t)
            unique_tools.append(t)

    result = {
        "response": full_response,
        "graph_dot": graph_dot,
        "steps": steps,
        "tools_used": unique_tools,
        "cache_hit": False,
    }

    if not image_b64:
        store_response(user_message, result)

    yield {"type": "done", **result}

# agent/supervisor.py
#
# Multi-agent supervisor using LangGraph.
#
# Architecture:
#   The supervisor is a router LLM that reads the user query and decides
#   which specialist agent should handle it. Each specialist is a sub-graph
#   that carries its own tool set and system prompt.
#
#   Supervisor
#       |
#       +-- research_agent   (RAG search + PubMed + arXiv) 
#       |                     Best for: factual questions, paper lookups
#       |
#       +-- clinical_agent   (ClinicalTrials.gov + treatment info)
#       |                     Best for: trial searches, treatment options
#       |
#       +-- support_agent    (Sentiment + empathetic responses)
#                             Best for: distressed patients, general patient questions
#
# How routing works:
#   1. User message arrives at the supervisor node.
#   2. The supervisor LLM reads the message and emits a routing decision:
#      one of "research_agent", "clinical_agent", "support_agent", or "FINISH".
#   3. The appropriate specialist sub-graph is invoked.
#   4. The specialist's response is appended to the shared message state.
#   5. Control returns to the supervisor, which decides if more work is needed
#      or if the final answer is ready ("FINISH").
#
# NOTE: This module is available but is NOT used by default in app.py.
# To switch from the single ReAct agent to the supervisor, change app.py to
# call build_supervisor() instead of build_agent() in the load_agent()
# function. Everything else stays the same because both return the same
# LangGraph compiled graph interface.

import os
from typing import Annotated, Literal

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

from tools.onco_tools import (
    oncology_rag_search,
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
# Shared state
# ---------------------------------------------------------------------------
# All agents share the same message list. The supervisor appends its routing
# decision; specialists append their responses.

class SupervisorState(TypedDict):
    messages: Annotated[list, add_messages]
    # The name of the next agent to call, set by the supervisor node.
    next_agent: str


# ---------------------------------------------------------------------------
# Specialist system prompts
# ---------------------------------------------------------------------------

RESEARCH_AGENT_PROMPT = """
You are the Research Specialist for OncoBot.
Your job is to find and synthesize factual oncology information.

You have access to:
- oncology_rag_search: search the local knowledge base of medical QA pairs and papers
- fetch_pubmed_abstracts: search PubMed for recent peer-reviewed research
- summarize_arxiv_paper: look up a specific arXiv paper by ID
- generate_pathway_diagram: create a biological pathway visualization

Use oncology_rag_search first. If the user asks for latest/recent research,
also call fetch_pubmed_abstracts. Always cite your sources.
"""

CLINICAL_AGENT_PROMPT = """
You are the Clinical Specialist for OncoBot.
Your job is to help users understand treatment options and find clinical trials.

You have access to:
- search_clinical_trials: find recruiting trials on ClinicalTrials.gov
- oncology_rag_search: search the local knowledge base for treatment information
- generate_pathway_diagram: visualize treatment pathways if requested

When answering about treatments, always mention the standard of care first,
then discuss experimental options. Always remind users to consult their oncologist.
"""

SUPPORT_AGENT_PROMPT = """
You are the Patient Support Specialist for OncoBot.
Your job is to provide compassionate, clear information to patients and caregivers
who may be anxious, scared, or overwhelmed.

You have access to:
- get_sentiment_tone: confirm the emotional state of the user
- oncology_rag_search: find relevant patient-friendly information
- search_clinical_trials: help patients understand their options

Lead with empathy. Use plain language, not medical jargon. Be reassuring.
Acknowledge feelings before providing information.
"""

SUPERVISOR_PROMPT = """
You are the OncoBot supervisor. Your job is to read the user's query and route
it to the most appropriate specialist agent.

Available agents:
- research_agent: Handles factual oncology questions, paper lookups, mechanism questions,
  pathway visualizations, and anything requiring deep scientific information.
- clinical_agent: Handles questions about treatments, clinical trials, chemotherapy
  protocols, drug choices, and standard-of-care questions.
- support_agent: Handles queries from distressed patients or caregivers, general
  "I was diagnosed with..." questions, fear-based queries, and emotional support needs.

After a specialist has responded, check if the user's question is fully answered.
If yes, respond with "FINISH". If the question needs input from another specialist,
route to them. Do not route to the same specialist twice in one turn.

Respond with ONLY the agent name to route to, or "FINISH". Nothing else.
"""


# ---------------------------------------------------------------------------
# Helper: build a specialist sub-graph
# ---------------------------------------------------------------------------

def _build_specialist(
    llm: ChatGoogleGenerativeAI,
    tools: list,
    system_prompt: str,
    agent_name: str
) -> callable:
    """
    Build and return a single specialist agent as a callable function.

    Rather than building a full sub-graph for each specialist (which would
    add complexity), each specialist is a simple function that:
    1. Binds its tool set to the LLM
    2. Runs a mini ReAct loop (max 5 iterations to prevent infinite loops)
    3. Returns the final response text

    This function is used as a LangGraph node -- it receives the state dict
    and returns an updated state dict.

    Args:
        llm: The base LLM to use (tools will be bound to it).
        tools: List of @tool functions this specialist can call.
        system_prompt: The specialist's role and instructions.
        agent_name: Display name used in step logs.

    Returns:
        A function with signature (state: SupervisorState) -> dict
        that can be directly added as a LangGraph node.
    """
    llm_with_tools = llm.bind_tools(tools)
    tool_executor = {t.name: t for t in tools}

    def specialist_node(state: SupervisorState) -> dict:
        """
        Run the specialist agent for one turn.

        Executes a ReAct loop: the specialist LLM thinks and calls tools
        until it produces a final text response or hits the iteration limit.

        Args:
            state: Current shared graph state.

        Returns:
            Dict with 'messages' containing the specialist's response(s).
        """
        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        new_messages = []

        # Run the ReAct loop for a limited number of iterations.
        max_iterations = 5
        for _ in range(max_iterations):
            response = llm_with_tools.invoke(messages + new_messages)
            new_messages.append(response)

            # If no tool calls, we have a final answer.
            if not response.tool_calls:
                break

            # Execute each requested tool.
            from langchain_core.messages import ToolMessage
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

    # Rename the function so LangGraph node names are meaningful in traces.
    specialist_node.__name__ = agent_name
    return specialist_node


# ---------------------------------------------------------------------------
# Supervisor node
# ---------------------------------------------------------------------------

def _build_supervisor_node(llm: ChatGoogleGenerativeAI) -> callable:
    """
    Build the supervisor routing node.

    The supervisor reads all messages and emits a routing decision:
    the name of the next specialist to call, or "FINISH".
    The decision is written to state["next_agent"].

    Args:
        llm: The base LLM (no tools needed for the supervisor).

    Returns:
        A LangGraph node function.
    """
    def supervisor_node(state: SupervisorState) -> dict:
        """
        Decide which agent should handle the current state of the conversation.

        Reads the full message history and asks the supervisor LLM to choose
        the next agent or declare "FINISH".

        Args:
            state: Current shared graph state.

        Returns:
            Dict with 'next_agent' set to the routing decision.
        """
        messages = [SystemMessage(content=SUPERVISOR_PROMPT)] + state["messages"]
        response = llm.invoke(messages)

        decision = response.content.strip().lower()

        # Normalize to valid agent names.
        if "research" in decision:
            next_agent = "research_agent"
        elif "clinical" in decision:
            next_agent = "clinical_agent"
        elif "support" in decision:
            next_agent = "support_agent"
        else:
            next_agent = "FINISH"

        return {"next_agent": next_agent}

    return supervisor_node


def _route_from_supervisor(state: SupervisorState) -> str:
    """
    Conditional edge function that reads state["next_agent"] and returns
    the name of the next graph node to go to.

    LangGraph calls this function after the supervisor node runs to determine
    which edge to follow.

    Args:
        state: Current graph state, with 'next_agent' set by supervisor node.

    Returns:
        The name of the next node: "research_agent", "clinical_agent",
        "support_agent", or END.
    """
    next_agent = state.get("next_agent", "FINISH")
    if next_agent == "FINISH":
        return END
    return next_agent


# ---------------------------------------------------------------------------
# Public builder function
# ---------------------------------------------------------------------------

def build_supervisor():
    """
    Build and compile the multi-agent supervisor graph.

    This is an alternative to build_agent() in onco_agent.py.
    Use this when you want routed specialist agents instead of a single
    generalist agent. Swap it into app.py's load_agent() function.

    The graph structure:
        START -> supervisor -> research_agent -> supervisor -> ...
                            -> clinical_agent -> supervisor -> ...
                            -> support_agent  -> supervisor -> ...
                            -> END

    Returns:
        A compiled LangGraph StateGraph with MemorySaver checkpointing.

    Raises:
        EnvironmentError: If GEMINI_API_KEY is not set.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY is not set.")

    # Use a slightly higher temperature for the support agent's empathetic
    # responses, but keep it factual for research and clinical.
    base_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        api_key=api_key,
        temperature=0.3,
    )

    # Build specialist nodes
    research_node = _build_specialist(
        llm=base_llm,
        tools=[oncology_rag_search, fetch_pubmed_abstracts, summarize_arxiv_paper, generate_pathway_diagram],
        system_prompt=RESEARCH_AGENT_PROMPT,
        agent_name="research_agent",
    )

    clinical_node = _build_specialist(
        llm=base_llm,
        tools=[search_clinical_trials, oncology_rag_search, generate_pathway_diagram],
        system_prompt=CLINICAL_AGENT_PROMPT,
        agent_name="clinical_agent",
    )

    support_node = _build_specialist(
        llm=base_llm,
        tools=[get_sentiment_tone, oncology_rag_search, search_clinical_trials],
        system_prompt=SUPPORT_AGENT_PROMPT,
        agent_name="support_agent",
    )

    supervisor_node = _build_supervisor_node(base_llm)

    # Assemble the graph
    graph = StateGraph(SupervisorState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("research_agent", research_node)
    graph.add_node("clinical_agent", clinical_node)
    graph.add_node("support_agent", support_node)

    # Start at the supervisor
    graph.add_edge(START, "supervisor")

    # After supervisor, route conditionally based on next_agent
    graph.add_conditional_edges(
        "supervisor",
        _route_from_supervisor,
        {
            "research_agent": "research_agent",
            "clinical_agent": "clinical_agent",
            "support_agent": "support_agent",
            END: END,
        }
    )

    # After each specialist, go back to the supervisor to check if done
    graph.add_edge("research_agent", "supervisor")
    graph.add_edge("clinical_agent", "supervisor")
    graph.add_edge("support_agent", "supervisor")

    checkpointer = get_checkpointer()
    return graph.compile(checkpointer=checkpointer)

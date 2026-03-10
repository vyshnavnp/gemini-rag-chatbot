# agent/memory.py
#
# This module manages per-session conversational memory for the OncoBot agent.
#
# How it works:
#   - Each Streamlit user session gets a unique thread_id (stored in
#     st.session_state at the app layer).
#   - LangGraph's MemorySaver checkpointer stores the full message history
#     for that thread_id in memory (RAM-based, lives as long as the process).
#   - The agent passes {"configurable": {"thread_id": thread_id}} on every
#     invoke call, which makes LangGraph look up and continue the right thread.
#
# Why RAM instead of disk persistence?
#   - The EC2 instance uses Docker volumes for chroma_db and knowledge_base.
#   - Conversation memory does not need to survive container restarts -- it
#     would be inappropriate for a medical context to retain old sessions.
#   - If you want persistence across restarts, swap MemorySaver for
#     SqliteSaver("conversations.db") and add the file to the Docker volume.
#
# NOTE: MemorySaver is imported from langgraph.checkpoint.memory and is
# included in the langgraph package. No extra dependencies needed.

from langgraph.checkpoint.memory import MemorySaver


# A single shared checkpointer instance for the whole process.
# The agent graph is built once (cached by Streamlit) and this checkpointer
# is injected into it at build time. All threads share the same checkpointer
# but their message histories are isolated by thread_id.
_checkpointer = MemorySaver()


def get_checkpointer() -> MemorySaver:
    """
    Return the shared MemorySaver checkpointer.

    The agent graph is compiled with this checkpointer, which enables
    LangGraph to store and restore message state between calls using
    the thread_id passed in the run config.

    Returns:
        A MemorySaver instance shared across all sessions.
    """
    return _checkpointer


def make_run_config(thread_id: str) -> dict:
    """
    Build the LangGraph run config dict for a specific session.

    This dict is passed as the 'config' argument to agent.invoke() or
    agent.stream(). LangGraph reads the thread_id from it to load the
    correct conversation history from the checkpointer.

    Args:
        thread_id: A unique string identifying the user's session.
                   In the Streamlit app this comes from st.session_state.

    Returns:
        A dict of the form {"configurable": {"thread_id": thread_id}}.
    """
    return {"configurable": {"thread_id": thread_id}}

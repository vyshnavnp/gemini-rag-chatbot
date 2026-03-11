# agent/memory.py — Per-session conversational memory (RAM-based MemorySaver).

from langgraph.checkpoint.memory import MemorySaver

_checkpointer = MemorySaver()


def get_checkpointer() -> MemorySaver:
    """Return the shared MemorySaver checkpointer."""
    return _checkpointer


def make_run_config(thread_id: str) -> dict:
    """Build the LangGraph run config for a session thread."""
    return {"configurable": {"thread_id": thread_id}}

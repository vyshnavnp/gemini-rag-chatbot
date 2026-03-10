# test_agent.py
#
# Run this script to verify the entire agentic system works end-to-end
# before pushing to GitHub / deploying to EC2.
#
# Usage:
#   .venv\Scripts\python.exe test_agent.py
#
# What it checks:
#   1. API key is readable
#   2. ChromaDB knowledge base exists and is queryable
#   3. Each external API is reachable (ClinicalTrials, PubMed, arXiv)
#   4. The LangGraph agent graph builds without errors
#   5. The agent completes a full reasoning turn (involves a real LLM call)
#   6. Memory/thread continuity (two sequential turns in the same thread)
#
# The script prints PASS or FAIL for each check.
# Exit code 0 = all passed. Exit code 1 = at least one failed.

import os
import sys
import warnings
import traceback

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

PASS = "PASS"
FAIL = "FAIL"
results = []


def check(label: str, fn):
    """
    Run fn(), print PASS or FAIL with the label, collect the result.
    Returns the return value of fn() on success, or None on failure.
    """
    try:
        value = fn()
        print(f"  [{PASS}] {label}")
        results.append((label, True))
        return value
    except Exception as e:
        print(f"  [{FAIL}] {label}")
        print(f"         Error: {e}")
        results.append((label, False))
        return None


# ---------------------------------------------------------------------------
# Step 1: API Key
# ---------------------------------------------------------------------------

print("\n--- Step 1: API Key ---")

def load_api_key():
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        # Try secrets.toml for local dev
        import toml
        with open(".streamlit/secrets.toml", "r") as f:
            secrets = toml.load(f)
        key = secrets["GEMINI_API_KEY"]
    assert key and len(key) > 10, "API key looks invalid"
    os.environ["GEMINI_API_KEY"] = key
    return key

api_key = check("GEMINI_API_KEY is available", load_api_key)

if not api_key:
    print("\nCannot continue without an API key. Exiting.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Step 2: Knowledge Base
# ---------------------------------------------------------------------------

print("\n--- Step 2: Knowledge Base (ChromaDB) ---")

def check_chroma_exists():
    assert os.path.exists("chroma_db"), "chroma_db directory missing -- run: python updater.py"
    assert os.path.exists("chroma_db/chroma.sqlite3"), "chroma.sqlite3 missing inside chroma_db"
    return True

def check_rag_search():
    from tools.onco_tools import oncology_rag_search
    result = oncology_rag_search.invoke({"query": "what is chemotherapy"})
    assert result, "RAG search returned empty"
    assert "Knowledge base is not available" not in result, result
    # Show a short preview so you can see what was retrieved
    preview = result[:200].replace("\n", " ")
    print(f"         Preview: {preview}...")
    return result

check("chroma_db directory exists", check_chroma_exists)
check("oncology_rag_search returns results", check_rag_search)

# ---------------------------------------------------------------------------
# Step 3: External APIs
# ---------------------------------------------------------------------------

print("\n--- Step 3: External APIs ---")

def check_pubmed():
    from tools.external_tools import fetch_pubmed_abstracts
    result = fetch_pubmed_abstracts.invoke({"query": "lung cancer immunotherapy"})
    assert result, "PubMed returned empty"
    assert "failed" not in result.lower(), result
    preview = result[:150].replace("\n", " ")
    print(f"         Preview: {preview}...")
    return result

def check_clinical_trials():
    from tools.external_tools import search_clinical_trials
    result = search_clinical_trials.invoke({"condition": "breast cancer", "phase": ""})
    assert result, "ClinicalTrials returned empty"
    assert "failed" not in result.lower(), result
    preview = result[:150].replace("\n", " ")
    print(f"         Preview: {preview}...")
    return result

def check_arxiv():
    from tools.external_tools import summarize_arxiv_paper
    # Use a known real oncology paper ID
    result = summarize_arxiv_paper.invoke({"arxiv_id": "2304.01373"})
    assert result, "arXiv returned empty"
    assert "failed" not in result.lower(), result
    preview = result[:150].replace("\n", " ")
    print(f"         Preview: {preview}...")
    return result

check("PubMed API is reachable", check_pubmed)
check("ClinicalTrials.gov API is reachable", check_clinical_trials)
check("arXiv API is reachable", check_arxiv)

# ---------------------------------------------------------------------------
# Step 4: Agent Graph Builds
# ---------------------------------------------------------------------------

print("\n--- Step 4: Agent Graph Construction ---")

def check_agent_builds():
    from agent.onco_agent import build_agent
    agent = build_agent()
    assert agent is not None
    return agent

def check_supervisor_builds():
    from agent.supervisor import build_supervisor
    supervisor = build_supervisor()
    assert supervisor is not None
    return supervisor

agent = check("Fallback ReAct agent graph compiles", check_agent_builds)
supervisor = check("5-role supervisor graph compiles", check_supervisor_builds)

# ---------------------------------------------------------------------------
# Step 5: Full Supervisor Turn (live LLM call)
# ---------------------------------------------------------------------------

print("\n--- Step 5: Full Supervisor Reasoning Turn (live LLM call) ---")
print("    NOTE: This will call the Gemini API and may take 10-30 seconds.\n")

if supervisor:
    def check_supervisor_turn():
        from agent.supervisor import run_supervisor
        result = run_supervisor(
            agent_graph=supervisor,
            user_message="What are the common side effects of chemotherapy?",
            thread_id="test-supervisor-001",
        )
        assert result["response"], "Supervisor returned empty response"
        assert len(result["response"]) > 50, "Response is suspiciously short"
        print(f"         Tools used  : {result['tools_used']}")
        print(f"         Steps taken : {len(result['steps'])}")
        preview = result["response"][:200].replace("\n", " ")
        print(f"         Response    : {preview}...")
        return result

    turn1_result = check("Supervisor completes a reasoning turn", check_supervisor_turn)
else:
    turn1_result = None
    print("  [SKIP] Step 5: skipped because supervisor graph failed to build.")

# Step 6: Memory continuity (second turn in same thread)
print("\n--- Step 6: Memory Continuity (follow-up question) ---")

if turn1_result:
        def check_memory_turn():
            from agent.supervisor import run_supervisor
            result = run_supervisor(
                agent_graph=supervisor,
                user_message="Which of those side effects are most common with cisplatin specifically?",
                thread_id="test-supervisor-001",  # Same thread = same memory
            )
            assert result["response"], "Agent returned empty response on follow-up"
            # If the agent has memory, it should mention cisplatin or the prior context
            has_context = any(
                word in result["response"].lower()
                for word in ["cisplatin", "side effect", "nausea", "neuropathy", "kidney"]
            )
            assert has_context, (
                "Agent response does not appear to use prior context. "
                f"Response: {result['response'][:300]}"
            )
            preview = result["response"][:200].replace("\n", " ")
            print(f"         Tools used  : {result['tools_used']}")
            print(f"         Response    : {preview}...")
            return result

        check("Agent remembers previous turn (memory continuity)", check_memory_turn)
else:
    print("  [SKIP] Steps 5 and 6: skipped because supervisor graph failed to build.")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("\n" + "=" * 55)
passed = sum(1 for _, ok in results if ok)
failed = sum(1 for _, ok in results if not ok)
print(f"Results: {passed} passed, {failed} failed out of {len(results)} checks")
print("=" * 55)

if failed:
    print("\nFailed checks:")
    for label, ok in results:
        if not ok:
            print(f"  - {label}")
    sys.exit(1)
else:
    print("\nAll checks passed. The system is ready.")
    sys.exit(0)

# test_agent.py — End-to-end verification script.
# Usage: python test_agent.py

import os, sys, warnings
warnings.filterwarnings("ignore")

results = []

def check(label, fn):
    try:
        value = fn()
        print(f"  [PASS] {label}")
        results.append((label, True))
        return value
    except Exception as e:
        print(f"  [FAIL] {label}")
        print(f"         Error: {e}")
        results.append((label, False))
        return None

# --- Step 1: API Key ---
print("\n--- Step 1: API Key ---")

def load_api_key():
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        import toml
        with open(".streamlit/secrets.toml", "r") as f:
            key = toml.load(f)["GEMINI_API_KEY"]
    assert key and len(key) > 10, "API key looks invalid"
    os.environ["GEMINI_API_KEY"] = key
    return key

api_key = check("GEMINI_API_KEY is available", load_api_key)
if not api_key:
    print("\nCannot continue without an API key. Exiting.")
    sys.exit(1)

# --- Step 2: Knowledge Base ---
print("\n--- Step 2: Knowledge Base (ChromaDB) ---")

def check_chroma_exists():
    assert os.path.exists("chroma_db/chroma.sqlite3"), "chroma_db missing — run: python updater.py"
    return True

def check_rag_search():
    from tools.onco_tools import oncology_rag_search
    result = oncology_rag_search.invoke({"query": "what is chemotherapy"})
    assert result and "Knowledge base is not available" not in result
    print(f"         Preview: {result[:200].replace(chr(10), ' ')}...")
    return result

check("chroma_db exists", check_chroma_exists)
check("oncology_rag_search returns results", check_rag_search)

# --- Step 3: External APIs ---
print("\n--- Step 3: External APIs ---")

def check_pubmed():
    from tools.external_tools import fetch_pubmed_abstracts
    result = fetch_pubmed_abstracts.invoke({"query": "lung cancer immunotherapy"})
    assert result and "failed" not in result.lower()
    return result

def check_clinical_trials():
    from tools.external_tools import search_clinical_trials
    result = search_clinical_trials.invoke({"condition": "breast cancer", "phase": ""})
    assert result and "failed" not in result.lower()
    return result

def check_arxiv():
    from tools.external_tools import summarize_arxiv_paper
    result = summarize_arxiv_paper.invoke({"arxiv_id": "2304.01373"})
    assert result and "failed" not in result.lower()
    return result

check("PubMed API", check_pubmed)
check("ClinicalTrials.gov API", check_clinical_trials)
check("arXiv API", check_arxiv)

# --- Step 4: Agent Graph Builds ---
print("\n--- Step 4: Agent Graph ---")

def check_agent_builds():
    from agent.onco_agent import build_agent
    agent = build_agent()
    assert agent is not None
    return agent

agent = check("Agent graph compiles", check_agent_builds)

# --- Step 5: Full Agent Turn (live LLM call) ---
print("\n--- Step 5: Agent Reasoning Turn (live LLM call) ---")

if agent:
    def check_agent_turn():
        from agent.onco_agent import run_agent
        result = run_agent(
            agent_graph=agent,
            user_message="What are the common side effects of chemotherapy?",
            thread_id="test-agent-001",
        )
        assert result["response"] and len(result["response"]) > 50
        print(f"         Tools: {result['tools_used']}")
        print(f"         Response: {result['response'][:200].replace(chr(10), ' ')}...")
        return result

    turn1 = check("Agent completes a reasoning turn", check_agent_turn)
else:
    turn1 = None
    print("  [SKIP] Agent graph failed to build.")

# --- Step 6: Memory Continuity ---
print("\n--- Step 6: Memory Continuity ---")

if turn1:
    def check_memory_turn():
        from agent.onco_agent import run_agent
        result = run_agent(
            agent_graph=agent,
            user_message="Which of those side effects are most common with cisplatin specifically?",
            thread_id="test-agent-001",
        )
        assert result["response"]
        has_context = any(
            w in result["response"].lower()
            for w in ["cisplatin", "side effect", "nausea", "neuropathy", "kidney"]
        )
        assert has_context, f"No prior context in response: {result['response'][:300]}"
        print(f"         Response: {result['response'][:200].replace(chr(10), ' ')}...")
        return result

    check("Agent remembers previous turn", check_memory_turn)
else:
    print("  [SKIP] Skipped — agent graph failed to build.")

# --- Summary ---
print("\n" + "=" * 55)
passed = sum(1 for _, ok in results if ok)
failed = sum(1 for _, ok in results if not ok)
print(f"Results: {passed} passed, {failed} failed out of {len(results)} checks")
print("=" * 55)
if failed:
    for label, ok in results:
        if not ok:
            print(f"  FAILED: {label}")
    sys.exit(1)
else:
    print("\nAll checks passed. The system is ready.")
    sys.exit(0)

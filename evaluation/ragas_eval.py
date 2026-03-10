# evaluation/ragas_eval.py
#
# RAGAS-based evaluation for OncoBot responses.
#
# Measures two key RAG quality metrics:
#   - Faithfulness:      Is the answer grounded in the retrieved context?
#                        Score near 1.0 means the answer doesn't hallucinate.
#   - Answer Relevancy: Is the answer actually relevant to the question?
#                        Score near 1.0 means the response addresses the query.
#
# Both metrics use an LLM-as-judge approach (Gemini) and do NOT require
# pre-labeled ground truth, making them suitable for live evaluation.
#
# Usage from Streamlit sidebar:
#   from evaluation.ragas_eval import evaluate_last_response
#   scores = evaluate_last_response(question, answer)
#
# Usage as standalone script:
#   python -m evaluation.ragas_eval
#
# Requirements:
#   pip install ragas datasets
#   GEMINI_API_KEY must be set in the environment.

import os


def evaluate_last_response(question: str, answer: str) -> dict:
    """
    Evaluate a single Q&A pair against RAGAS faithfulness and answer relevancy.

    Retrieves relevant contexts from the local RAG system for the given question,
    then scores how faithfully the answer is grounded in those contexts and how
    relevant the answer is to the question.

    Args:
        question: The user's original query.
        answer:   The agent's final response to evaluate.

    Returns:
        Dict with float scores:
            {"faithfulness": float, "answer_relevancy": float}
        Or on failure:
            {"error": str}
    """
    try:
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import faithfulness, answer_relevancy
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
        from datasets import Dataset
    except ImportError as exc:
        return {"error": f"RAGAS dependencies not installed: {exc}. Run: pip install ragas datasets"}

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return {"error": "GEMINI_API_KEY not set in environment"}

    # -------------------------------------------------------------------------
    # Retrieve RAG contexts for this question.
    # We call oncology_rag_search directly so we get the same retrieval the
    # agent would have used, without needing to instrument the agent run.
    # -------------------------------------------------------------------------
    try:
        from tools.onco_tools import oncology_rag_search
        raw_result = oncology_rag_search.invoke({"query": question, "top_k": 5})
        # oncology_rag_search returns a formatted string; treat it as one context chunk.
        # If the tool returns a list-of-dicts in future, adjust here.
        contexts = [str(raw_result)] if raw_result else ["No RAG context available."]
    except Exception as exc:
        contexts = [f"RAG retrieval failed: {exc}"]

    # -------------------------------------------------------------------------
    # Build RAGAS dataset.
    # -------------------------------------------------------------------------
    dataset = Dataset.from_dict({
        "question": [question],
        "answer":   [answer],
        "contexts": [contexts],
    })

    # -------------------------------------------------------------------------
    # Configure Gemini as the RAGAS judge LLM and embedding model.
    # Using gemini-3.1-flash-lite-preview keeps costs minimal (500 RPD free).
    # -------------------------------------------------------------------------
    judge_llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(
            model="gemini-3.1-flash-lite-preview",
            google_api_key=api_key,
            temperature=0,
        )
    )
    judge_embeddings = LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key,
        )
    )

    # -------------------------------------------------------------------------
    # Run evaluation.
    # -------------------------------------------------------------------------
    try:
        result = ragas_evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy],
            llm=judge_llm,
            embeddings=judge_embeddings,
            raise_exceptions=False,  # Return NaN rather than crashing on edge cases
        )
        scores = result.to_pandas()
        return {
            "faithfulness":     round(float(scores["faithfulness"].iloc[0]), 3),
            "answer_relevancy": round(float(scores["answer_relevancy"].iloc[0]), 3),
        }
    except Exception as exc:
        return {"error": str(exc)[:200]}


# ---------------------------------------------------------------------------
# Standalone script entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Quick smoke-test with a sample oncology Q&A pair.
    SAMPLE_QUESTION = "What are the common side effects of pembrolizumab?"
    SAMPLE_ANSWER = (
        "Pembrolizumab (Keytruda) is an anti-PD-1 checkpoint inhibitor. "
        "Common immune-related adverse events include fatigue, rash, diarrhea, "
        "and immune-related pneumonitis. Patients should consult their oncologist "
        "for monitoring and management. (Source: oncology_rag_search)"
    )

    print("Running RAGAS evaluation on sample Q&A pair...")
    print(f"  Question: {SAMPLE_QUESTION}")
    print(f"  Answer:   {SAMPLE_ANSWER[:80]}...")
    print()

    scores = evaluate_last_response(SAMPLE_QUESTION, SAMPLE_ANSWER)
    if "error" in scores:
        print(f"  ERROR: {scores['error']}")
    else:
        for metric, score in scores.items():
            bar = "#" * int(score * 20)
            print(f"  {metric:<22} {score:.3f}  [{bar:<20}]")

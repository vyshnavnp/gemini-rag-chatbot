# evaluation/ragas_eval.py
#
# LLM-as-judge evaluation for OncoBot responses.
#
# Measures two key RAG quality metrics:
#   - Faithfulness:      Is the answer grounded in the retrieved context?
#                        Score near 1.0 means the answer doesn't hallucinate.
#   - Answer Relevancy:  Is the answer actually relevant to the question?
#                        Score near 1.0 means the response addresses the query.
#
# Uses Gemini directly as the judge LLM (no external ragas package needed).
#
# Usage from Streamlit sidebar:
#   from evaluation.ragas_eval import evaluate_last_response
#   scores = evaluate_last_response(question, answer)
#
# Usage as standalone script:
#   python -m evaluation.ragas_eval
#
# Requirements:
#   GEMINI_API_KEY must be set in the environment.

import os
import json
import re


def _call_gemini(prompt: str, api_key: str) -> str:
    """Call Gemini with a plain text prompt and return the response text."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite-preview",
        google_api_key=api_key,
        temperature=0,
    )
    resp = llm.invoke(prompt)
    content = resp.content
    if isinstance(content, list):
        return " ".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content) if content else ""


def evaluate_last_response(question: str, answer: str) -> dict:
    """
    Evaluate a single Q&A pair for faithfulness and answer relevancy.

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
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return {"error": "GEMINI_API_KEY not set in environment"}

    # -------------------------------------------------------------------------
    # Retrieve RAG contexts for this question.
    # -------------------------------------------------------------------------
    try:
        from tools.onco_tools import oncology_rag_search
        raw_result = oncology_rag_search.invoke({"query": question})
        contexts = str(raw_result) if raw_result else "No RAG context available."
    except Exception as exc:
        contexts = f"RAG retrieval failed: {exc}"

    # -------------------------------------------------------------------------
    # Faithfulness: Is the answer grounded in the retrieved context?
    # -------------------------------------------------------------------------
    faithfulness_prompt = f"""You are an evaluation judge. Score how faithfully the Answer is grounded in the Context.

A score of 1.0 means every claim in the answer is supported by the context.
A score of 0.0 means the answer is entirely unsupported or contradicts the context.

Context:
{contexts[:3000]}

Question: {question}

Answer: {answer[:2000]}

Respond with ONLY a JSON object: {{"score": <float between 0.0 and 1.0>, "reason": "<one sentence>"}}"""

    # -------------------------------------------------------------------------
    # Answer Relevancy: Does the answer address the question?
    # -------------------------------------------------------------------------
    relevancy_prompt = f"""You are an evaluation judge. Score how relevant the Answer is to the Question.

A score of 1.0 means the answer fully and directly addresses the question.
A score of 0.0 means the answer is completely off-topic or irrelevant.

Question: {question}

Answer: {answer[:2000]}

Respond with ONLY a JSON object: {{"score": <float between 0.0 and 1.0>, "reason": "<one sentence>"}}"""

    scores = {}
    for metric, prompt in [("faithfulness", faithfulness_prompt), ("answer_relevancy", relevancy_prompt)]:
        try:
            raw = _call_gemini(prompt, api_key)
            # Try JSON parse first
            m = re.search(r'\{[^}]+\}', raw)
            if m:
                data = json.loads(m.group())
                score = float(data.get("score", 0))
                scores[metric] = round(min(max(score, 0.0), 1.0), 3)
            else:
                scores[metric] = 0.0
        except Exception as exc:
            return {"error": f"{metric} evaluation failed: {str(exc)[:150]}"}

    return scores


# ---------------------------------------------------------------------------
# Standalone script entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    SAMPLE_QUESTION = "What are the common side effects of pembrolizumab?"
    SAMPLE_ANSWER = (
        "Pembrolizumab (Keytruda) is an anti-PD-1 checkpoint inhibitor. "
        "Common immune-related adverse events include fatigue, rash, diarrhea, "
        "and immune-related pneumonitis. Patients should consult their oncologist "
        "for monitoring and management. (Source: oncology_rag_search)"
    )

    print("Running evaluation on sample Q&A pair...")
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

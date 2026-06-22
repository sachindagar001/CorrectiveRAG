"""
Evaluator — LLM-as-judge that grades the relevance of retrieved documents.

This is the heart of CRAG. For each retrieved chunk, the LLM outputs a relevance
score in [0, 1] plus a short reasoning. The aggregator decides:
  - "relevant"    -> use retrieved docs for generation
  - "irrelevant"  -> trigger web search fallback
  - "ambiguous"   -> combine both (docs + web)
"""
from typing import List, Dict, Any
from src.crag.llm import llm_invoke_json


EVALUATOR_SYSTEM = """You are a strict retrieval relevance evaluator.
Given a user query and a retrieved document, judge whether the document actually
helps answer the query. Output a relevance score in [0.0, 1.0]:
  - 1.0 = perfectly on-topic and directly answers the query
  - 0.7 = mostly relevant, covers part of the answer
  - 0.4 = tangentially related (same topic, different focus)
  - 0.0 = completely irrelevant

Respond ONLY with valid JSON:
{"score": <float>, "reasoning": "<one short sentence>"}
"""

AGGREGATOR_SYSTEM = """You are an aggregation judge.
Given a list of document relevance scores and a user query, decide:
  - "relevant"   if at least one doc has score >= 0.6
  - "irrelevant" if all docs have score < 0.3
  - "ambiguous"  otherwise (mixed signals)

Respond ONLY with valid JSON:
{"decision": "<relevant|irrelevant|ambiguous>", "overall_score": <float>, "reasoning": "<one sentence>"}
"""


def evaluate_single_doc(query: str, doc_text: str) -> Dict[str, Any]:
    """Score a single document's relevance to the query."""
    # Truncate long docs to keep prompt short
    doc_text = doc_text[:1500]
    prompt = (
        f"User Query: {query}\n\n"
        f"Retrieved Document:\n{doc_text}\n\n"
        f"Judge relevance. Output JSON only."
    )
    result = llm_invoke_json(prompt, system=EVALUATOR_SYSTEM)
    # Sanitize
    score = float(result.get("score", 0.0))
    score = max(0.0, min(1.0, score))
    return {
        "score": score,
        "reasoning": result.get("reasoning", ""),
    }


def evaluate_retrieved_docs(
    query: str,
    docs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Evaluate all retrieved docs and return aggregated decision.

    Returns:
        {
            "per_doc_scores": [float, ...],
            "overall_relevance": float,
            "decision": "relevant" | "irrelevant" | "ambiguous",
            "reasoning": str,
        }
    """
    if not docs:
        return {
            "per_doc_scores": [],
            "overall_relevance": 0.0,
            "decision": "irrelevant",
            "reasoning": "No documents were retrieved.",
        }

    per_doc_scores: List[float] = []
    per_doc_reasoning: List[str] = []

    for doc in docs:
        result = evaluate_single_doc(query, doc.get("text", ""))
        per_doc_scores.append(result["score"])
        per_doc_reasoning.append(result["reasoning"])

    overall = sum(per_doc_scores) / len(per_doc_scores) if per_doc_scores else 0.0

    # Build the aggregator prompt
    doc_summaries = "\n".join(
        f"  Doc {i+1}: score={s:.2f} | {r}"
        for i, (s, r) in enumerate(zip(per_doc_scores, per_doc_reasoning))
    )
    agg_prompt = (
        f"User Query: {query}\n\n"
        f"Retrieved docs:\n{doc_summaries}\n\n"
        f"Aggregate these and decide. Output JSON only."
    )
    agg_result = llm_invoke_json(agg_prompt, system=AGGREGATOR_SYSTEM)

    return {
        "per_doc_scores": per_doc_scores,
        "overall_relevance": float(agg_result.get("overall_score", overall)),
        "decision": agg_result.get("decision", "ambiguous"),
        "reasoning": agg_result.get("reasoning", ""),
    }

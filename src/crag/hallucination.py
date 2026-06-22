"""
Hallucination Checker — LLM-as-judge that scores how well the generated
answer is grounded in the provided source documents.

Score in [0, 1]:
  - 1.0 = every claim in the answer is directly supported by the sources
  - 0.5 = some claims are supported, others are not
  - 0.0 = answer is largely fabricated / not supported by sources

If the score is below threshold, the answer is flagged as hallucinated.
"""
from typing import List, Dict, Any
from src.crag.llm import llm_invoke_json


HALLUCINATION_SYSTEM = """You are a strict factual-grounding evaluator.
Given a generated answer and a set of source documents, judge whether EVERY
factual claim in the answer is directly supported by the sources.

Output JSON only:
{
  "score": <float in [0,1]>,
  "reasoning": "<one short sentence>",
  "unsupported_claims": ["<claim 1>", "<claim 2>", ...]
}

Scoring rubric:
  - 1.0 = every claim is supported
  - 0.7 = most claims supported, minor unsupported details
  - 0.4 = several unsupported claims
  - 0.0 = answer is largely fabricated
"""

HALLUCINATION_THRESHOLD = 0.5


def check_hallucination(
    answer: str,
    sources: List[Dict[str, Any]],
    threshold: float = HALLUCINATION_THRESHOLD,
) -> Dict[str, Any]:
    """Score the answer's grounding in the provided sources.

    Returns:
        {
            "score": float,
            "reasoning": str,
            "unsupported_claims": List[str],
            "is_hallucinated": bool,
        }
    """
    if not sources:
        # No sources = pure generation = treat as fully hallucinated
        return {
            "score": 0.0,
            "reasoning": "No source documents provided — answer is ungrounded.",
            "unsupported_claims": [],
            "is_hallucinated": True,
        }

    sources_text = "\n\n".join(
        f"[Source {i+1}] ({s.get('source', 'unknown')}): {s.get('text', '')[:1000]}"
        for i, s in enumerate(sources)
    )

    prompt = (
        f"Generated Answer:\n{answer}\n\n"
        f"Source Documents:\n{sources_text}\n\n"
        f"Judge factual grounding. Output JSON only."
    )

    result = llm_invoke_json(prompt, system=HALLUCINATION_SYSTEM)
    score = float(result.get("score", 0.0))
    score = max(0.0, min(1.0, score))

    return {
        "score": score,
        "reasoning": result.get("reasoning", ""),
        "unsupported_claims": result.get("unsupported_claims", []),
        "is_hallucinated": score < threshold,
    }

"""
Query Rewriter — Implements two advanced retrieval techniques:

1. Multi-Query Expansion: ask the LLM to reformulate the user query into
   3 different phrasings, then retrieve for all of them and merge results.
   This increases recall by surfacing docs that match different phrasings.

2. HyDE (Hypothetical Document Embeddings): ask the LLM to draft a hypothetical
   answer to the query, then embed THAT (instead of the query) for retrieval.
   The intuition: a fake-but-plausible answer is closer in embedding space to
   real answers than the short query is.
"""
from typing import List, Optional
from src.crag.llm import llm_invoke


MULTI_QUERY_SYSTEM = """You are a search query optimizer.
Given a user question, generate 3 alternative phrasings that preserve the
intent but use different vocabulary. Each variant must be a complete question.

Output ONE question per line. No numbering, no bullets, no preamble.
Max 3 lines.
"""

HYDE_SYSTEM = """You are a hypothetical document generator.
Given a user query, write a short (3-5 sentence) plausible answer that would
satisfy the query. Do NOT add disclaimers like "I'm not sure" — write as if
you were a confident expert answering. This document will be used for embedding
retrieval, not shown to the user.
"""


def rewrite_query_multi(query: str, n_variants: int = 3) -> List[str]:
    """Generate `n_variants` alternative phrasings of the query."""
    prompt = (
        f"User question: {query}\n\n"
        f"Generate {n_variants} alternative phrasings."
    )
    raw = llm_invoke(prompt, system=MULTI_QUERY_SYSTEM)
    variants = [
        line.strip().lstrip("0123456789.-) ").strip()
        for line in raw.strip().split("\n")
        if line.strip()
    ]
    # Deduplicate while preserving order
    seen = set()
    unique: List[str] = []
    for v in variants:
        v_lower = v.lower()
        if v_lower not in seen and v_lower != query.lower():
            seen.add(v_lower)
            unique.append(v)
    return unique[:n_variants]


def generate_hyde_document(query: str) -> str:
    """Generate a hypothetical answer to use as a retrieval embedding."""
    prompt = f"User query: {query}\n\nWrite a 3-5 sentence hypothetical answer."
    return llm_invoke(prompt, system=HYDE_SYSTEM).strip()


def rewrite_query_full(
    query: str,
    use_multi_query: bool = True,
    use_hyde: bool = True,
) -> dict:
    """Run both query rewriting techniques.

    Returns:
        {
            "rewritten_queries": List[str],  # multi-query variants (excluding original)
            "hyde_document": Optional[str],  # hypothetical answer or None
            "all_search_queries": List[str], # everything to embed for retrieval
        }
    """
    rewrites: List[str] = []
    hyde_doc: Optional[str] = None

    if use_multi_query:
        try:
            rewrites = rewrite_query_multi(query, n_variants=3)
        except Exception:
            rewrites = []

    if use_hyde:
        try:
            hyde_doc = generate_hyde_document(query)
        except Exception:
            hyde_doc = None

    # All queries to use for multi-query retrieval
    all_queries = [query] + rewrites
    if hyde_doc:
        all_queries.append(hyde_doc)

    return {
        "rewritten_queries": rewrites,
        "hyde_document": hyde_doc,
        "all_search_queries": all_queries,
    }

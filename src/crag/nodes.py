"""
CRAG Nodes — The individual functions that make up the LangGraph workflow.

Each node takes the CRAGState, does one job, and returns a partial state dict
to be merged back in. The graph orchestrates the flow.

Workflow:
    START
      ↓
    query_rewrite  (multi-query + HyDE)
      ↓
    retrieve       (FAISS top-K)
      ↓
    evaluate       (LLM grades each doc)
      ↓
    <route> ─────────────────────────────────┐
      ↓                ↓                     ↓
   relevant       ambiguous              irrelevant
      ↓                ↓                     ↓
   refine        refine + web_search       web_search
      ↓                ↓                     ↓
      └────────────────┴─────────────────────┘
                       ↓
                   generate
                       ↓
                  hallucination_check
                       ↓
                      END
"""
import time
from typing import Dict, Any, List
from src.crag.state import CRAGState, RetrievedDoc
from src.crag import embeddings as emb
from src.crag.query_rewriter import rewrite_query_full
from src.crag.evaluator import evaluate_retrieved_docs
from src.crag.web_search import web_search
from src.crag.hallucination import check_hallucination
from src.crag.llm import llm_invoke


# --------------------------------------------------------------------- helpers
def _now() -> str:
    from datetime import datetime
    return datetime.utcnow().isoformat()


def _track(state: CRAGState, node_name: str, start: float) -> None:
    """Record this node's execution in the trace and latency map."""
    elapsed_ms = (time.perf_counter() - start) * 1000
    state.setdefault("node_trace", []).append(node_name)
    state.setdefault("latency_ms", {})[node_name] = round(elapsed_ms, 2)


# =================================================================== NODE 1
def node_query_rewrite(state: CRAGState) -> Dict[str, Any]:
    """Rewrite the query using multi-query expansion + HyDE."""
    start = time.perf_counter()
    query = state["original_query"]
    use_rw = state.get("use_query_rewriting", True)

    if not use_rw:
        _track(state, "query_rewrite", start)
        return {
            "rewritten_queries": [],
            "hyde_document": None,
        }

    try:
        result = rewrite_query_full(query, use_multi_query=True, use_hyde=True)
    except Exception as e:
        print(f"[query_rewrite] Error: {e}")
        result = {"rewritten_queries": [], "hyde_document": None, "all_search_queries": [query]}

    _track(state, "query_rewrite", start)
    return {
        "rewritten_queries": result["rewritten_queries"],
        "hyde_document": result["hyde_document"],
    }


# =================================================================== NODE 2
def node_retrieve(state: CRAGState) -> Dict[str, Any]:
    """Retrieve documents from FAISS. Uses multi-query if available."""
    start = time.perf_counter()
    # Lazy import to avoid circular dependency
    from src.crag.retriever import FAISSRetriever

    retriever = FAISSRetriever()
    if not retriever.is_ready():
        loaded = retriever.load()
        if not loaded:
            _track(state, "retrieve", start)
            return {
                "retrieved_docs": [],
                "retrieval_method": "none",
                "error": "FAISS index not found. Run scripts/build_index.py first.",
            }

    queries = state.get("rewritten_queries", [])
    hyde = state.get("hyde_document")

    # If we have multi-query or HyDE, use multi-search
    all_queries = [state["original_query"]] + queries
    if hyde:
        all_queries.append(hyde)

    if len(all_queries) > 1:
        docs = retriever.search_multi(all_queries, top_k=5)
        method = "multi_query+hyde" if hyde else "multi_query"
    else:
        docs = retriever.search(state["original_query"], top_k=5)
        method = "faiss"

    _track(state, "retrieve", start)
    return {
        "retrieved_docs": docs,
        "retrieval_method": method,
    }


# =================================================================== NODE 3
def node_evaluate(state: CRAGState) -> Dict[str, Any]:
    """Evaluate retrieved docs with the LLM-as-judge."""
    start = time.perf_counter()
    query = state["original_query"]
    docs = state.get("retrieved_docs", [])

    if not docs:
        _track(state, "evaluate", start)
        return {
            "relevance_scores": [],
            "overall_relevance": 0.0,
            "relevance_decision": "irrelevant",
            "evaluator_reasoning": "No docs retrieved.",
        }

    try:
        result = evaluate_retrieved_docs(query, docs)
    except Exception as e:
        print(f"[evaluate] Error: {e}")
        result = {
            "per_doc_scores": [0.5] * len(docs),
            "overall_relevance": 0.5,
            "decision": "ambiguous",
            "reasoning": f"Evaluator error: {e}",
        }

    # Attach relevance scores back to the docs
    for doc, score in zip(docs, result["per_doc_scores"]):
        doc["relevance_score"] = float(score)

    _track(state, "evaluate", start)
    return {
        "retrieved_docs": docs,
        "relevance_scores": result["per_doc_scores"],
        "overall_relevance": result["overall_relevance"],
        "relevance_decision": result["decision"],
        "evaluator_reasoning": result["reasoning"],
    }


# =================================================================== NODE 4 (router)
def node_route(state: CRAGState) -> str:
    """Conditional router — decides the next node based on the evaluator decision.

    Returns one of: 'relevant', 'ambiguous', 'irrelevant'
    """
    decision = state.get("relevance_decision", "ambiguous")
    if decision == "relevant":
        return "relevant"
    elif decision == "irrelevant":
        return "irrelevant"
    else:
        return "ambiguous"


# =================================================================== NODE 5a
def node_web_search(state: CRAGState) -> Dict[str, Any]:
    """Fallback: search DuckDuckGo when local docs are insufficient."""
    start = time.perf_counter()
    query = state["original_query"]

    # Use a refined search query: prefer the first rewritten query if available
    rewrites = state.get("rewritten_queries", [])
    search_query = rewrites[0] if rewrites else query

    try:
        results = web_search(search_query, max_results=5)
    except Exception as e:
        print(f"[web_search] Error: {e}")
        results = []

    _track(state, "web_search", start)
    return {
        "web_search_used": True,
        "web_search_results": results,
        "web_search_query": search_query,
    }


# =================================================================== NODE 5b
def node_knowledge_refinement(state: CRAGState) -> Dict[str, Any]:
    """Strip noise from retrieved docs to produce a clean 'refined_knowledge' string.

    Simple version: just concatenate top docs up to a token budget.
    """
    start = time.perf_counter()
    docs = state.get("retrieved_docs", [])
    web_results = state.get("web_search_results", [])

    # Keep only docs with relevance >= 0.3
    relevant_docs = [d for d in docs if d.get("relevance_score", 0.0) >= 0.3]
    # If nothing passed the threshold, keep top 3 by score
    if not relevant_docs and docs:
        relevant_docs = sorted(docs, key=lambda d: d.get("relevance_score", 0.0), reverse=True)[:3]

    all_sources = relevant_docs + web_results

    # Token budget ~ 4000 chars
    BUDGET = 4000
    refined_parts: List[str] = []
    used = 0
    for s in all_sources:
        text = s.get("text", "")
        if used + len(text) > BUDGET:
            text = text[: BUDGET - used]
        refined_parts.append(f"[{s.get('source', 'unknown')}] {text}")
        used += len(text)
        if used >= BUDGET:
            break

    refined = "\n\n".join(refined_parts)
    _track(state, "knowledge_refinement", start)
    return {"refined_knowledge": refined}


# =================================================================== NODE 6
def node_generate(state: CRAGState) -> Dict[str, Any]:
    """Generate the final answer using the refined knowledge as context."""
    start = time.perf_counter()
    query = state["original_query"]
    context = state.get("refined_knowledge", "")

    system = """You are a precise, factual AI assistant. Answer the user's question
using ONLY the provided context. If the context does not contain the answer,
say "I don't have enough information to answer this." Do not make up facts.

Cite sources in your answer using [source] tags where applicable."""

    prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer (cite sources like [arxiv:xxxx] or [web:duckduckgo]):"
    )

    try:
        answer = llm_invoke(prompt, system=system)
    except Exception as e:
        answer = f"Error generating answer: {e}"

    # Build citations list from the sources we actually used
    docs = state.get("retrieved_docs", [])
    web_results = state.get("web_search_results", [])
    citations = []
    for d in docs:
        if d.get("relevance_score", 0.0) >= 0.3:
            citations.append({
                "source": d.get("source", "unknown"),
                "title": d.get("title", ""),
                "score": d.get("score", 0.0),
                "relevance_score": d.get("relevance_score", 0.0),
                "snippet": d.get("text", "")[:200],
            })
    for w in web_results:
        citations.append({
            "source": "web:duckduckgo",
            "title": w.get("title", ""),
            "url": w.get("url", ""),
            "score": w.get("score", 0.0),
            "snippet": w.get("text", "")[:200],
        })

    _track(state, "generate", start)
    return {
        "final_answer": answer,
        "citations": citations,
    }


# =================================================================== NODE 7
def node_hallucination_check(state: CRAGState) -> Dict[str, Any]:
    """Check whether the final answer is grounded in the sources."""
    start = time.perf_counter()
    answer = state.get("final_answer", "")
    docs = state.get("retrieved_docs", [])
    web_results = state.get("web_search_results", [])

    sources = []
    for d in docs:
        if d.get("relevance_score", 0.0) >= 0.3:
            sources.append(d)
    sources.extend(web_results)

    try:
        result = check_hallucination(answer, sources)
    except Exception as e:
        print(f"[hallucination_check] Error: {e}")
        result = {
            "score": 0.5,
            "reasoning": f"Check failed: {e}",
            "unsupported_claims": [],
            "is_hallucinated": False,
        }

    _track(state, "hallucination_check", start)
    return {
        "hallucination_score": result["score"],
        "hallucination_reasoning": result["reasoning"],
        "is_hallucinated": result["is_hallucinated"],
    }


# =================================================================== NODE 8 (baseline)
def node_naive_rag(state: CRAGState) -> Dict[str, Any]:
    """Naive RAG baseline: retrieve top-1 doc, generate directly.
    Used for the eval dashboard to show CRAG > naive RAG."""
    start = time.perf_counter()
    from src.crag.retriever import FAISSRetriever

    retriever = FAISSRetriever()
    if not retriever.is_ready():
        retriever.load()

    query = state["original_query"]
    docs = retriever.search(query, top_k=3)
    context = "\n\n".join(d.get("text", "")[:800] for d in docs)

    prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\nAnswer:"
    )
    try:
        answer = llm_invoke(prompt)
    except Exception as e:
        answer = f"Error: {e}"

    _track(state, "naive_rag", start)
    return {"naive_rag_answer": answer}

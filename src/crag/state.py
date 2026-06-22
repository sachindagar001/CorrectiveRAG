"""
CRAG State — TypedDict defining the state that flows through the LangGraph workflow.

Each node reads/writes fields of this state. The graph orchestrates the flow.
"""
from typing import TypedDict, List, Optional, Dict, Any
from datetime import datetime


class RetrievedDoc(TypedDict):
    """A single retrieved document chunk."""
    text: str
    source: str              # e.g., "arxiv:2401.15884" or "web:duckduckgo"
    title: Optional[str]
    score: float             # similarity score from retriever
    relevance_score: float   # LLM-graded relevance (0-1), filled by evaluator


class CRAGState(TypedDict):
    """Full state of the CRAG agent across the workflow."""
    # --- Input ---
    original_query: str
    conversation_history: List[Dict[str, str]]

    # --- Query Rewriting (multi-query + HyDE) ---
    rewritten_queries: List[str]      # expanded queries
    hyde_document: Optional[str]      # hypothetical answer for HyDE
    use_query_rewriting: bool

    # --- Retrieval ---
    retrieved_docs: List[RetrievedDoc]
    retrieval_method: str             # "faiss", "hybrid", etc.

    # --- Evaluation ---
    relevance_scores: List[float]     # per-doc relevance
    overall_relevance: float          # aggregated score
    relevance_decision: str           # "relevant" | "irrelevant" | "ambiguous"
    evaluator_reasoning: str

    # --- Knowledge Refinement ---
    refined_knowledge: str            # stripped-down relevant content

    # --- Web Search Fallback ---
    web_search_used: bool
    web_search_results: List[RetrievedDoc]
    web_search_query: str

    # --- Generation ---
    final_answer: str
    citations: List[Dict[str, Any]]

    # --- Hallucination Check ---
    hallucination_score: float        # 0-1 grounding score
    hallucination_reasoning: str
    is_hallucinated: bool

    # --- Naive RAG baseline (for comparison dashboard) ---
    naive_rag_answer: str

    # --- Observability ---
    node_trace: List[str]             # ordered list of executed node names
    latency_ms: Dict[str, float]      # per-node latency
    total_latency_ms: float
    timestamp: str
    error: Optional[str]

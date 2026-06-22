"""
LangGraph workflow definition for the CRAG agent.

Builds a conditional-edge graph:

    START → query_rewrite → retrieve → evaluate
                                            │
                                            ▼
                                       [router]
                                  /       |        \
                            relevant  ambiguous  irrelevant
                                │        │           │
                                ▼        ▼           ▼
                          refine_kn  refine_kn   web_search
                                │        │           │
                                │        ▼           │
                                │   web_search      │
                                │        │           │
                                └────────┴───────────┘
                                         │
                                         ▼
                                     generate
                                         │
                                         ▼
                                hallucination_check
                                         │
                                         ▼
                                        END
"""
import time
from typing import Optional, Dict, Any
from langgraph.graph import StateGraph, END

from src.crag.state import CRAGState
from src.crag.nodes import (
    node_query_rewrite,
    node_retrieve,
    node_evaluate,
    node_route,
    node_web_search,
    node_knowledge_refinement,
    node_generate,
    node_hallucination_check,
    node_naive_rag,
)


def build_crag_graph(use_naive_baseline: bool = True):
    """Build and compile the CRAG LangGraph workflow.

    Args:
        use_naive_baseline: If True, also run a naive RAG baseline in parallel
            for the eval dashboard. Adds ~1 LLM call latency.

    Returns:
        A compiled LangGraph runnable.
    """
    g = StateGraph(CRAGState)

    # ----- add nodes -----
    g.add_node("query_rewrite", node_query_rewrite)
    g.add_node("retrieve", node_retrieve)
    g.add_node("evaluate", node_evaluate)
    g.add_node("knowledge_refinement", node_knowledge_refinement)
    g.add_node("web_search", node_web_search)
    g.add_node("generate", node_generate)
    g.add_node("hallucination_check", node_hallucination_check)
    if use_naive_baseline:
        g.add_node("naive_rag", node_naive_rag)

    # ----- entry point -----
    g.set_entry_point("query_rewrite")

    # ----- linear edges -----
    g.add_edge("query_rewrite", "retrieve")
    g.add_edge("retrieve", "evaluate")

    # ----- conditional routing after evaluate -----
    g.add_conditional_edges(
        "evaluate",
        node_route,
        {
            "relevant": "knowledge_refinement",
            "ambiguous": "web_search",
            "irrelevant": "web_search",
        },
    )

    # ----- after web_search, go to knowledge_refinement (which merges docs + web) -----
    g.add_edge("web_search", "knowledge_refinement")

    # ----- generate, then hallucination check -----
    g.add_edge("knowledge_refinement", "generate")
    g.add_edge("generate", "hallucination_check")

    # ----- after hallucination check, END (or run naive baseline) -----
    if use_naive_baseline:
        g.add_edge("hallucination_check", "naive_rag")
        g.add_edge("naive_rag", END)
    else:
        g.add_edge("hallucination_check", END)

    return g.compile()


class CRAGGraph:
    """Convenience wrapper around the compiled LangGraph runnable.

    Usage:
        graph = CRAGGraph()
        result = graph.run("What is retrieval-augmented generation?")
    """

    def __init__(self, use_naive_baseline: bool = True):
        self.compiled = build_crag_graph(use_naive_baseline=use_naive_baseline)
        self.use_naive_baseline = use_naive_baseline

    def run(
        self,
        query: str,
        use_query_rewriting: bool = True,
        conversation_history: Optional[list] = None,
    ) -> CRAGState:
        """Run the full CRAG pipeline on a user query."""
        start_time = time.perf_counter()

        initial_state: CRAGState = {
            "original_query": query,
            "conversation_history": conversation_history or [],
            "rewritten_queries": [],
            "hyde_document": None,
            "use_query_rewriting": use_query_rewriting,
            "retrieved_docs": [],
            "retrieval_method": "",
            "relevance_scores": [],
            "overall_relevance": 0.0,
            "relevance_decision": "",
            "evaluator_reasoning": "",
            "refined_knowledge": "",
            "web_search_used": False,
            "web_search_results": [],
            "web_search_query": "",
            "final_answer": "",
            "citations": [],
            "hallucination_score": 0.0,
            "hallucination_reasoning": "",
            "is_hallucinated": False,
            "naive_rag_answer": "",
            "node_trace": [],
            "latency_ms": {},
            "total_latency_ms": 0.0,
            "timestamp": _now(),
            "error": None,
        }

        try:
            final_state = self.compiled.invoke(initial_state)
            final_state["total_latency_ms"] = round(
                (time.perf_counter() - start_time) * 1000, 2
            )
            return final_state
        except Exception as e:
            initial_state["error"] = str(e)
            initial_state["total_latency_ms"] = round(
                (time.perf_counter() - start_time) * 1000, 2
            )
            return initial_state

    def get_mermaid(self) -> str:
        """Return the workflow as a Mermaid diagram string (for UI rendering)."""
        # LangGraph provides .get_graph().draw_mermaid() — but we override with
        # a custom-styled version for the Streamlit UI.
        return _MERMAID_DIAGRAM

    def get_nodes(self) -> list:
        """Return ordered list of node names for the trace UI."""
        return [
            "query_rewrite",
            "retrieve",
            "evaluate",
            "knowledge_refinement",
            "web_search",
            "generate",
            "hallucination_check",
            "naive_rag",
        ]


def _now() -> str:
    from datetime import datetime
    return datetime.utcnow().isoformat()


# Custom mermaid diagram (used by the Streamlit UI)
_MERMAID_DIAGRAM = """flowchart TD
    START([User Query]) --> QR[Query Rewriter<br/>Multi-Query + HyDE]
    QR --> RET[FAISS Retriever<br/>BGE Embeddings]
    RET --> EVAL[Relevance Evaluator<br/>LLM-as-Judge]

    EVAL --> DEC{Relevant?}
    DEC -->|Relevant| REF[Knowledge Refinement]
    DEC -->|Ambiguous| WS[Web Search<br/>DuckDuckGo]
    DEC -->|Irrelevant| WS

    WS --> REF
    REF --> GEN[Answer Generator<br/>LLM via DeepSeek]
    GEN --> HALL[Hallucination Check<br/>LLM-as-Judge]
    HALL --> END([Final Answer + Citations])

    style START fill:#3b82f6,color:#fff,stroke:#1e40af
    style END fill:#10b981,color:#fff,stroke:#047857
    style EVAL fill:#f59e0b,color:#fff,stroke:#b45309
    style WS fill:#ef4444,color:#fff,stroke:#b91c1c
    style HALL fill:#8b5cf6,color:#fff,stroke:#6d28d9
    style DEC fill:#fbbf24,color:#000,stroke:#b45309
"""

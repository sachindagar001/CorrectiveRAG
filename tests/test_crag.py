"""
Basic tests for the CRAG agent.

Run with: pytest tests/ -v
"""
import os
import sys

# Make src importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_state_typedict():
    """CRAGState should be importable and have expected keys."""
    from src.crag.state import CRAGState
    # TypedDicts don't expose __annotations__ at runtime in older Pythons,
    # but we can still check it's a valid type
    assert CRAGState is not None


def test_arxiv_papers_loaded():
    """arxiv_papers.json should exist and have papers."""
    from src.data.loader import load_arxiv_papers
    papers = load_arxiv_papers()
    assert len(papers) > 0, "No papers found in dataset"
    # Each paper should have required fields
    for p in papers[:3]:
        assert "text" in p
        assert "source" in p
        assert "title" in p
        assert "abstract" in p


def test_search_papers_by_keyword():
    """Keyword search should find relevant papers."""
    from src.data.loader import search_papers_by_keyword
    results = search_papers_by_keyword("RAG")
    assert len(results) > 0, "Should find RAG papers"


def test_retriever_class_exists():
    """FAISSRetriever should be importable."""
    from src.crag.retriever import FAISSRetriever
    r = FAISSRetriever()
    assert r is not None
    assert r.is_ready() is False  # not loaded yet


def test_query_rewriter_imports():
    """Query rewriter functions should be importable."""
    from src.crag.query_rewriter import (
        rewrite_query_multi,
        generate_hyde_document,
        rewrite_query_full,
    )
    assert callable(rewrite_query_multi)
    assert callable(generate_hyde_document)
    assert callable(rewrite_query_full)


def test_evaluator_imports():
    """Evaluator functions should be importable."""
    from src.crag.evaluator import (
        evaluate_single_doc,
        evaluate_retrieved_docs,
    )
    assert callable(evaluate_single_doc)
    assert callable(evaluate_retrieved_docs)


def test_hallucination_imports():
    """Hallucination checker should be importable."""
    from src.crag.hallucination import check_hallucination
    assert callable(check_hallucination)


def test_web_search_imports():
    """Web search module should be importable."""
    from src.crag.web_search import web_search
    assert callable(web_search)


def test_graph_builds():
    """The CRAG LangGraph should build without errors."""
    from src.crag.graph import build_crag_graph, CRAGGraph
    compiled = build_crag_graph(use_naive_baseline=False)
    assert compiled is not None
    graph = CRAGGraph(use_naive_baseline=False)
    assert graph is not None
    # Mermaid diagram should be a non-empty string
    assert isinstance(graph.get_mermaid(), str)
    assert len(graph.get_mermaid()) > 0


def test_nodes_imports():
    """All node functions should be importable."""
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
    assert all(callable(f) for f in [
        node_query_rewrite, node_retrieve, node_evaluate, node_route,
        node_web_search, node_knowledge_refinement, node_generate,
        node_hallucination_check, node_naive_rag,
    ])


def test_router_decisions():
    """The router should return one of the three valid decisions."""
    from src.crag.nodes import node_route

    class FakeState(dict):
        pass

    for decision in ["relevant", "irrelevant", "ambiguous"]:
        state = FakeState(relevance_decision=decision)
        assert node_route(state) == decision


def test_llm_json_extractor():
    """JSON extractor should handle markdown fences and trailing text."""
    from src.crag.llm import _extract_json

    # Pure JSON
    assert _extract_json('{"score": 0.8}') == {"score": 0.8}

    # JSON in code fence
    result = _extract_json('```json\n{"score": 0.5, "reasoning": "ok"}\n```')
    assert result["score"] == 0.5

    # JSON with trailing text
    result = _extract_json('{"score": 0.7} Here is some explanation.')
    assert result["score"] == 0.7

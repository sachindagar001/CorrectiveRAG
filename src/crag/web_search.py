"""
Web Search Fallback — DuckDuckGo-powered web search for when the local
knowledge base fails to answer the query.

Used when the evaluator decides retrieved docs are irrelevant.
"""
from typing import List, Dict, Any
from src.crag import embeddings as emb


def web_search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Search DuckDuckGo and return results as RetrievedDoc-shaped dicts.

    Returns:
        [{"text": ..., "source": "web:duckduckgo", "title": ..., "url": ..., "score": 0.0}]
    """
    results: List[Dict[str, Any]] = []
    raw = []

    # Prefer the new `ddgs` package; fall back to legacy `duckduckgo_search`
    try:
        from ddgs import DDGS
        ddgs = DDGS()
        raw = list(ddgs.text(query, max_results=max_results))
    except ImportError:
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                raw = list(ddgs.text(query, max_results=max_results))
        except Exception as e:
            print(f"[web_search] DuckDuckGo error: {e}")
            return []
    except Exception as e:
        print(f"[web_search] DuckDuckGo error: {e}")
        return []

    for r in raw:
        body = r.get("body") or r.get("snippet") or r.get("description") or ""
        title = r.get("title") or ""
        url = r.get("href") or r.get("link") or r.get("url") or ""
        if not body and not title:
            continue
        text = f"{title}. {body}" if title and body else (title or body)
        results.append({
            "text": text,
            "source": "web:duckduckgo",
            "title": title,
            "url": url,
            "score": 0.0,  # filled in after re-embedding
        })

    # Re-rank web results by semantic similarity to the query
    if results:
        try:
            query_vec = emb.embed_query(query).astype("float32")
            doc_vecs = emb.embed_documents([r["text"] for r in results]).astype("float32")
            scores = doc_vecs @ query_vec
            for r, s in zip(results, scores):
                r["score"] = float(s)
            results.sort(key=lambda x: x["score"], reverse=True)
        except Exception:
            pass

    return results

"""
ArXiv data loader — loads pre-curated ML/AI paper abstracts from JSON.

If the JSON doesn't exist, falls back to fetching real papers via the arxiv API
(see scripts/fetch_arxiv.py for batch fetching).
"""
import json
import os
from typing import List, Dict, Any

# Path: src/data/loader.py -> project root is 3 dirnames up
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DATA_PATH = os.path.join(_PROJECT_ROOT, "data", "arxiv_papers.json")


def load_arxiv_papers(path: str = None) -> List[Dict[str, Any]]:
    """Load arxiv papers from the JSON file.

    Returns a list of dicts with keys: title, abstract, authors, arxiv_id, url, text, source.
    """
    path = path or DEFAULT_DATA_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Arxiv data file not found at {path}. "
            f"Run scripts/fetch_arxiv.py to generate it."
        )

    with open(path, "r", encoding="utf-8") as f:
        papers = json.load(f)

    # Normalize: ensure each paper has a 'text' field (used by the retriever)
    for p in papers:
        if "text" not in p or not p["text"]:
            p["text"] = f"{p.get('title', '')}. {p.get('abstract', '')}"
        if "source" not in p:
            p["source"] = f"arxiv:{p.get('arxiv_id', 'unknown')}"
    return papers


def get_paper_count(path: str = None) -> int:
    """Return the number of papers in the dataset."""
    try:
        return len(load_arxiv_papers(path))
    except Exception:
        return 0


def search_papers_by_keyword(keyword: str, path: str = None) -> List[Dict[str, Any]]:
    """Simple keyword filter over the papers (for the UI sidebar)."""
    papers = load_arxiv_papers(path)
    kw = keyword.lower()
    return [
        p for p in papers
        if kw in p.get("title", "").lower() or kw in p.get("abstract", "").lower()
    ]

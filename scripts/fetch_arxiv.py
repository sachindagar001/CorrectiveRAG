"""
Fetch real arXiv paper abstracts on RAG, LLMs, transformers, and related topics.

Usage:
    python scripts/fetch_arxiv.py [--max-per-topic 10] [--output data/arxiv_papers.json]

Topics covered:
    - Retrieval-Augmented Generation (RAG)
    - Large Language Models (LLMs)
    - Transformers / Attention
    - Vector databases / Embeddings
    - Agent workflows / Tool use
    - Hallucination detection
"""
import os
import sys
import argparse
import time
import json

# Make src importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import arxiv

OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "arxiv_papers.json",
)

TOPICS = [
    "retrieval augmented generation",
    "large language models",
    "transformer attention mechanism",
    "vector database embeddings",
    "LLM agents tool use",
    "hallucination detection language models",
    "instruction tuning",
    "chain of thought reasoning",
    "semantic search dense retrieval",
    "knowledge grounded generation",
    "prompt engineering",
    "fine-tuning language models",
]


def fetch_papers(topic: str, max_results: int = 10) -> list:
    """Fetch `max_results` arXiv papers for a given topic."""
    print(f"  Searching arXiv for: '{topic}' (max {max_results})...")
    client = arxiv.Client(num_retries=3, page_size=100)
    search = arxiv.Search(
        query=topic,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    papers = []
    try:
        for result in client.results(search):
            papers.append({
                "arxiv_id": result.entry_id.split("/abs/")[-1],
                "title": result.title.strip(),
                "abstract": result.summary.replace("\n", " ").strip(),
                "authors": [str(a) for a in result.authors][:5],
                "published": result.published.strftime("%Y-%m-%d") if result.published else "",
                "url": result.entry_id,
                "categories": [c for c in result.categories] if result.categories else [],
                "topic": topic,
            })
    except Exception as e:
        print(f"    ERROR fetching '{topic}': {e}")
    return papers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-per-topic", type=int, default=8,
                        help="Max papers per topic (default: 8)")
    parser.add_argument("--output", type=str, default=OUTPUT_PATH,
                        help="Output JSON path")
    args = parser.parse_args()

    print(f"Fetching arXiv papers across {len(TOPICS)} topics...")
    all_papers = []
    seen_ids = set()
    for topic in TOPICS:
        papers = fetch_papers(topic, max_results=args.max_per_topic)
        for p in papers:
            if p["arxiv_id"] not in seen_ids:
                p["text"] = f"{p['title']}. {p['abstract']}"
                p["source"] = f"arxiv:{p['arxiv_id']}"
                all_papers.append(p)
                seen_ids.add(p["arxiv_id"])
        time.sleep(1.0)  # be polite to arXiv

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_papers, f, indent=2, ensure_ascii=False)

    print(f"\nFetched {len(all_papers)} unique papers.")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()

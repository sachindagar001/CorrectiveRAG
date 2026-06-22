"""
Build the FAISS index from the arXiv papers dataset.

Usage:
    python scripts/build_index.py

This embeds all papers with BGE and saves the FAISS index to disk.
The Streamlit app loads this index at startup.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import load_arxiv_papers
from src.crag.retriever import FAISSRetriever
from src.crag import embeddings as emb


def build_index():
    print("=" * 60)
    print("  CRAG — Building FAISS Index")
    print("=" * 60)

    # 1. Load papers
    print("\n[1/3] Loading arXiv papers...")
    try:
        papers = load_arxiv_papers()
        print(f"  Loaded {len(papers)} papers.")
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        print("  Run scripts/fetch_arxiv.py first, or use data/arxiv_papers.json fallback.")
        sys.exit(1)

    if not papers:
        print("  ERROR: No papers found.")
        sys.exit(1)

    # 2. Load embedding model
    print("\n[2/3] Loading embedding model...")
    model_name = emb.get_model_name()
    print(f"  Model: {model_name}")
    _ = emb.embed_query("warmup")  # warmup
    dim = emb.get_embedding_dim()
    print(f"  Embedding dim: {dim}")

    # 3. Build & save FAISS index
    print("\n[3/3] Building FAISS index...")
    start = time.perf_counter()
    retriever = FAISSRetriever()
    retriever.build(papers)
    print(f"  Built index with {len(retriever)} documents in {time.perf_counter()-start:.2f}s")

    retriever.save()
    print(f"  Saved to: {retriever.index_path}.faiss (+ .meta.pkl)")
    print("\nDone. You can now run: streamlit run app.py")


if __name__ == "__main__":
    build_index()

"""
FAISS Retriever — Vector store over the document corpus using BGE embeddings.

Supports:
- Loading a pre-built index from disk
- Building a new index from a list of documents
- Top-K similarity search
- MMR-like diversity (optional)
"""
import os
import json
import pickle
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dotenv import load_dotenv

from src.crag import embeddings as emb

load_dotenv()

FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "index/faiss_index")
TOP_K = int(os.getenv("TOP_K_RETRIEVAL", "5"))


class FAISSRetriever:
    """A simple, self-contained FAISS retriever with document metadata."""

    def __init__(self, index_path: str = FAISS_INDEX_PATH):
        self.index_path = index_path
        self.index = None
        self.documents: List[Dict[str, Any]] = []  # [{text, source, title, ...}]
        self.dimension: int = 0
        self._loaded = False

    # ------------------------------------------------------------------ build
    def build(self, documents: List[Dict[str, Any]]) -> None:
        """Build a FAISS index from a list of documents.

        Each document dict should contain at least 'text' and 'source'.
        """
        import faiss

        if not documents:
            raise ValueError("Cannot build index from empty document list.")

        texts = [doc["text"] for doc in documents]
        vectors = emb.embed_documents(texts).astype(np.float32)
        self.dimension = vectors.shape[1]
        self.documents = documents

        # Use IndexFlatIP for cosine similarity (with normalized vectors)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(vectors)
        self._loaded = True

    # ----------------------------------------------------------------- save/load
    def save(self, path: Optional[str] = None) -> None:
        """Persist the FAISS index and document metadata to disk."""
        import faiss

        path = path or self.index_path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        faiss.write_index(self.index, f"{path}.faiss")
        with open(f"{path}.meta.pkl", "wb") as f:
            pickle.dump(
                {"documents": self.documents, "dimension": self.dimension},
                f,
            )

    def load(self, path: Optional[str] = None) -> bool:
        """Load the FAISS index from disk. Returns True if loaded successfully."""
        import faiss

        path = path or self.index_path
        faiss_path = f"{path}.faiss"
        meta_path = f"{path}.meta.pkl"

        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            return False

        self.index = faiss.read_index(faiss_path)
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        self.documents = meta["documents"]
        self.dimension = meta["dimension"]
        self._loaded = True
        return True

    # ----------------------------------------------------------------- search
    def search(
        self,
        query: str,
        top_k: int = TOP_K,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Search the index for the top-K most similar documents.

        Returns a list of dicts with keys: text, source, title, score.
        """
        if not self._loaded:
            raise RuntimeError("Retriever not loaded. Call load() or build() first.")

        query_vec = emb.embed_query(query).astype(np.float32).reshape(1, -1)
        k = min(top_k, len(self.documents))
        scores, indices = self.index.search(query_vec, k)

        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or score < min_score:
                continue
            doc = dict(self.documents[idx])
            doc["score"] = float(score)
            results.append(doc)
        return results

    def search_multi(
        self,
        queries: List[str],
        top_k: int = TOP_K,
    ) -> List[Dict[str, Any]]:
        """Search with multiple queries (used for multi-query expansion).
        Deduplicates and re-ranks by max score across queries.
        """
        seen: Dict[int, float] = {}  # doc_idx -> best score
        for q in queries:
            if not q.strip():
                continue
            query_vec = emb.embed_query(q).astype(np.float32).reshape(1, -1)
            k = min(top_k * 2, len(self.documents))
            scores, indices = self.index.search(query_vec, k)
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:
                    continue
                if idx not in seen or score > seen[idx]:
                    seen[idx] = float(score)

        # Sort by best score, take top_k
        ranked = sorted(seen.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results: List[Dict[str, Any]] = []
        for idx, score in ranked:
            doc = dict(self.documents[idx])
            doc["score"] = score
            results.append(doc)
        return results

    # ----------------------------------------------------------------- stats
    def __len__(self) -> int:
        return len(self.documents)

    def is_ready(self) -> bool:
        return self._loaded and self.index is not None

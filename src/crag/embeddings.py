"""
BGE Embeddings — HuggingFace BAAI/bge-small-en-v1.5 for query & document encoding.

This is a top-tier open embedding model on the MTEB leaderboard, runs locally,
and is small enough (~130MB) to load fast on CPU.
"""
import os

# Force single-threaded OpenMP before torch is imported. PyTorch's OpenMP
# runtime segfaults on Python 3.13 + macOS 26 (Apple Silicon) when running
# multi-threaded CPU kernels (e.g. LayerNorm). Setting OMP_NUM_THREADS=1
# disables the parallel kernel path that crashes. Must be set before
# sentence-transformers/torch are imported (done lazily in _get_model).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from typing import List
import numpy as np
from dotenv import load_dotenv

load_dotenv()

_MODEL = None
_MODEL_NAME = None


def _get_model():
    """Lazy-init the sentence-transformers model."""
    global _MODEL, _MODEL_NAME
    if _MODEL is not None:
        return _MODEL

    from sentence_transformers import SentenceTransformer

    _MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    _MODEL = SentenceTransformer(_MODEL_NAME, device="cpu")
    return _MODEL


def embed_texts(texts: List[str], normalize: bool = True) -> np.ndarray:
    """Embed a list of texts. Returns a (N, D) numpy array.

    BGE models work best with normalized embeddings + cosine similarity.
    """
    model = _get_model()
    if isinstance(texts, str):
        texts = [texts]
    embeddings = model.encode(
        texts,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return embeddings


def embed_query(text: str) -> np.ndarray:
    """Embed a single query. Returns a (D,) numpy array."""
    return embed_texts([text])[0]


def embed_documents(texts: List[str]) -> np.ndarray:
    """Embed multiple document chunks."""
    return embed_texts(texts)


def get_model_name() -> str:
    return os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")


def get_embedding_dim() -> int:
    """Return the embedding dimension."""
    return embed_query("test").shape[0]

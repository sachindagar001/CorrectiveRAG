"""
CRAG FastAPI Backend — exposes the Python CRAG pipeline as REST endpoints.

Runs on port 8000 (a mini-service). The Next.js frontend calls it via the
Caddy gateway using XTransformPort=8000.

Endpoints:
  GET  /api/health          — health check + system status
  GET  /api/papers          — list arXiv papers (with optional ?q= filter)
  GET  /api/topics          — topic distribution
  GET  /api/architecture    — mermaid diagram + node descriptions
  POST /api/query           — run the full CRAG pipeline on a query
"""
import os
import sys
import json
import time
import tempfile

# Force single-threaded OpenMP before any torch import. PyTorch's OpenMP
# runtime segfaults on Python 3.13 + macOS 26 (Apple Silicon) when running
# multi-threaded CPU kernels. Must be set before sentence-transformers/torch
# are imported (they're imported lazily below).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Make the project root importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# Ensure we have a usable temp directory (the /tmp may be full on small sandboxes)
# Use a project-local tmp dir as fallback
_tmp_dir = os.path.join(PROJECT_ROOT, ".tmp")
os.makedirs(_tmp_dir, exist_ok=True)
os.environ.setdefault("TMPDIR", _tmp_dir)
tempfile.tempdir = _tmp_dir

app = FastAPI(title="CRAG Agent API", version="1.0.0")

# CORS — allow the Next.js frontend to call us
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================ models
class QueryRequest(BaseModel):
    query: str
    use_query_rewriting: bool = True
    use_baseline: bool = True


class ConfigRequest(BaseModel):
    deepseek_api_key: str


# ============================================================ lazy singletons
_retriever = None
_papers_cache = None
_graph_cache: Dict[str, Any] = {}


def _get_retriever():
    global _retriever
    if _retriever is None:
        from src.crag.retriever import FAISSRetriever
        _retriever = FAISSRetriever()
        loaded = _retriever.load()
        if not loaded:
            raise RuntimeError(
                "FAISS index not loaded. Run: python scripts/build_index.py"
            )
    return _retriever


def _get_papers():
    global _papers_cache
    if _papers_cache is None:
        from src.data.loader import load_arxiv_papers
        _papers_cache = load_arxiv_papers()
    return _papers_cache


def _get_graph(use_baseline: bool = True):
    """Get or build a CRAG graph. Cache one per use_baseline setting."""
    key = f"baseline_{use_baseline}"
    if key not in _graph_cache:
        from src.crag.graph import CRAGGraph
        _graph_cache[key] = CRAGGraph(use_naive_baseline=use_baseline)
    return _graph_cache[key]


# ============================================================ helpers
def _check_deepseek_key() -> bool:
    key = os.getenv("DEEPSEEK_API_KEY", "")
    return bool(key) and key != "your_deepseek_api_key_here"


def _serialize(obj: Any) -> Any:
    """Recursively convert obj to JSON-serializable types."""
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize(x) for x in obj]
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    return str(obj)


# ============================================================ routes
@app.get("/api/health")
async def health():
    """Health check + system status (used by the sidebar)."""
    faiss_path_env = os.getenv("FAISS_INDEX_PATH", "index/faiss_index")
    faiss_path = os.path.join(PROJECT_ROOT, faiss_path_env) if not os.path.isabs(faiss_path_env) else faiss_path_env

    n_papers = 0
    try:
        n_papers = len(_get_papers())
    except Exception:
        pass

    retriever_ready = False
    try:
        r = _get_retriever()
        retriever_ready = r.is_ready()
    except Exception:
        retriever_ready = False

    return {
        "status": "ok",
        "deepseek_key_set": _check_deepseek_key(),
        "deepseek_model": os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        "deepseek_use_reasoning": os.getenv("DEEPSEEK_USE_REASONING", "false").lower() == "true",
        "embedding_model": os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        "faiss_index_built": os.path.exists(f"{faiss_path}.faiss"),
        "retriever_ready": retriever_ready,
        "n_papers": n_papers,
        "top_k": int(os.getenv("TOP_K_RETRIEVAL", "5")),
    }


@app.post("/api/config")
async def save_config(req: ConfigRequest):
    """Save the DeepSeek API key to the .env file and reload it in the current environment."""
    env_path = os.path.join(PROJECT_ROOT, ".env")
    lines = []
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            lines = f.readlines()
    
    key_found = False
    new_lines = []
    for line in lines:
        if line.strip().startswith("DEEPSEEK_API_KEY="):
            new_lines.append(f"DEEPSEEK_API_KEY={req.deepseek_api_key}\n")
            key_found = True
        else:
            new_lines.append(line)
            
    if not key_found:
        new_lines.append(f"DEEPSEEK_API_KEY={req.deepseek_api_key}\n")
        
    with open(env_path, "w") as f:
        f.writelines(new_lines)
        
    # Reload in current environment
    os.environ["DEEPSEEK_API_KEY"] = req.deepseek_api_key
    
    # Reset lazy singleton cache in llm module
    try:
        import src.crag.llm
        src.crag.llm._DEEPSEEK_API_KEY = req.deepseek_api_key
    except Exception:
        pass
        
    return {"status": "success", "message": "API key updated successfully"}


@app.get("/api/papers")
async def list_papers(q: Optional[str] = None, limit: int = 100):
    """List arXiv papers, optionally filtered by keyword."""
    papers = _get_papers()
    if q:
        ql = q.lower()
        papers = [
            p for p in papers
            if ql in p.get("title", "").lower()
            or ql in p.get("abstract", "").lower()
            or ql in p.get("topic", "").lower()
        ]
    return {"count": len(papers), "papers": papers[:limit]}


@app.get("/api/topics")
async def topic_distribution():
    """Return the topic distribution for the knowledge base chart."""
    papers = _get_papers()
    counts: Dict[str, int] = {}
    for p in papers:
        t = p.get("topic", "unknown")
        counts[t] = counts.get(t, 0) + 1
    items = sorted(
        [{"topic": t, "count": c} for t, c in counts.items()],
        key=lambda x: -x["count"],
    )
    return {"topics": items, "total_papers": len(papers)}


@app.get("/api/architecture")
async def architecture():
    """Return the mermaid diagram + node descriptions."""
    from src.crag.graph import CRAGGraph
    g = CRAGGraph(use_naive_baseline=True)
    return {
        "mermaid": g.get_mermaid(),
        "nodes": [
            {
                "name": "query_rewrite",
                "label": "Query Rewriter",
                "description": "Generates 3 alternative phrasings of the query (Multi-Query) AND a hypothetical answer document (HyDE). Both are used as retrieval queries to maximize recall.",
            },
            {
                "name": "retrieve",
                "label": "FAISS Retriever",
                "description": "Embeds all queries with BGE (bge-small-en-v1.5) and searches a FAISS IndexFlatIP. Merges results across queries and deduplicates by max score.",
            },
            {
                "name": "evaluate",
                "label": "Relevance Evaluator",
                "description": "LLM-as-judge grades each retrieved doc 0-1 for relevance to the query. Aggregator decides: relevant (>=0.6), irrelevant (<0.3), or ambiguous.",
            },
            {
                "name": "web_search",
                "label": "Web Search Fallback",
                "description": "Triggered when retrieval is irrelevant or ambiguous. Searches DuckDuckGo, re-ranks results by semantic similarity to the query.",
            },
            {
                "name": "knowledge_refinement",
                "label": "Knowledge Refinement",
                "description": "Strips noise from retrieved docs and web results. Keeps only docs with relevance >= 0.3, up to a 4000-char budget.",
            },
            {
                "name": "generate",
                "label": "Answer Generator",
                "description": "The LLM (configurable via DeepSeek — defaults to deepseek-v4-flash) generates the final answer using ONLY the refined knowledge as context. Instructed to cite sources inline.",
            },
            {
                "name": "hallucination_check",
                "label": "Hallucination Checker",
                "description": "LLM-as-judge scores how well the answer is grounded in the sources (0-1). Flags as hallucinated if below 0.5.",
            },
            {
                "name": "naive_rag",
                "label": "Naive RAG Baseline",
                "description": "For comparison: retrieves top-3 docs and generates directly without evaluation, web search, or refinement. Shows what 'naive RAG' would have produced.",
            },
        ],
    }


@app.post("/api/query")
async def run_query(req: QueryRequest):
    """Run the full CRAG pipeline on a user query."""
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    if not _check_deepseek_key():
        raise HTTPException(
            status_code=401,
            detail="DEEPSEEK_API_KEY is not set. Add it to .env and restart the backend.",
        )

    try:
        _get_retriever()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retriever init failed: {e}")

    try:
        graph = _get_graph(use_baseline=req.use_baseline)
        t0 = time.perf_counter()
        result = graph.run(
            query=req.query.strip(),
            use_query_rewriting=req.use_query_rewriting,
        )
        if "total_latency_ms" not in result or not result["total_latency_ms"]:
            result["total_latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        return _serialize(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"CRAG pipeline error: {e}")


@app.get("/")
async def root():
    return {"name": "CRAG Agent API", "version": "1.0.0", "docs": "/docs"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("CRAG_API_PORT", "8000"))
    print(f"Starting CRAG API on port {port}...")
    print(f"Project root: {PROJECT_ROOT}")
    # Use multiple workers + longer timeout so long CRAG queries don't block
    # health checks. Note: with multiple workers, the lazy singletons are
    # per-worker, but that's fine for this use case.
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
        timeout_keep_alive=300,  # 5 min keep-alive
    )

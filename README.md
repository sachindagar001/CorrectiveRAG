# CorrectiveRAG — Self-Correcting RAG Agent (CRAG)

> **Basic RAG is dead.** This agent evaluates its own retrieval quality, falls back to web search when local docs are irrelevant, and checks its own answers for hallucinations — all orchestrated as an 8-node LangGraph workflow.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-State%20Machine-green.svg)](https://github.com/langchain-ai/langgraph)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com)
[![Next.js](https://img.shields.io/badge/Frontend-Next.js%2014-black.svg)](https://nextjs.org)
[![FAISS](https://img.shields.io/badge/Vector%20DB-FAISS-orange.svg)](https://github.com/facebookresearch/faiss)
[![BGE](https://img.shields.io/badge/Embeddings-BGE%20bge--small--en--v1.5-purple.svg)](https://huggingface.co/BAAI/bge-small-en-v1.5)
[![DeepSeek](https://img.shields.io/badge/LLM-DeepSeek-blue.svg)](https://platform.deepseek.com)

---

## Why This Project?

Every company is building RAG (chat-with-your-data), but **naive RAG fails** when:
- The local knowledge base doesn't cover the question
- Retrieved docs are off-topic but the model still hallucinates from them
- The user asks something time-sensitive (e.g., "stock price today")

**CRAG (Corrective RAG)** solves this by adding a **self-correction loop**:

1. **Retrieve** docs from FAISS
2. **Evaluate** each doc with an LLM-as-judge (relevance score 0-1)
3. **Route** — if docs are irrelevant, fall back to DuckDuckGo web search
4. **Refine** knowledge (strip noise, keep only relevant chunks)
5. **Generate** the final answer (with citations)
6. **Check** the answer for hallucinations (grounding score 0-1)

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────┐
│  Query Rewriter             │  Multi-Query Expansion + HyDE
│  (3 alt phrasings + 1       │  (boosts retrieval recall)
│   hypothetical doc)         │
└────────────┬────────────────┘
             ▼
┌─────────────────────────────┐
│  FAISS Retriever            │  BGE embeddings (bge-small-en-v1.5)
│  (top-5 cosine similarity)  │  Multi-query merge & dedup
└────────────┬────────────────┘
             ▼
┌─────────────────────────────┐
│  Relevance Evaluator        │  LLM-as-Judge per doc
│  (score 0-1 per doc)        │  Aggregator → relevant/irrelevant/ambiguous
└────────────┬────────────────┘
             ▼
       ┌─────┴─────┬──────────┐
       ▼           ▼          ▼
   relevant   ambiguous   irrelevant
       │           │          │
       │           ▼          ▼
       │     ┌───────────┐
       │     │ Web Search│  DuckDuckGo API
       │     │ (5 results│  re-ranked semantically
       │     │  reranked)│
       │     └─────┬─────┘
       │           │
       └─────┬─────┘
             ▼
┌─────────────────────────────┐
│  Knowledge Refinement       │  Keep only docs with relevance ≥ 0.3
│  (token budget = 4000 chars)│  Merge local + web sources
└────────────┬────────────────┘
             ▼
┌─────────────────────────────┐
│  Answer Generator           │  LLM via DeepSeek
│  (cites sources inline)     │  Prompted to use ONLY the refined knowledge
└────────────┬────────────────┘
             ▼
┌─────────────────────────────┐
│  Hallucination Check        │  LLM-as-Judge scores grounding (0-1)
│  (flags if < 0.5)           │  Identifies unsupported claims
└────────────┬────────────────┘
             ▼
       Final Answer + Citations + Scores
```

---

## What Makes This Project Stand Out

| Feature | Why It Matters |
|---|---|
| **LangGraph workflow** | Shows understanding of Agentic AI, not just prompt engineering |
| **LLM-as-Judge evaluation** | Industry-standard technique for RAG evaluation (RAGAS, etc.) |
| **Web search fallback** | Demonstrates routing & graceful degradation patterns |
| **Hallucination scoring** | Shows NLP depth — every answer has a grounding score |
| **Query Rewriting (Multi-Query + HyDE)** | Advanced retrieval techniques beyond basic similarity search |
| **CRAG vs Naive RAG comparison** | Proves the architecture adds value, side-by-side |
| **Per-node latency dashboard** | Production-grade observability |
| **Citation tracking** | Every claim is traceable to a source |
| **Full-stack** | Python ML backend + React/TypeScript frontend, not just a notebook |

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/sachindagar001/CorrectiveRAG.git
cd CorrectiveRAG

# Python setup
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Node.js setup
npm install
```

### 2. Get a DeepSeek API Key

1. Go to [platform.deepseek.com](https://platform.deepseek.com)
2. Sign up and create an API key (starts with `sk-`)
3. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
4. Edit `.env` and paste your key:
   ```
   DEEPSEEK_API_KEY=sk-your_actual_key_here
   ```

### 3. Run the App

**Option A — Start both servers (recommended):**
```bash
bash scripts/start_servers.sh
```

**Option B — Start manually (two terminals):**
```bash
# Terminal 1: Backend
source venv/bin/activate
python mini-services/crag-api/server.py

# Terminal 2: Frontend
npm run dev
```

Open **http://localhost:3000** and start asking questions.

> The FAISS index is pre-built and included in the repo, so you can skip `python scripts/build_index.py`.

---

## Try These Demo Questions

| Question | What it demonstrates |
|---|---|
| "What is retrieval-augmented generation?" | Standard RAG — hits local KB |
| "How does chain-of-thought prompting work?" | Standard RAG — hits local KB |
| "What is HyDE and when is it useful?" | Finds the HyDE paper directly |
| "What are the latest techniques to detect hallucinations in LLMs?" | Multi-doc synthesis |
| "What is the stock price of Apple today?" | Forces web search fallback (not in arXiv KB) |
| "Who won the latest FIFA World Cup?" | Forces web search fallback |

---

## Tech Stack

| Layer | Tech |
|---|---|
| **Orchestration** | LangGraph |
| **LLM** | DeepSeek (OpenAI-compatible API) |
| **Embeddings** | BAAI/bge-small-en-v1.5 (HuggingFace, local) |
| **Vector Store** | FAISS (IndexFlatIP, cosine similarity) |
| **Web Search** | DuckDuckGo (`ddgs`) |
| **Backend** | FastAPI + Python |
| **Frontend** | Next.js 14, React, TypeScript, Tailwind CSS, shadcn/ui |
| **Data** | 50 arXiv paper abstracts (RAG, LLMs, transformers, agents) |

---

## Project Structure

```
CorrectiveRAG/
├── app.py                          # Streamlit UI (legacy)
├── requirements.txt                # Python dependencies
├── package.json                    # Node.js dependencies
├── .env.example                    # Environment variable template
│
├── data/
│   └── arxiv_papers.json           # 50 ML/AI paper abstracts (knowledge base)
│
├── index/                          # Pre-built FAISS index (ready to use)
│   ├── faiss_index.faiss
│   └── faiss_index.meta.pkl
│
├── src/
│   ├── crag/
│   │   ├── state.py                # CRAGState TypedDict
│   │   ├── graph.py                # LangGraph workflow + conditional routing
│   │   ├── nodes.py                # 8 node functions
│   │   ├── retriever.py            # FAISS retriever with BGE embeddings
│   │   ├── evaluator.py            # LLM-as-judge relevance scorer
│   │   ├── query_rewriter.py       # Multi-Query + HyDE
│   │   ├── web_search.py           # DuckDuckGo fallback
│   │   ├── hallucination.py        # Grounding score checker
│   │   ├── llm.py                  # DeepSeek LLM wrapper (retry + fallback)
│   │   └── embeddings.py           # BGE embedding wrapper
│   └── data/
│       └── loader.py               # arXiv papers JSON loader
│
├── mini-services/
│   └── crag-api/
│       └── server.py               # FastAPI backend (5 REST endpoints)
│
├── src/ (frontend)
│   ├── app/                        # Next.js app router
│   ├── components/crag/            # Chat, architecture, dashboard, KB components
│   └── lib/crag-api.ts             # TypeScript API client
│
├── scripts/
│   ├── build_index.py              # Build FAISS index from papers
│   ├── fetch_arxiv.py              # Fetch fresh arXiv papers (optional)
│   └── start_servers.sh            # Start both servers at once
│
├── tests/
│   └── test_crag.py                # Unit tests
│
└── index/                          # (generated) FAISS index files
```

---

## Configuration (`.env`)

| Variable | Default | Description |
|---|---|---|
| `DEEPSEEK_API_KEY` | — | **Required.** Get from [platform.deepseek.com](https://platform.deepseek.com) |
| `DEEPSEEK_MODEL` | `deepseek-v4-flash` | DeepSeek model (fast, free-tier friendly) |
| `DEEPSEEK_FALLBACK_MODEL` | `deepseek-chat` | Fallback if primary model fails |
| `DEEPSEEK_USE_REASONING` | `false` | Set to `true` for reasoning models (much slower) |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | HuggingFace sentence-transformers model |
| `FAISS_INDEX_PATH` | `index/faiss_index` | Path to the FAISS index (without extension) |
| `TOP_K_RETRIEVAL` | `5` | Number of docs to retrieve |
| `RELEVANCE_THRESHOLD` | `0.5` | Below this → trigger web search |
| `HALLUCINATION_THRESHOLD` | `0.5` | Below this → flag as hallucinated |

---

## Key Concepts Demonstrated

### 1. Corrective RAG (CRAG)
The core idea from [Yan et al., 2024](https://arxiv.org/abs/2401.15884): don't trust retrieval blindly. Evaluate it, and fall back to other sources when it fails.

### 2. LangGraph State Machine
The workflow is a directed graph where each node is a pure function `(state) → partial_state`. LangGraph handles the orchestration, routing, and parallel execution.

### 3. LLM-as-Judge
Using an LLM to grade the output of another LLM call. This is the same technique used by [RAGAS](https://arxiv.org/abs/2309.15217), [Self-RAG](https://arxiv.org/abs/2310.11511), and most modern RAG eval frameworks.

### 4. Multi-Query Expansion
Generate 3 alternative phrasings of the query, retrieve for all of them, and merge results. Boosts recall by catching docs that match different vocabulary.

### 5. HyDE (Hypothetical Document Embeddings)
Generate a hypothetical answer to the query, then embed THAT (not the query) for retrieval. The intuition: a fake-but-plausible answer is closer in embedding space to real answers than the short query is. From [Gao et al., 2022](https://arxiv.org/abs/2210.07128).

### 6. Hallucination Detection
After generating the answer, a separate LLM call scores how well every claim is supported by the source documents. This is a production-grade pattern for high-stakes RAG deployments.

---

## Run Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## LinkedIn Post

If you found this project helpful and want to share it, here's a post template:

> Basic RAG is dead. 🪦
>
> I built a Self-Correcting RAG Agent that doesn't just retrieve and generate — it evaluates its own retrieval quality, falls back to web search when local docs are irrelevant, and scores its own answers for hallucinations.
>
> The pipeline runs as an 8-node LangGraph state machine:
> query_rewrite → retrieve → evaluate → [route] → web_search → refine → generate → hallucination_check
>
> What makes it different from naive RAG:
> ▸ LLM-as-judge evaluates every retrieved doc (relevance score 0-1)
> ▸ Falls back to DuckDuckGo web search when retrieval is irrelevant
> ▸ Query rewriting with Multi-Query + HyDE for better recall
> ▸ Hallucination checker scores how well the answer is grounded in sources
> ▸ Side-by-side CRAG vs Naive RAG comparison
> ▸ Per-node latency dashboard for observability
>
> Tech stack: LangGraph, DeepSeek, BGE embeddings, FAISS, FastAPI, Next.js 14
>
> Code: https://github.com/sachindagar001/CorrectiveRAG
>
> #RAG #LangGraph #LLM #MachineLearning #AI #AgenticAI #FullStack

---

## License

MIT — feel free to use this for your portfolio, job applications, or as a starting point for production RAG systems.

---

## Acknowledgments

- [LangGraph](https://github.com/langchain-ai/langgraph) — agentic workflow framework
- [BGE embeddings](https://huggingface.co/BAAI/bge-small-en-v1.5) — top-tier open embeddings
- [DeepSeek](https://platform.deepseek.com) — fast, OpenAI-compatible LLM API
- [CRAG paper](https://arxiv.org/abs/2401.15884) — the original Corrective RAG idea
- [Self-RAG](https://arxiv.org/abs/2310.11511) — inspiration for self-reflection in RAG
- [HyDE](https://arxiv.org/abs/2210.07128) — hypothetical document embeddings

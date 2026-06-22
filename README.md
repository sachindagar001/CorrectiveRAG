# CorrectiveRAG — Self-Correcting RAG Agent (CRAG)

> **Basic RAG is dead.** This agent evaluates its own retrieval quality, falls back to web search when local docs are irrelevant, and checks its own answers for hallucinations — all orchestrated as a LangGraph workflow.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2.34-green.svg)](https://github.com/langchain-ai/langgraph)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.39-red.svg)](https://streamlit.io)
[![FAISS](https://img.shields.io/badge/FAISS-IndexFlatIP-orange.svg)](https://github.com/facebookresearch/faiss)
[![BGE](https://img.shields.io/badge/BGE-bge--small--en--v1.5-purple.svg)](https://huggingface.co/BAAI/bge-small-en-v1.5)
[![OpenRouter](https://img.shields.io/badge/LLM-OpenRouter%20Multi--Model-black.svg)](https://openrouter.ai)

---

## 📌 Why This Project?

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

## 🏗️ Architecture

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
│  Answer Generator           │  LLM via OpenRouter (Llama/Gemini/etc.)
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

## 🧪 What Makes This Project Stand Out

| Feature | Why It Matters |
|---|---|
| **LangGraph workflow** | Shows you understand Agentic AI, not just prompt engineering |
| **LLM-as-Judge evaluation** | Industry-standard technique for RAG evaluation (RAGAS, etc.) |
| **Web search fallback** | Demonstrates routing & graceful degradation patterns |
| **Hallucination scoring** | Shows NLP depth — every answer has a grounding score |
| **Query Rewriting (Multi-Query + HyDE)** | Advanced retrieval techniques beyond basic similarity search |
| **CRAG vs Naive RAG comparison** | Proves the architecture adds value, side-by-side |
| **Per-node latency dashboard** | Production-grade observability |
| **Citation tracking** | Every claim is traceable to a source |

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone <your-repo-url>
cd crag-agent

# Create virtual env (optional but recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Get a Free OpenRouter API Key

1. Go to [openrouter.ai/keys](https://openrouter.ai/keys)
2. Sign up (free) and create an API key
3. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
4. Edit `.env` and paste your key:
   ```
   OPENROUTER_API_KEY=sk-or-v1-your_actual_key_here
   ```
5. (Optional) Change the model — free options include:
   - `meta-llama/llama-3.1-8b-instruct:free` (default, fast)
   - `meta-llama/llama-3.3-70b-instruct:free` (better quality)
   - `google/gemini-2.0-flash-exp:free` (very fast)
   - `nex-agi/nex-n2-pro:free` (reasoning model — slower, see note below)
   - `mistralai/mistral-7b-instruct:free`
   - `qwen/qwen-2.5-7b-instruct:free`

### 3. Build the FAISS Index

```bash
python scripts/build_index.py
```

This loads the arXiv papers from `data/arxiv_papers.json`, embeds them with BGE, and saves the FAISS index to `index/`.

> **Optional:** Refresh the paper dataset with real arXiv papers:
> ```bash
> python scripts/fetch_arxiv.py --max-per-topic 10
> python scripts/build_index.py
> ```

### 4. Run the Streamlit App

```bash
streamlit run app.py
```

Open `http://localhost:8501` and start asking questions!

---

## 🎯 Try These Demo Questions

| Question | What it demonstrates |
|---|---|
| "What is retrieval-augmented generation?" | Standard RAG — should hit local KB |
| "How does chain-of-thought prompting work?" | Standard RAG — should hit local KB |
| "What is HyDE and when is it useful?" | Should find the HyDE paper directly |
| "What are the latest techniques to detect hallucinations in LLMs?" | Multi-doc synthesis |
| "What is the stock price of Apple today?" | Forces web search fallback (not in arXiv KB) |
| "Who won the latest FIFA World Cup?" | Forces web search fallback |

---

## 📊 Evaluation Dashboard

The app includes a built-in eval dashboard that tracks:
- **Total queries** run in the session
- **Average latency** (total + per-node breakdown)
- **Average relevance score** (from the evaluator)
- **Average grounding score** (from the hallucination check)
- **Web search trigger count** (how often fallback was used)
- **CRAG vs Naive RAG** side-by-side comparison
- **CSV export** of all query results

---

## 📁 Project Structure

```
crag-agent/
├── app.py                          # Streamlit UI (chat + dashboard + arch viz)
├── requirements.txt
├── .env.example
├── README.md
│
├── data/
│   └── arxiv_papers.json           # 50+ ML/AI paper abstracts (RAG, LLMs, etc.)
│
├── src/
│   ├── crag/
│   │   ├── state.py                # CRAGState TypedDict
│   │   ├── graph.py                # LangGraph workflow definition
│   │   ├── nodes.py                # All node functions (retrieve, evaluate, etc.)
│   │   ├── retriever.py            # FAISS retriever with BGE embeddings
│   │   ├── evaluator.py            # LLM-as-judge relevance scorer
│   │   ├── query_rewriter.py       # Multi-Query + HyDE
│   │   ├── web_search.py           # DuckDuckGo fallback
│   │   ├── hallucination.py        # Grounding score checker
│   │   ├── llm.py                  # OpenRouter LLM wrapper (multi-model)
│   │   └── embeddings.py           # BGE embedding wrapper
│   └── data/
│       └── loader.py               # arXiv papers JSON loader
│
├── scripts/
│   ├── build_index.py              # Build FAISS index from papers
│   └── fetch_arxiv.py              # Fetch fresh arXiv papers (optional)
│
├── tests/
│   └── test_crag.py                # Unit tests
│
└── index/                          # (generated) FAISS index files
    ├── faiss_index.faiss
    └── faiss_index.meta.pkl
```

---

## 🔧 Configuration (`.env`)

| Variable | Default | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | — | **Required.** Get from [openrouter.ai/keys](https://openrouter.ai/keys) |
| `OPENROUTER_MODEL` | `meta-llama/llama-3.1-8b-instruct:free` | Any OpenRouter model. Free: `meta-llama/llama-3.3-70b-instruct:free`, `google/gemini-2.0-flash-exp:free`, `nex-agi/nex-n2-pro:free` (reasoning), `mistralai/mistral-7b-instruct:free`, `qwen/qwen-2.5-7b-instruct:free` |
| `OPENROUTER_USE_REASONING` | `false` | Set to `true` to enable reasoning (only works on reasoning-capable models like `nex-agi/nex-n2-pro:free`). Much slower. |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | HuggingFace sentence-transformers model |
| `FAISS_INDEX_PATH` | `index/faiss_index` | Path to the FAISS index (without extension) |
| `TOP_K_RETRIEVAL` | `5` | Number of docs to retrieve |
| `RELEVANCE_THRESHOLD` | `0.5` | Below this → trigger web search |
| `HALLUCINATION_THRESHOLD` | `0.5` | Below this → flag as hallucinated |

---

## 🧠 Key Concepts Demonstrated

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

## 🛠️ Tech Stack

| Layer | Tech |
|---|---|
| **Orchestration** | LangGraph |
| **LLM** | Any model via OpenRouter (Llama 3.1, Gemini, Mistral, GPT-4o, etc.) — defaults to free Llama 3.1 8B |
| **Embeddings** | BAAI/bge-small-en-v1.5 (HuggingFace, local) |
| **Vector Store** | FAISS (IndexFlatIP, cosine similarity) |
| **Web Search** | DuckDuckGo (`duckduckgo-search`) |
| **UI** | Streamlit |
| **Data** | 50+ arXiv paper abstracts (RAG, LLMs, transformers, agents) |

---

## 🧠 Reasoning Models (Optional)

OpenRouter supports reasoning-capable models like `nex-agi/nex-n2-pro:free`, `openai/o1-mini`, and `deepseek/deepseek-r1`. These models "think before they speak" — they produce a hidden chain-of-thought before the final answer.

To enable reasoning in CRAG:

1. Set `OPENROUTER_MODEL=nex-agi/nex-n2-pro:free` in `.env`
2. Set `OPENROUTER_USE_REASONING=true` in `.env`

**Trade-offs:**
- ✅ Better quality on complex reasoning (math, multi-step logic)
- ❌ Much slower (10-60s per call vs 1-3s for non-reasoning models)
- ❌ The evaluator makes 5+ LLM calls per query, so reasoning makes the whole pipeline ~5x slower
- 💡 **Recommendation:** Keep reasoning OFF for the demo. Use it only if you want to show off the reasoning capability for a specific query.

The OpenRouter integration preserves `reasoning_details` across multi-turn conversations, so you can build agentic loops where the model continues reasoning from where it left off.

---

## 🧪 Run Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## 📈 LinkedIn Post Template

> Basic RAG is dead. 🪦
>
> I built a **Self-Correcting RAG Agent** that:
> ✅ Evaluates its own retrieval quality with an LLM-as-judge
> ✅ Falls back to DuckDuckGo web search when local docs are irrelevant
> ✅ Rewrites queries with Multi-Query + HyDE for better recall
> ✅ Scores its own answers for hallucinations (grounding score 0-1)
> ✅ Shows CRAG vs Naive RAG side-by-side
>
> Built with: LangGraph, FAISS, BGE embeddings, OpenRouter LLMs (Llama/Gemini/etc.), Streamlit
>
> The agent uses a LangGraph state machine with 8 nodes:
> query_rewrite → retrieve → evaluate → [route] → web_search → refine → generate → hallucination_check
>
> Here's the architecture... [screenshot]
>
> Code: [GitHub link]
>
> #RAG #LangGraph #LLM #MachineLearning #AI #NLP

---

## 🎯 What to Show Recruiters

1. **The architecture diagram** (Architecture tab in the app)
2. **A web-search fallback query** (e.g., "stock price today") — shows the routing in action
3. **The eval dashboard** — shows you think about metrics, not just features
4. **The hallucination score** — every answer has one, proves NLP depth
5. **The CRAG vs Naive RAG comparison** — proves the architecture adds value

---

## 📝 License

MIT — feel free to use this for your portfolio, job applications, or as a starting point for production RAG systems.

---

## 🙏 Acknowledgments

- [LangGraph](https://github.com/langchain-ai/langgraph) — agentic workflow framework
- [BGE embeddings](https://huggingface.co/BAAI/bge-small-en-v1.5) — top-tier open embeddings
- [OpenRouter](https://openrouter.ai) — single API for many LLMs (free tier available)
- [CRAG paper](https://arxiv.org/abs/2401.15884) — the original Corrective RAG idea
- [Self-RAG](https://arxiv.org/abs/2310.11511) — inspiration for self-reflection in RAG
- [HyDE](https://arxiv.org/abs/2210.07128) — hypothetical document embeddings

# CRAG Agent — Setup Guide

> Complete step-by-step instructions to run the Self-Correcting RAG Agent on your machine.

---

## 1. System Requirements

### Minimum Hardware
| Resource | Minimum | Recommended | What We Used |
|---|---|---|---|
| **RAM** | 4 GB | 8 GB | 8 GB |
| **Disk space** | 5 GB free | 8 GB free | 8 GB |
| **CPU** | 2 cores | 4 cores | 4 cores |
| **Internet** | Required (for DeepSeek API + DuckDuckGo + model download) | — | — |

### Why these requirements?
- **RAM**: The BGE embedding model (~500MB) + FAISS index + Next.js dev server + FastAPI = ~2.5GB RAM in use. 4GB is the bare minimum.
- **Disk**: Python venv with torch + transformers = ~1.5GB. Node modules = ~1.3GB. Plus build caches. 5GB free is safe.
- **CPU**: BGE embeddings run on CPU. 2 cores works but is slow; 4 cores is comfortable.

### Software Prerequisites

| Software | Version | Why |
|---|---|---|
| **Python** | 3.10, 3.11, or 3.12 | Tested with 3.12.13. 3.13+ may work but untested. |
| **Node.js** | 18+ | Tested with v24.16.0. Required for Next.js. |
| **bun** (optional) | 1.0+ | Faster than npm. Tested with 1.3.14. |
| **pip** | 23+ | For Python package installation |
| **git** | any | To clone the repo (if using git) |

---

## 2. Get the Code

### Option A: Download the ZIP (easiest)

Download `crag-agent-source.zip` (~234 KB) and unzip it:

```bash
unzip crag-agent-source.zip -d crag-agent
cd crag-agent
```

### Option B: Clone from git (if you pushed to GitHub)

```bash
git clone https://github.com/yourusername/crag-agent.git
cd crag-agent
```

### What's in the download

```
crag-agent/
├── app.py                          # (Legacy Streamlit UI — not used, kept for reference)
├── ARCHITECTURE.md                 # Full architecture guide for your LinkedIn video
├── README.md                       # Project overview
├── requirements.txt                # Python dependencies
├── .env.example                    # Template for your .env file
├── .gitignore
│
├── data/
│   └── arxiv_papers.json           # 50 ML/AI paper abstracts (knowledge base)
│
├── index/                          # Pre-built FAISS index (ready to use!)
│   ├── faiss_index.faiss           # 76 KB — the vector index
│   └── faiss_index.meta.pkl        # 101 KB — document metadata
│
├── src/                            # Python CRAG pipeline (the ML/DL code)
│   ├── crag/
│   │   ├── state.py                # CRAGState TypedDict
│   │   ├── graph.py                # LangGraph workflow definition
│   │   ├── nodes.py                # 9 node functions
│   │   ├── retriever.py            # FAISS retriever
│   │   ├── evaluator.py            # LLM-as-judge relevance scorer
│   │   ├── query_rewriter.py       # Multi-Query + HyDE
│   │   ├── web_search.py           # DuckDuckGo fallback
│   │   ├── hallucination.py        # Grounding score checker
│   │   ├── llm.py                  # DeepSeek LLM wrapper
│   │   └── embeddings.py           # BGE embedding wrapper
│   └── data/
│       └── loader.py               # arXiv papers JSON loader
│
├── mini-services/
│   └── crag-api/
│       ├── server.py               # FastAPI backend
│       └── package.json            # Service definition
│
├── scripts/
│   ├── build_index.py              # Rebuild FAISS index from papers
│   ├── fetch_arxiv.py              # Fetch fresh arXiv papers (optional)
│   └── start_servers.sh            # Start both servers at once
│
└── tests/
    └── test_crag.py                # 12 unit tests
```

---

## 3. Installation Steps

### Step 1: Create a Python virtual environment

```bash
cd crag-agent

# Create venv
python3 -m venv venv

# Activate it
source venv/bin/activate        # Linux/Mac
# OR
venv\Scripts\activate           # Windows
```

### Step 2: Install Python dependencies

```bash
# Install torch FIRST (CPU-only, smaller download — ~190MB vs 750MB)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install the rest
pip install -r requirements.txt
```

This installs:
- `langgraph` — workflow orchestration
- `fastapi` + `uvicorn` — backend API
- `sentence-transformers` — BGE embeddings
- `faiss-cpu` — vector database
- `transformers` — HuggingFace model loading
- `requests` — DeepSeek API calls
- `ddgs` — DuckDuckGo web search
- `python-dotenv` — env file loading
- `pydantic` — data validation
- `arxiv` — paper fetching (optional)
- `pytest` — testing

### Step 3: Install Node.js dependencies

```bash
# Using bun (faster — recommended)
bun install

# OR using npm (slower but works)
npm install
```

This installs Next.js 16, React, TypeScript, Tailwind CSS, shadcn/ui, mermaid, and all UI components.

### Step 4: Get a free DeepSeek API key

1. Go to **https://platform.deepseek.com**
2. Sign up (free)
3. Create an API key (starts with `sk-`)
4. Copy the key

### Step 5: Configure your environment

```bash
# Copy the template
cp .env.example .env

# Edit .env with your editor (nano, vim, VS Code, etc.)
nano .env
```

Set these values in `.env`:

```env
# Your DeepSeek API key (required)
DEEPSEEK_API_KEY=sk-your-actual-key-here

# Model (deepseek-v4-flash is fast and free-tier friendly)
DEEPSEEK_MODEL=deepseek-v4-flash
DEEPSEEK_FALLBACK_MODEL=deepseek-chat
DEEPSEEK_USE_REASONING=false

# Embeddings (runs locally, no API key needed)
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5

# FAISS index path
FAISS_INDEX_PATH=index/faiss_index

# Retrieval settings
TOP_K_RETRIEVAL=5
RELEVANCE_THRESHOLD=0.5
HALLUCINATION_THRESHOLD=0.5
```

### Step 6: Verify the FAISS index exists

The download includes a pre-built FAISS index (`index/faiss_index.faiss` + `index/faiss_index.meta.pkl`), so you can skip building it.

If you want to rebuild it (e.g., after adding more papers):

```bash
python scripts/build_index.py
```

### Step 7: Run the tests (optional but recommended)

```bash
python -m pytest tests/ -v
```

You should see all 12 tests pass.

---

## 4. Running the Application

### Option A: Use the startup script (Linux/Mac)

```bash
bash scripts/start_servers.sh
```

This starts both:
- **FastAPI backend** on port 8000
- **Next.js frontend** on port 3000

### Option B: Start servers manually (in 2 terminals)

**Terminal 1 — Backend:**
```bash
source venv/bin/activate
python mini-services/crag-api/server.py
```

**Terminal 2 — Frontend:**
```bash
bun run dev
# OR
npm run dev
```

### Step 8: Open the app

Open your browser to: **http://localhost:3000**

You should see:
- Sidebar showing "DeepSeek API Key — Configured"
- "FAISS Index — Built"
- "Retriever — Ready"
- 4 tabs: Chat, Architecture, Eval Dashboard, Knowledge Base

### Step 9: Try a query

Go to the **Chat** tab and type:
```
What is retrieval-augmented generation?
```

Click **Run CRAG**. Wait ~30-45 seconds (the pipeline makes 6+ LLM calls). You'll see:
- The answer with inline citations like [arxiv:2401.15884]
- Relevance score (0-1)
- Grounding score (0-1)
- Execution trace showing all 8 nodes
- CRAG vs Naive RAG comparison

---

## 5. Troubleshooting

### "ModuleNotFoundError: No module named 'X'"

Make sure your venv is activated:
```bash
source venv/bin/activate
```

Then reinstall:
```bash
pip install -r requirements.txt
```

### "Connection refused" on port 8000 or 3000

The servers died. Restart them:
```bash
bash scripts/start_servers.sh
```

### "No space left on device"

The Python venv + node_modules take ~3GB. Free up space:
```bash
# Clear pip cache
pip cache purge

# Clear npm cache
npm cache clean --force

# Clear Next.js build cache
rm -rf .next
```

### "DeepSeek API key not set"

Edit your `.env` file and make sure `DEEPSEEK_API_KEY` is set to your actual key (not the placeholder).

### Query takes too long (>60 seconds)

The first query is slow because the BGE embedding model loads into memory (~5 seconds). Subsequent queries should be faster (~30-45 seconds for the full pipeline).

If still slow:
- Set `DEEPSEEK_USE_REASONING=false` in `.env` (reasoning models are much slower)
- Disable "Compare vs Naive RAG" in the sidebar (saves 1 LLM call)

### DuckDuckGo web search not working

The `ddgs` package sometimes gets rate-limited. Try:
```bash
pip install --upgrade ddgs
```

---

## 6. System Configuration Summary

Here's exactly what you need:

### For Development (running locally)

| Component | Size | Required? |
|---|---|---|
| Python 3.10-3.12 | ~50MB | Yes |
| Python venv + packages | ~1.5GB | Yes |
| Node.js 18+ | ~50MB | Yes |
| node_modules | ~1.3GB | Yes |
| Project source code | ~62MB | Yes |
| FAISS index (pre-built) | ~180KB | Yes (included) |
| DeepSeek API key | Free | Yes |
| **Total disk needed** | **~3GB** | |
| **Total RAM needed** | **~3GB active** | |

### For Production (deploying)

If you want to deploy this to a cloud server (AWS, GCP, HuggingFace Spaces, etc.):

| Component | Requirement |
|---|---|
| **Server** | 2 vCPU, 4GB RAM minimum |
| **Storage** | 10GB SSD |
| **Python** | 3.10-3.12 |
| **Node.js** | 18+ (for building the frontend) |
| **Ports** | 80 (HTTP) or 443 (HTTPS) |
| **API keys** | DeepSeek (required) |

For HuggingFace Spaces (free tier):
- Use a Docker Space
- Expose port 7860
- The free tier has 16GB RAM and 2 vCPUs — plenty

---

## 7. Quick Start (TL;DR)

```bash
# 1. Unzip
unzip crag-agent-source.zip -d crag-agent
cd crag-agent

# 2. Python setup
python3 -m venv venv
source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 3. Node setup
bun install  # or: npm install

# 4. Configure
cp .env.example .env
# Edit .env and add your DEEPSEEK_API_KEY

# 5. Run
bash scripts/start_servers.sh

# 6. Open
# http://localhost:3000
```

Total time: **~10-15 minutes** (most of it is package installation).

---

## 8. What Each File Does (for your understanding)

### The CRAG Pipeline (`src/crag/`)

| File | Purpose | Lines |
|---|---|---|
| `state.py` | Defines `CRAGState` TypedDict — the data that flows through the pipeline | ~50 |
| `graph.py` | Builds the LangGraph workflow with conditional routing | ~220 |
| `nodes.py` | 9 node functions: query_rewrite, retrieve, evaluate, web_search, etc. | ~280 |
| `retriever.py` | FAISS vector store with BGE embeddings | ~150 |
| `evaluator.py` | LLM-as-judge that grades retrieval relevance 0-1 | ~100 |
| `query_rewriter.py` | Multi-Query expansion + HyDE | ~80 |
| `web_search.py` | DuckDuckGo fallback with semantic re-ranking | ~60 |
| `hallucination.py` | LLM-as-judge that scores answer grounding 0-1 | ~80 |
| `llm.py` | DeepSeek API wrapper with retry + fallback | ~270 |
| `embeddings.py` | BGE embedding model wrapper | ~60 |

### The Backend (`mini-services/crag-api/`)

| File | Purpose |
|---|---|
| `server.py` | FastAPI app with 5 endpoints: /api/health, /api/papers, /api/topics, /api/architecture, /api/query |

### The Frontend (`src/`)

| File | Purpose |
|---|---|
| `src/app/page.tsx` | Main page with 4 tabs + sidebar |
| `src/lib/crag-api.ts` | TypeScript API client |
| `src/components/crag/sidebar.tsx` | Status, settings, config display |
| `src/components/crag/chat-tab.tsx` | Chat input + result panel |
| `src/components/crag/architecture-tab.tsx` | Mermaid diagram + node docs |
| `src/components/crag/eval-dashboard-tab.tsx` | Metrics + CRAG vs Naive RAG |
| `src/components/crag/knowledge-base-tab.tsx` | Paper browser |

---

## 9. Next Steps

After you have it running:

1. **Read `ARCHITECTURE.md`** — Full architecture guide for your LinkedIn video
2. **Record your demo** — Use the suggested video script in ARCHITECTURE.md
3. **Post on LinkedIn** — Use the post template in ARCHITECTURE.md
4. **Customize** — Add your own papers to `data/arxiv_papers.json` and rebuild the index with `python scripts/build_index.py`

---

## 10. Need Help?

If something doesn't work:

1. Check the **Troubleshooting** section above
2. Run the tests: `python -m pytest tests/ -v`
3. Check the API health: `curl http://localhost:8000/api/health`
4. Check the Next.js dev log: `tail -f dev.log`
5. Check the FastAPI log: `tail -f /tmp/crag_api.log`

---

**Happy building!** This project demonstrates agentic AI workflows, LLM-as-judge evaluation, advanced retrieval techniques, and full-stack development — all highly valued skills in the current ML/AI job market.

# CRAG Agent — Architecture Guide for LinkedIn Video

> A self-correcting RAG system that evaluates its own retrieval quality, falls back to web search, and checks its answers for hallucinations.

---

## 1. The Problem with Naive RAG

Naive RAG (Retrieval-Augmented Generation) has 3 critical flaws:

1. **Blind trust in retrieval** — If the retriever returns irrelevant docs, the LLM still generates an answer from them (hallucination)
2. **No fallback** — If the local knowledge base doesn't cover the question, the system fails or hallucinates
3. **No self-checking** — There's no mechanism to verify if the generated answer is actually grounded in the sources

**CRAG (Corrective RAG)** solves all 3 problems.

---

## 2. System Architecture (High-Level)

```
┌─────────────────────────────────────────────────────────────┐
│                    USER (Browser)                            │
│                  Next.js + React UI                          │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP (REST API)
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Caddy Reverse Gateway (port 81)                 │
│   Routes / → Next.js, /api/*?XTransformPort=8000 → FastAPI  │
└────────────────────────┬────────────────────────────────────┘
                         │
            ┌────────────┴────────────┐
            ▼                         ▼
┌───────────────────┐        ┌───────────────────┐
│  Next.js UI       │        │  FastAPI Backend  │
│  (port 3000)      │        │  (port 8000)      │
│                   │        │                   │
│  - Chat tab       │        │  /api/health      │
│  - Architecture   │        │  /api/papers      │
│  - Eval Dashboard │        │  /api/topics      │
│  - Knowledge Base │        │  /api/architecture│
│                   │        │  /api/query       │
└───────────────────┘        └────────┬──────────┘
                                      │
                                      ▼
                          ┌───────────────────────┐
                          │  CRAG Pipeline        │
                          │  (LangGraph workflow) │
                          └────────┬──────────────┘
                                   │
                ┌──────────────────┼──────────────────┐
                ▼                  ▼                  ▼
        ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
        │ FAISS Index  │  │ DeepSeek LLM │  │ DuckDuckGo   │
        │ (50 arXiv    │  │ (v4-flash)   │  │ Web Search   │
        │  papers)     │  │              │  │              │
        │ BGE embeds   │  │  - Evaluator │  │  - Fallback  │
        │ (384-dim)    │  │  - Generator │  │    when local│
        │              │  │  - Halluc.   │  │    docs fail │
        │              │  │    checker   │  │              │
        └──────────────┘  └──────────────┘  └──────────────┘
```

---

## 3. The CRAG Pipeline (8 Nodes)

This is the heart of the project. Each node is a Python function that transforms a typed state dictionary. LangGraph orchestrates the flow.

### Node 1: Query Rewriter
**What it does:** Takes the user's question and generates:
- 3 alternative phrasings (Multi-Query Expansion)
- 1 hypothetical answer document (HyDE)

**Why:** Different phrasings match different vocabulary in the docs. HyDE works because a fake-but-plausible answer is closer in embedding space to real answers than the short question is.

**Example:**
- Input: "What is RAG?"
- Multi-Query outputs: "How does retrieval-augmented generation work?", "Explain RAG in NLP", "What is the RAG technique?"
- HyDE output: "RAG combines a retrieval step with a generative LLM. The retriever finds relevant documents from a knowledge base, and the generator uses those documents as context to produce grounded answers..."

### Node 2: FAISS Retriever
**What it does:** Embeds all the queries (original + 3 rewrites + HyDE doc) using BGE, searches a FAISS index of 50 arXiv paper abstracts, and returns the top-5 most similar docs.

**Why multi-query:** Boosts recall by catching docs that match different phrasings. The retriever merges results across queries and deduplicates by max score.

### Node 3: Relevance Evaluator (LLM-as-Judge)
**What it does:** For each retrieved doc, an LLM grades its relevance to the query on a scale of 0.0 to 1.0. Then an aggregator decides:
- **relevant** (at least one doc ≥ 0.6) → use the docs
- **irrelevant** (all docs < 0.3) → trigger web search
- **ambiguous** (mixed signals) → use both docs AND web search

**Why this is the key innovation:** This is what makes it "self-correcting." Instead of blindly trusting retrieval, the system evaluates whether the retrieved docs actually help answer the question.

### Node 4: Web Search Fallback
**What it does:** If the evaluator says "irrelevant" or "ambiguous", search DuckDuckGo for the query, retrieve 5 results, and re-rank them by semantic similarity to the query.

**Why:** When the local knowledge base doesn't cover the question (e.g., "What's Apple's stock price today?"), the system gracefully falls back to the web instead of hallucinating.

### Node 5: Knowledge Refinement
**What it does:** Strips noise from the retrieved docs and web results. Keeps only docs with relevance ≥ 0.3, up to a 4000-character budget. Merges local + web sources into a single "refined_knowledge" string.

**Why:** Removes irrelevant chunks that would otherwise distract the generator.

### Node 6: Answer Generator
**What it does:** DeepSeek LLM generates the final answer using ONLY the refined knowledge as context. Prompted to cite sources inline like [arxiv:2401.15884].

**Why "ONLY":** This is what prevents hallucination — the model is explicitly told not to use its parametric knowledge, only the provided context.

### Node 7: Hallucination Checker (LLM-as-Judge)
**What it does:** A separate LLM call scores how well the generated answer is grounded in the sources (0.0 to 1.0). Identifies unsupported claims. Flags as hallucinated if score < 0.5.

**Why:** This is the second self-correction mechanism. Even after generation, the system checks its own work.

### Node 8: Naive RAG Baseline (for comparison)
**What it does:** Runs a simple retrieve-top-3 → generate pipeline (no evaluation, no web search, no refinement, no hallucination check). This is what "naive RAG" would have produced.

**Why:** The Eval Dashboard shows CRAG vs Naive RAG side-by-side, proving the architecture adds value.

---

## 4. The LangGraph Workflow

```
START
  │
  ▼
query_rewrite ──────────► retrieve ──────────► evaluate
                                                  │
                                          [router decision]
                                          /       |        \
                                  relevant   ambiguous   irrelevant
                                      │        │           │
                                      │        ▼           ▼
                                      │   web_search   web_search
                                      │        │           │
                                      └────────┴───────────┘
                                               │
                                               ▼
                                     knowledge_refinement
                                               │
                                               ▼
                                          generate
                                               │
                                               ▼
                                    hallucination_check
                                               │
                                               ▼
                                    naive_rag (baseline)
                                               │
                                               ▼
                                             END
```

**Key concept:** The router after `evaluate` is a conditional edge — it routes to different nodes based on the evaluator's decision. This is what makes it "agentic" rather than a linear pipeline.

---

## 5. Tech Stack (and why each was chosen)

| Layer | Technology | Why |
|---|---|---|
| **Workflow orchestration** | LangGraph | Industry-standard for agentic workflows. State machine model fits CRAG perfectly. |
| **LLM** | DeepSeek v4-flash | Fast (~1-2s per call), no rate limits, OpenAI-compatible API |
| **Embeddings** | BAAI/bge-small-en-v1.5 | Top-tier open model on MTEB leaderboard, runs locally, 384-dim (small) |
| **Vector DB** | FAISS (IndexFlatIP) | Facebook's library, fast cosine similarity, no server needed |
| **Web search** | DuckDuckGo (ddgs package) | Free, no API key, no rate limits |
| **Backend** | FastAPI | Async, fast, automatic OpenAPI docs |
| **Frontend** | Next.js 16 + TypeScript + shadcn/ui | Modern React, type-safe, beautiful components |
| **Knowledge base** | 50 arXiv paper abstracts | Real ML/AI papers on RAG, LLMs, transformers, hallucination |

---

## 6. Sample Query Walkthrough (for the video)

**User asks:** "How does chain-of-thought prompting work?"

### Step 1: Query Rewrite
- 3 alternative phrasings generated
- HyDE document generated (a hypothetical answer)

### Step 2: Retrieve
- All 4 queries (original + 3 rewrites + HyDE) embedded with BGE
- FAISS searches 50 arXiv papers, returns top-5
- Top result: [arxiv:2201.11903] "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models" (score: 0.69)

### Step 3: Evaluate
- LLM grades each of the 5 docs 0-1
- Aggregator decision: **relevant** (top doc scored ≥ 0.6)
- No web search needed

### Step 4: Refine
- Keep docs with relevance ≥ 0.3
- Merge into a 4000-char context string

### Step 5: Generate
- DeepSeek generates the answer using ONLY the refined context
- Cites sources inline: [arxiv:2201.11903]

### Step 6: Hallucination Check
- LLM scores grounding: **1.00** (every claim supported by sources)

### Step 7: Naive RAG Baseline
- Same query, but simple retrieve → generate (no evaluation, no refinement)
- Shown side-by-side in the Eval Dashboard

**Final result:**
- Total latency: 38.7s
- Relevance: 1.00
- Grounding: 1.00
- 5 citations

---

## 7. What Makes This Project Stand Out (for recruiters)

1. **Agentic workflow** — Not just prompt engineering. Shows understanding of state machines, routing, and conditional logic.
2. **LLM-as-Judge** — Industry-standard technique used by RAGAS, Self-RAG, and production RAG systems.
3. **Self-correction** — The system evaluates its own retrieval AND its own answers.
4. **Graceful degradation** — Falls back to web search when local knowledge fails.
5. **Advanced retrieval** — Multi-Query + HyDE, not just basic similarity search.
6. **Production patterns** — Per-node latency tracking, hallucination scoring, citation tracking.
7. **A/B comparison** — CRAG vs Naive RAG side-by-side proves the architecture adds value.
8. **Full-stack** — Python backend + React frontend + vector DB + LLM API integration.

---

## 8. Suggested Video Script (60-90 seconds)

**Hook (5s):** "Basic RAG is dead. I built a self-correcting RAG agent that evaluates its own retrieval, falls back to web search, and checks its answers for hallucinations."

**Problem (10s):** "Naive RAG has 3 problems: it trusts retrieval blindly, has no fallback, and never checks its own work."

**Solution (15s):** "My CRAG agent fixes this with an 8-node LangGraph workflow. The key innovation is an LLM-as-judge that grades every retrieved document 0 to 1. If the docs are irrelevant, it automatically falls back to DuckDuckGo web search."

**Demo (30s):** [Screen record the app]
- Ask "What is chain-of-thought prompting?"
- Show the result panel: relevance 1.00, grounding 1.00, 5 citations
- Switch to Architecture tab, show the mermaid diagram
- Switch to Eval Dashboard, show CRAG vs Naive RAG comparison

**Tech stack (10s):** "Built with LangGraph, FAISS, BGE embeddings, DeepSeek v4-flash, and a Next.js UI. The knowledge base is 50 real arXiv papers on RAG and LLMs."

**Call to action (5s):** "Link in comments. Let me know what you think!"

---

## 9. LinkedIn Post Template

> Basic RAG is dead.
>
> I built a Self-Correcting RAG Agent (CRAG) that:
> - Evaluates its own retrieval quality with an LLM-as-judge
> - Falls back to DuckDuckGo web search when local docs are irrelevant
> - Rewrites queries with Multi-Query + HyDE for better recall
> - Scores its own answers for hallucinations (grounding score 0-1)
> - Shows CRAG vs Naive RAG side-by-side
>
> Architecture: 8-node LangGraph workflow
> query_rewrite → retrieve → evaluate → [route] → web_search → refine → generate → hallucination_check
>
> Tech stack:
> - LangGraph (workflow orchestration)
> - DeepSeek v4-flash (LLM)
> - BGE bge-small-en-v1.5 (embeddings)
> - FAISS (vector database)
> - DuckDuckGo (web search fallback)
> - Next.js + TypeScript + shadcn/ui (frontend)
> - FastAPI (backend)
> - 50 real arXiv papers as knowledge base
>
> The agent uses 2 LLM-as-judge calls per query:
> 1. Evaluator grades each retrieved doc 0-1 for relevance
> 2. Hallucination checker scores how well the answer is grounded in sources
>
> If retrieval is irrelevant, it automatically searches the web. If the answer isn't grounded, it flags it.
>
> This is what production RAG looks like — not just prompt engineering.
>
> #RAG #LangGraph #LLM #MachineLearning #AI #NLP #DeepSeek #FAISS

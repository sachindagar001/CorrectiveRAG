"""
CRAG Agent — Streamlit UI
A Self-Correcting RAG Agent that evaluates its own retrieval quality and falls
back to web search when local docs are insufficient.

Usage:
    streamlit run app.py

Run these first:
    1. python scripts/build_index.py   (builds the FAISS index)
    2. Set GROQ_API_KEY in .env
"""
import os
import sys
import time
import json
from datetime import datetime

# Add project root to path
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ----------------------------------------------------------------- page config
st.set_page_config(
    page_title="CRAG Agent — Self-Correcting RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------- custom CSS
# Clean Pro theme: light bg, blue accents, Inter font
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
    --primary: #2563eb;
    --primary-dark: #1e40af;
    --primary-light: #dbeafe;
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
    --bg: #ffffff;
    --surface: #f8fafc;
    --border: #e2e8f0;
    --text: #1e293b;
    --text-muted: #64748b;
}

* {
    font-family: 'Inter', sans-serif !important;
}

.block-container {
    padding-top: 1.5rem !important;
    max-width: 1400px !important;
}

/* App title */
.app-title {
    font-size: 2rem !important;
    font-weight: 800 !important;
    color: var(--text) !important;
    margin-bottom: 0 !important;
    letter-spacing: -0.025em !important;
}
.app-subtitle {
    font-size: 0.95rem !important;
    color: var(--text-muted) !important;
    font-weight: 400 !important;
    margin-top: 0 !important;
}

/* Stat cards */
.stat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
    transition: all 0.2s ease;
}
.stat-card:hover {
    border-color: var(--primary);
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.08);
}
.stat-value {
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    color: var(--primary) !important;
    line-height: 1.1 !important;
}
.stat-label {
    font-size: 0.75rem !important;
    color: var(--text-muted) !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 0.3rem !important;
}

/* Section headers */
.section-header {
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    color: var(--text) !important;
    margin-top: 1.5rem !important;
    margin-bottom: 0.6rem !important;
    padding-bottom: 0.5rem !important;
    border-bottom: 2px solid var(--border);
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* Answer box */
.answer-box {
    background: linear-gradient(135deg, #f8fafc 0%, #eff6ff 100%);
    border: 1px solid var(--border);
    border-left: 4px solid var(--primary);
    border-radius: 8px;
    padding: 1.2rem 1.4rem;
    line-height: 1.6;
    color: var(--text);
    font-size: 0.95rem;
}

/* Hallucination badge */
.badge {
    display: inline-block;
    padding: 0.25rem 0.7rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-left: 0.5rem;
}
.badge-success { background: #d1fae5; color: #065f46; }
.badge-warning { background: #fef3c7; color: #92400e; }
.badge-danger { background: #fee2e2; color: #991b1b; }

/* Decision pill */
.decision-pill {
    display: inline-block;
    padding: 0.4rem 1rem;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 600;
}
.decision-relevant { background: #d1fae5; color: #065f46; }
.decision-ambiguous { background: #fef3c7; color: #92400e; }
.decision-irrelevant { background: #fee2e2; color: #991b1b; }

/* Citation cards */
.citation-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.6rem;
    font-size: 0.85rem;
    transition: all 0.2s;
}
.citation-card:hover {
    border-color: var(--primary);
    transform: translateY(-1px);
}

/* Latency bars */
.latency-bar {
    height: 8px;
    background: var(--border);
    border-radius: 4px;
    overflow: hidden;
    margin-top: 0.3rem;
}
.latency-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--primary) 0%, #60a5fa 100%);
    border-radius: 4px;
    transition: width 0.5s ease;
}

/* Node trace pills */
.trace-pill {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    background: var(--primary-light);
    color: var(--primary-dark);
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 500;
    margin: 0.15rem 0.15rem;
}
.trace-pill.active {
    background: var(--primary);
    color: white;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--text) !important;
}

/* Status indicator */
.status-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 0.4rem;
}
.status-ready { background: var(--success); box-shadow: 0 0 0 3px #d1fae5; }
.status-error { background: var(--danger); box-shadow: 0 0 0 3px #fee2e2; }
.status-pending { background: var(--warning); box-shadow: 0 0 0 3px #fef3c7; }

/* Divider */
hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 1.5rem 0 !important;
}

/* Buttons */
.stButton > button {
    background: var(--primary) !important;
    color: white !important;
    border: none !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.5rem !important;
    border-radius: 8px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: var(--primary-dark) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2) !important;
}

/* Code blocks */
.stCodeBlock {
    border-radius: 8px !important;
    border: 1px solid var(--border) !important;
}

/* Chat input */
.stChatInput textarea {
    border-radius: 12px !important;
    border: 1px solid var(--border) !important;
}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------- sidebar
def render_sidebar():
    """Render the sidebar with system info and controls."""
    with st.sidebar:
        st.markdown("## 🧠 CRAG Agent")
        st.caption("Self-Correcting Retrieval-Augmented Generation")

        st.markdown("---")

        # System status
        st.markdown("### System Status")

        # Check DeepSeek API key
        ds_key = os.getenv("DEEPSEEK_API_KEY", "")
        if ds_key and ds_key != "your_deepseek_api_key_here":
            st.markdown("✅ **DeepSeek API Key** — Configured")
        else:
            st.markdown("❌ **DeepSeek API Key** — Missing")
            st.caption("Get a key from [platform.deepseek.com](https://platform.deepseek.com)")

        # Check FAISS index
        faiss_path = os.getenv("FAISS_INDEX_PATH", "index/faiss_index")
        index_exists = os.path.exists(f"{faiss_path}.faiss")
        if index_exists:
            st.markdown("✅ **FAISS Index** — Built")
        else:
            st.markdown("⚠️ **FAISS Index** — Not built")
            st.caption("Run: `python scripts/build_index.py`")

        # Check papers dataset
        papers_path = os.path.join(ROOT, "data", "arxiv_papers.json")
        if os.path.exists(papers_path):
            try:
                with open(papers_path) as f:
                    n_papers = len(json.load(f))
                st.markdown(f"✅ **Knowledge Base** — {n_papers} arXiv papers")
            except Exception:
                st.markdown("⚠️ **Knowledge Base** — Error loading")
        else:
            st.markdown("❌ **Knowledge Base** — Missing")

        st.markdown("---")

        # Settings
        st.markdown("### Pipeline Settings")

        use_qr = st.toggle(
            "Query Rewriting (Multi-Query + HyDE)",
            value=True,
            help="Generates 3 alternative phrasings + a hypothetical document for retrieval.",
        )

        use_baseline = st.toggle(
            "Compare vs Naive RAG",
            value=True,
            help="Also runs a naive RAG baseline to show the difference.",
        )

        st.markdown("---")

        # Model info
        st.markdown("### Configuration")
        st.caption(f"**LLM Provider:** DeepSeek")
        st.caption(f"**LLM Model:** `{os.getenv('DEEPSEEK_MODEL', 'deepseek-v4-flash')}`")
        st.caption(f"**Reasoning:** {os.getenv('DEEPSEEK_USE_REASONING', 'false')}")
        st.caption(f"**Embeddings:** {os.getenv('EMBEDDING_MODEL', 'BAAI/bge-small-en-v1.5')}")
        st.caption(f"**Vector DB:** FAISS (IndexFlatIP)")
        st.caption(f"**Top-K:** {os.getenv('TOP_K_RETRIEVAL', '5')}")

        st.markdown("---")

        # About
        st.markdown("### About")
        st.caption(
            "Built with LangGraph, FAISS, BGE embeddings, and DeepSeek LLMs. "
            "The agent retrieves docs, evaluates them with an LLM-as-judge, and falls "
            "back to DuckDuckGo web search when retrieval is irrelevant."
        )

        return {"use_query_rewriting": use_qr, "use_baseline": use_baseline}


# ----------------------------------------------------------------- main
def main():
    # Header
    st.markdown("# 🧠 CRAG Agent — Self-Correcting RAG")
    st.markdown(
        "<p class='app-subtitle'>"
        "Evaluates retrieval quality, falls back to web search when needed, "
        "and checks its own answer for hallucinations."
        "</p>",
        unsafe_allow_html=True,
    )

    settings = render_sidebar()

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "history" not in st.session_state:
        st.session_state.history = []  # list of full state dicts
    if "graph" not in st.session_state:
        st.session_state.graph = None

    # ----- Tabs -----
    tab_chat, tab_arch, tab_eval, tab_data = st.tabs([
        "💬 Chat",
        "🔧 Architecture",
        "📊 Eval Dashboard",
        "📚 Knowledge Base",
    ])

    # ===================================================== TAB 1: CHAT
    with tab_chat:
        col_input, col_result = st.columns([1.2, 1])

        with col_input:
            st.markdown("### Ask a Question")
            st.caption("Try questions about RAG, LLMs, transformers, hallucination, or agent workflows.")

            # Sample questions
            sample_qs = [
                "What is retrieval-augmented generation?",
                "How does chain-of-thought prompting work?",
                "What is HyDE and when is it useful?",
                "What are the latest techniques to detect hallucinations in LLMs?",
                "What is the stock price of Apple today?",  # forces web search
            ]
            with st.expander("💡 Try a sample question"):
                for q in sample_qs:
                    if st.button(q, key=f"sample_{q}"):
                        st.session_state["pending_query"] = q
                        st.rerun()

            # Manual input
            query = st.text_area(
                "Your question:",
                value=st.session_state.get("pending_query", ""),
                height=80,
                placeholder="e.g., What is CRAG and how does it improve on naive RAG?",
                key="chat_input",
            )

            col_btn1, col_btn2 = st.columns([1, 1])
            run_clicked = col_btn1.button("🚀 Run CRAG", type="primary", use_container_width=True)
            clear_clicked = col_btn2.button("🗑️ Clear", use_container_width=True)

            if clear_clicked:
                st.session_state.messages = []
                st.session_state.history = []
                if "pending_query" in st.session_state:
                    del st.session_state["pending_query"]
                st.rerun()

            if run_clicked and query.strip():
                _run_crag(query.strip(), settings)

            # Chat history
            if st.session_state.messages:
                st.markdown("---")
                st.markdown("### Conversation")
                for msg in st.session_state.messages:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])

        with col_result:
            _render_last_result()

    # ===================================================== TAB 2: ARCHITECTURE
    with tab_arch:
        _render_architecture(settings)

    # ===================================================== TAB 3: EVAL DASHBOARD
    with tab_eval:
        _render_eval_dashboard()

    # ===================================================== TAB 4: KNOWLEDGE BASE
    with tab_data:
        _render_knowledge_base()


# ----------------------------------------------------------------- helpers
def _run_crag(query: str, settings: dict):
    """Execute the CRAG pipeline on the user's query."""
    # Validate setup
    ds_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not ds_key or ds_key == "your_deepseek_api_key_here":
        st.error("❌ DeepSeek API key not set. Add DEEPSEEK_API_KEY to your .env file.")
        return

    faiss_path = os.getenv("FAISS_INDEX_PATH", "index/faiss_index")
    if not os.path.exists(f"{faiss_path}.faiss"):
        st.error("❌ FAISS index not built. Run: `python scripts/build_index.py`")
        return

    # Lazy-import graph (so UI loads fast)
    with st.spinner("Loading CRAG pipeline..."):
        try:
            from src.crag.graph import CRAGGraph
            if st.session_state.graph is None:
                st.session_state.graph = CRAGGraph(
                    use_naive_baseline=settings["use_baseline"]
                )
            graph = st.session_state.graph
        except Exception as e:
            st.error(f"❌ Failed to initialize CRAG graph: {e}")
            return

    # Run
    with st.spinner(f"🤔 Running CRAG on: '{query[:60]}...'"):
        t0 = time.perf_counter()
        result = graph.run(
            query=query,
            use_query_rewriting=settings["use_query_rewriting"],
        )
        elapsed = time.perf_counter() - t0

    if result.get("error"):
        st.error(f"❌ Pipeline error: {result['error']}")
        return

    # Save to history
    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.messages.append({"role": "assistant", "content": result["final_answer"]})
    st.session_state.history.append(result)
    st.session_state.last_result = result

    # Clear pending query
    if "pending_query" in st.session_state:
        del st.session_state["pending_query"]

    st.rerun()


def _render_last_result():
    """Render the most recent CRAG result in the right column."""
    result = st.session_state.get("last_result")
    if not result:
        st.info("👈 Run a query to see the CRAG pipeline in action.")
        return

    st.markdown("### 🎯 Latest Result")

    # Top stat row
    cols = st.columns(4)
    with cols[0]:
        _stat_card(f"{result['total_latency_ms']/1000:.2f}s", "Total Latency")
    with cols[1]:
        _stat_card(f"{result['overall_relevance']:.2f}", "Relevance Score")
    with cols[2]:
        _stat_card(f"{result['hallucination_score']:.2f}", "Grounding Score")
    with cols[3]:
        _stat_card(str(len(result.get("citations", []))), "Citations")

    st.markdown("")

    # Decision pill
    decision = result.get("relevance_decision", "ambiguous")
    pill_class = f"decision-{decision}"
    st.markdown(
        f"<div style='margin: 0.5rem 0;'>"
        f"<span class='decision-pill {pill_class}'>"
        f"🔍 Retrieval Decision: {decision.upper()}"
        f"</span></div>",
        unsafe_allow_html=True,
    )

    # Web search indicator
    if result.get("web_search_used"):
        st.markdown(
            "<span class='decision-pill decision-ambiguous'>"
            "🌐 Web Search Triggered (DuckDuckGo)"
            "</span>",
            unsafe_allow_html=True,
        )

    # Hallucination badge
    h_score = result["hallucination_score"]
    if h_score >= 0.7:
        badge_class = "badge-success"
        badge_text = "✓ Well-Grounded"
    elif h_score >= 0.4:
        badge_class = "badge-warning"
        badge_text = "⚠ Partially Grounded"
    else:
        badge_class = "badge-danger"
        badge_text = "⚠ Likely Hallucinated"
    st.markdown(
        f"<span class='badge {badge_class}'>{badge_text}</span>",
        unsafe_allow_html=True,
    )

    st.markdown("")

    # Answer
    st.markdown("#### 💬 Answer")
    st.markdown(
        f"<div class='answer-box'>{_format_answer(result['final_answer'])}</div>",
        unsafe_allow_html=True,
    )

    # Evaluator reasoning
    if result.get("evaluator_reasoning"):
        st.markdown("##### 🔎 Evaluator Reasoning")
        st.caption(result["evaluator_reasoning"])

    # Hallucination reasoning
    if result.get("hallucination_reasoning"):
        st.markdown("##### 🛡️ Hallucination Check")
        st.caption(result["hallucination_reasoning"])

    # Node trace
    st.markdown("##### 🔄 Execution Trace")
    trace_html = " → ".join(
        f"<span class='trace-pill active'>{n}</span>"
        for n in result.get("node_trace", [])
    )
    st.markdown(f"<div>{trace_html}</div>", unsafe_allow_html=True)


def _render_architecture(settings):
    """Render the architecture tab with the LangGraph diagram."""
    st.markdown("### 🔧 CRAG Architecture")
    st.caption(
        "Built with LangGraph. Each node is a function that transforms the state. "
        "The router after `evaluate` decides whether to use retrieved docs, "
        "search the web, or both."
    )

    # Mermaid diagram
    st.markdown("#### Workflow Graph")
    try:
        from src.crag.graph import CRAGGraph
        graph = CRAGGraph(use_naive_baseline=settings["use_baseline"])
        mermaid_code = graph.get_mermaid()
        try:
            from streamlit_mermaid import st_mermaid
            st_mermaid(mermaid_code, height=500)
        except ImportError:
            st.code(mermaid_code, language="mermaid")
            st.caption("Install `streamlit-mermaid` to render this diagram visually.")
    except Exception as e:
        st.error(f"Could not render graph: {e}")

    # Node descriptions
    st.markdown("---")
    st.markdown("#### Node Descriptions")

    nodes_info = [
        ("query_rewrite", "Query Rewriter",
         "Generates 3 alternative phrasings of the query (Multi-Query) AND a hypothetical answer document (HyDE). Both are used as retrieval queries to maximize recall."),
        ("retrieve", "FAISS Retriever",
         "Embeds all queries with BGE (bge-small-en-v1.5) and searches a FAISS IndexFlatIP. Merges results across queries and deduplicates by max score."),
        ("evaluate", "Relevance Evaluator",
         "LLM-as-judge grades each retrieved doc 0-1 for relevance to the query. Aggregator decides: relevant (≥0.6), irrelevant (<0.3), or ambiguous."),
        ("web_search", "Web Search Fallback",
         "Triggered when retrieval is irrelevant or ambiguous. Searches DuckDuckGo, re-ranks results by semantic similarity to the query."),
        ("knowledge_refinement", "Knowledge Refinement",
         "Strips noise from retrieved docs and web results. Keeps only docs with relevance ≥0.3, up to a 4000-char budget."),
        ("generate", "Answer Generator",
         "The LLM (configurable via DeepSeek — defaults to deepseek-v4-flash) generates the final answer using ONLY the refined knowledge as context. Instructed to cite sources inline."),
        ("hallucination_check", "Hallucination Checker",
         "LLM-as-judge scores how well the answer is grounded in the sources (0-1). Flags as hallucinated if below 0.5."),
        ("naive_rag", "Naive RAG Baseline",
         "For comparison: retrieves top-3 docs and generates directly without evaluation, web search, or refinement. Shows what 'naive RAG' would have produced."),
    ]

    for name, label, desc in nodes_info:
        st.markdown(f"**`{name}`** — {label}")
        st.caption(desc)

    # State schema
    st.markdown("---")
    st.markdown("#### State Schema (TypedDict)")
    with st.expander("View full CRAGState definition"):
        st.code("""
class CRAGState(TypedDict):
    original_query: str
    conversation_history: List[Dict[str, str]]

    # Query rewriting
    rewritten_queries: List[str]
    hyde_document: Optional[str]
    use_query_rewriting: bool

    # Retrieval
    retrieved_docs: List[RetrievedDoc]
    retrieval_method: str

    # Evaluation
    relevance_scores: List[float]
    overall_relevance: float
    relevance_decision: str  # "relevant" | "irrelevant" | "ambiguous"
    evaluator_reasoning: str

    # Knowledge refinement
    refined_knowledge: str

    # Web search
    web_search_used: bool
    web_search_results: List[RetrievedDoc]
    web_search_query: str

    # Generation
    final_answer: str
    citations: List[Dict[str, Any]]

    # Hallucination
    hallucination_score: float
    hallucination_reasoning: str
    is_hallucinated: bool

    # Baseline
    naive_rag_answer: str

    # Observability
    node_trace: List[str]
    latency_ms: Dict[str, float]
    total_latency_ms: float
    timestamp: str
    error: Optional[str]
""".strip())


def _render_eval_dashboard():
    """Render the eval dashboard tab with metrics across all queries."""
    st.markdown("### 📊 Evaluation Dashboard")
    st.caption("Aggregated metrics across all queries in this session.")

    history = st.session_state.get("history", [])

    if not history:
        st.info("No queries run yet. Run a query in the Chat tab to populate the dashboard.")
        return

    # Summary stats
    st.markdown("#### Session Summary")
    cols = st.columns(5)
    with cols[0]:
        _stat_card(str(len(history)), "Total Queries")
    with cols[1]:
        avg_lat = sum(h["total_latency_ms"] for h in history) / len(history)
        _stat_card(f"{avg_lat/1000:.2f}s", "Avg Latency")
    with cols[2]:
        avg_rel = sum(h["overall_relevance"] for h in history) / len(history)
        _stat_card(f"{avg_rel:.2f}", "Avg Relevance")
    with cols[3]:
        avg_hall = sum(h["hallucination_score"] for h in history) / len(history)
        _stat_card(f"{avg_hall:.2f}", "Avg Grounding")
    with cols[4]:
        ws_count = sum(1 for h in history if h.get("web_search_used"))
        _stat_card(str(ws_count), "Web Searches")

    st.markdown("---")

    # Per-node latency breakdown (latest)
    if history:
        st.markdown("#### Per-Node Latency (Latest Query)")
        latest = history[-1]
        latencies = latest.get("latency_ms", {})
        if latencies:
            max_lat = max(latencies.values()) if latencies else 1
            for node, ms in sorted(latencies.items(), key=lambda x: -x[1]):
                pct = (ms / max_lat) * 100 if max_lat > 0 else 0
                st.markdown(
                    f"<div style='display:flex; align-items:center; gap:0.8rem; margin:0.4rem 0;'>"
                    f"<div style='width:140px; font-weight:500; font-size:0.85rem;'>{node}</div>"
                    f"<div style='flex:1;'><div class='latency-bar'>"
                    f"<div class='latency-fill' style='width:{pct}%;'></div>"
                    f"</div></div>"
                    f"<div style='width:80px; text-align:right; font-weight:600; font-size:0.85rem;'>{ms:.0f} ms</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    st.markdown("---")

    # CRAG vs Naive RAG comparison
    st.markdown("#### CRAG vs Naive RAG")
    st.caption("Side-by-side comparison of the last query.")

    if "naive_rag_answer" in history[-1] and history[-1]["naive_rag_answer"]:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🧠 CRAG (Self-Correcting)**")
            st.caption(f"Grounding: {history[-1]['hallucination_score']:.2f} | "
                      f"Decision: {history[-1]['relevance_decision']}")
            st.markdown(
                f"<div class='answer-box' style='font-size:0.85rem; min-height:200px;'>"
                f"{_format_answer(history[-1]['final_answer'])}</div>",
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown("**🤖 Naive RAG (Baseline)**")
            st.caption("No evaluation, no web search, no refinement")
            st.markdown(
                f"<div class='answer-box' style='font-size:0.85rem; min-height:200px; "
                f"border-left-color: #94a3b8;'>"
                f"{_format_answer(history[-1]['naive_rag_answer'])}</div>",
                unsafe_allow_html=True,
            )
    else:
        st.info("Enable 'Compare vs Naive RAG' in the sidebar to see the comparison.")

    st.markdown("---")

    # Query history table
    st.markdown("#### Query History")
    import pandas as pd
    rows = []
    for i, h in enumerate(history, 1):
        rows.append({
            "#": i,
            "Query": h["original_query"][:60] + ("..." if len(h["original_query"]) > 60 else ""),
            "Decision": h["relevance_decision"],
            "Relevance": round(h["overall_relevance"], 2),
            "Grounding": round(h["hallucination_score"], 2),
            "Web Search": "Yes" if h.get("web_search_used") else "No",
            "Latency (s)": round(h["total_latency_ms"] / 1000, 2),
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Download CSV
    st.download_button(
        "📥 Download as CSV",
        df.to_csv(index=False).encode(),
        file_name="crag_eval_results.csv",
        mime="text/csv",
    )


def _render_knowledge_base():
    """Render the knowledge base browser tab."""
    st.markdown("### 📚 Knowledge Base")
    st.caption("Browse the arXiv ML/AI papers used as the CRAG agent's local knowledge base.")

    try:
        from src.data.loader import load_arxiv_papers
        papers = load_arxiv_papers()
    except Exception as e:
        st.error(f"Could not load papers: {e}")
        return

    st.markdown(f"**{len(papers)} papers** loaded.")

    # Search filter
    search = st.text_input("🔍 Filter by keyword", placeholder="e.g., RAG, transformer, hallucination")

    if search:
        filtered = [
            p for p in papers
            if search.lower() in p.get("title", "").lower()
            or search.lower() in p.get("abstract", "").lower()
            or search.lower() in p.get("topic", "").lower()
        ]
    else:
        filtered = papers

    st.caption(f"Showing {len(filtered)} of {len(papers)} papers.")

    # Topic distribution
    import pandas as pd
    topics = {}
    for p in papers:
        t = p.get("topic", "unknown")
        topics[t] = topics.get(t, 0) + 1

    st.markdown("#### Topic Distribution")
    topic_df = pd.DataFrame(
        [{"Topic": t, "Count": c} for t, c in sorted(topics.items(), key=lambda x: -x[1])]
    )
    st.bar_chart(topic_df.set_index("Topic"))

    st.markdown("---")

    # Paper cards
    st.markdown("#### Papers")
    for p in filtered[:30]:
        with st.expander(f"📄 {p.get('title', 'Untitled')}  ·  {p.get('arxiv_id', '')}"):
            st.markdown(f"**Topic:** {p.get('topic', '')}")
            st.markdown(f"**Authors:** {', '.join(p.get('authors', [])[:5])}")
            st.markdown(f"**Published:** {p.get('published', '')}")
            st.markdown(f"**URL:** {p.get('url', '')}")
            st.markdown("**Abstract:**")
            st.write(p.get("abstract", ""))

    if len(filtered) > 30:
        st.info(f"Showing first 30 of {len(filtered)} papers. Use the keyword filter to narrow down.")


def _stat_card(value, label):
    st.markdown(
        f"<div class='stat-card'>"
        f"<div class='stat-value'>{value}</div>"
        f"<div class='stat-label'>{label}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _format_answer(text: str) -> str:
    """Basic markdown-to-HTML-lite for the answer box."""
    if not text:
        return "<i>(empty)</i>"
    # Escape HTML
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    # Restore newlines
    text = text.replace("\n", "<br>")
    return text


if __name__ == "__main__":
    main()

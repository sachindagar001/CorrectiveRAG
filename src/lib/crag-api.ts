/**
 * CRAG API client — calls the FastAPI backend via the Caddy gateway.
 *
 * The gateway uses XTransformPort query param to route to the backend port.
 */

const API_PORT = 8000;

function apiUrl(path: string): string {
  // The Caddy gateway routes requests with XTransformPort=<port> to localhost:<port>
  const sep = path.includes("?") ? "&" : "?";
  return `${path}${sep}XTransformPort=${API_PORT}`;
}

export interface SystemStatus {
  status: string;
  deepseek_key_set: boolean;
  deepseek_model: string;
  deepseek_use_reasoning: boolean;
  embedding_model: string;
  faiss_index_built: boolean;
  retriever_ready: boolean;
  n_papers: number;
  top_k: number;
}

export interface Paper {
  arxiv_id: string;
  title: string;
  abstract: string;
  authors: string[];
  published: string;
  url: string;
  topic: string;
  categories?: string[];
  text?: string;
  source?: string;
}

export interface TopicCount {
  topic: string;
  count: number;
}

export interface ArchitectureNode {
  name: string;
  label: string;
  description: string;
}

export interface ArchitectureInfo {
  mermaid: string;
  nodes: ArchitectureNode[];
}

export interface RetrievedDoc {
  text: string;
  source: string;
  title?: string;
  url?: string;
  score: number;
  relevance_score?: number;
}

export interface Citation {
  source: string;
  title: string;
  url?: string;
  score: number;
  relevance_score?: number;
  snippet: string;
}

export interface CragResult {
  original_query: string;
  rewritten_queries: string[];
  hyde_document: string | null;
  retrieved_docs: RetrievedDoc[];
  retrieval_method: string;
  relevance_scores: number[];
  overall_relevance: number;
  relevance_decision: "relevant" | "irrelevant" | "ambiguous";
  evaluator_reasoning: string;
  refined_knowledge: string;
  web_search_used: boolean;
  web_search_results: RetrievedDoc[];
  web_search_query: string;
  final_answer: string;
  citations: Citation[];
  hallucination_score: number;
  hallucination_reasoning: string;
  is_hallucinated: boolean;
  naive_rag_answer: string;
  node_trace: string[];
  latency_ms: Record<string, number>;
  total_latency_ms: number;
  timestamp: string;
  error: string | null;
}

export interface QueryRequest {
  query: string;
  use_query_rewriting: boolean;
  use_baseline: boolean;
}

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const url = apiUrl(path);
  const res = await fetch(url, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers || {}),
    },
  });
  if (!res.ok) {
    let msg = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      msg = body.detail || msg;
    } catch {
      // ignore
    }
    throw new Error(msg);
  }
  return res.json() as Promise<T>;
}

export const cragApi = {
  health: () => fetchJson<SystemStatus>("/api/health"),
  papers: (q?: string) =>
    fetchJson<{ count: number; papers: Paper[] }>(
      `/api/papers${q ? `?q=${encodeURIComponent(q)}` : ""}`,
    ),
  topics: () => fetchJson<{ topics: TopicCount[]; total_papers: number }>("/api/topics"),
  architecture: () => fetchJson<ArchitectureInfo>("/api/architecture"),
  query: (req: QueryRequest) =>
    fetchJson<CragResult>("/api/query", {
      method: "POST",
      body: JSON.stringify(req),
    }),
};

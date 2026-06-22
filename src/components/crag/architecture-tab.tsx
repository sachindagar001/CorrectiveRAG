"use client";

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Network } from "lucide-react";
import type { ArchitectureInfo } from "@/lib/crag-api";
import { useEffect, useRef } from "react";

interface ArchitectureTabProps {
  archInfo: ArchitectureInfo | null;
}

export function ArchitectureTab({ archInfo }: ArchitectureTabProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  // Render mermaid diagram using the mermaid library (loaded dynamically)
  useEffect(() => {
    if (!archInfo?.mermaid || !containerRef.current) return;

    let cancelled = false;
    const renderMermaid = async () => {
      try {
        // Dynamically import mermaid
        const mermaid = (await import("mermaid")).default;
        mermaid.initialize({
          startOnLoad: false,
          theme: "base",
          themeVariables: {
            background: "#ffffff",
            primaryColor: "#f1f5f9",
            primaryTextColor: "#1e293b",
            primaryBorderColor: "#cbd5e1",
            lineColor: "#64748b",
            secondaryColor: "#f8fafc",
            tertiaryColor: "#ffffff",
            fontFamily: "Inter, sans-serif",
            fontSize: "14px",
          },
          flowchart: {
            curve: "basis",
            padding: 16,
          },
        });

        const id = `mermaid-${Date.now()}`;
        const { svg } = await mermaid.render(id, archInfo.mermaid);
        if (!cancelled && containerRef.current) {
          containerRef.current.innerHTML = svg;
        }
      } catch (e) {
        console.error("Mermaid render failed:", e);
        if (!cancelled && containerRef.current) {
          containerRef.current.innerHTML = `<pre class="text-xs text-slate-600 overflow-auto">${archInfo.mermaid}</pre>`;
        }
      }
    };
    renderMermaid();
    return () => {
      cancelled = true;
    };
  }, [archInfo?.mermaid]);

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <Network className="h-5 w-5 text-violet-600" />
            CRAG Architecture
          </CardTitle>
          <CardDescription>
            Built with LangGraph. Each node is a function that transforms the
            state. The router after <code className="text-xs bg-slate-100 px-1 rounded">evaluate</code>{" "}
            decides whether to use retrieved docs, search the web, or both.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div
            ref={containerRef}
            className="w-full overflow-x-auto rounded-lg border border-slate-200 bg-white p-6 min-h-[400px] flex items-center justify-center"
          >
            {!archInfo && (
              <div className="text-sm text-slate-500">Loading architecture...</div>
            )}
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Node Descriptions</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          {archInfo?.nodes?.map((node) => (
            <div key={node.name}>
              <div className="flex items-baseline gap-2 mb-1">
                <Badge
                  variant="secondary"
                  className="font-mono text-[11px] bg-violet-100 text-violet-800"
                >
                  {node.name}
                </Badge>
                <span className="text-sm font-semibold text-slate-900">
                  {node.label}
                </span>
              </div>
              <p className="text-xs text-slate-600 ml-1">{node.description}</p>
              <Separator className="mt-3" />
            </div>
          ))}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">State Schema</CardTitle>
          <CardDescription>The TypedDict that flows through every node.</CardDescription>
        </CardHeader>
        <CardContent>
          <details>
            <summary className="cursor-pointer text-xs text-slate-600 hover:text-slate-900">
              View full CRAGState definition
            </summary>
            <pre className="mt-3 text-[11px] bg-slate-900 text-slate-100 p-4 rounded-lg overflow-auto font-mono">
{`class CRAGState(TypedDict):
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
    error: Optional[str]`}
            </pre>
          </details>
        </CardContent>
      </Card>
    </div>
  );
}

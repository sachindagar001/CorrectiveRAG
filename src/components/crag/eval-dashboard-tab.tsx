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
import { Button } from "@/components/ui/button";
import { BarChart3, Download } from "lucide-react";
import type { CragResult } from "@/lib/crag-api";

interface EvalDashboardTabProps {
  history: CragResult[];
  lastResult: CragResult | null;
}

export function EvalDashboardTab({ history, lastResult }: EvalDashboardTabProps) {
  if (history.length === 0) {
    return (
      <Card className="min-h-[400px] flex items-center justify-center">
        <CardContent className="text-center py-20">
          <BarChart3 className="h-10 w-10 mx-auto mb-3 text-slate-300" />
          <p className="text-sm text-slate-600">No queries run yet.</p>
          <p className="text-xs text-slate-500 mt-1">
            Run a query in the Chat tab to populate the dashboard.
          </p>
        </CardContent>
      </Card>
    );
  }

  // Aggregate metrics
  const total = history.length;
  const avgLatency =
    history.reduce((s, h) => s + h.total_latency_ms, 0) / total / 1000;
  const avgRelevance = history.reduce((s, h) => s + h.overall_relevance, 0) / total;
  const avgGrounding = history.reduce((s, h) => s + h.hallucination_score, 0) / total;
  const webSearchCount = history.filter((h) => h.web_search_used).length;

  // Per-node latency from last result
  const lastLatencies = lastResult?.latency_ms ?? {};
  const maxLatency = Math.max(...Object.values(lastLatencies), 1);

  const downloadCsv = () => {
    const rows = [
      ["#", "Query", "Decision", "Relevance", "Grounding", "WebSearch", "Latency(s)"],
      ...history.map((h, i) => [
        String(i + 1),
        h.original_query.slice(0, 60),
        h.relevance_decision,
        h.overall_relevance.toFixed(2),
        h.hallucination_score.toFixed(2),
        h.web_search_used ? "Yes" : "No",
        (h.total_latency_ms / 1000).toFixed(2),
      ]),
    ];
    const csv = rows
      .map((r) => r.map((c) => `"${String(c).replace(/"/g, '""')}"`).join(","))
      .join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "crag_eval_results.csv";
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-4">
      {/* Summary */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <BarChart3 className="h-5 w-5 text-violet-600" />
            Session Summary
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
            <MetricCard value={String(total)} label="Total Queries" color="text-violet-600" />
            <MetricCard value={`${avgLatency.toFixed(2)}s`} label="Avg Latency" color="text-violet-600" />
            <MetricCard value={avgRelevance.toFixed(2)} label="Avg Relevance" color="text-amber-600" />
            <MetricCard
              value={avgGrounding.toFixed(2)}
              label="Avg Grounding"
              color={avgGrounding >= 0.7 ? "text-emerald-600" : "text-amber-600"}
            />
            <MetricCard value={String(webSearchCount)} label="Web Searches" color="text-rose-600" />
          </div>
        </CardContent>
      </Card>

      {/* Per-node latency */}
      {Object.keys(lastLatencies).length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Per-Node Latency (Latest Query)</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            {Object.entries(lastLatencies)
              .sort(([, a], [, b]) => b - a)
              .map(([node, ms]) => {
                const pct = (ms / maxLatency) * 100;
                return (
                  <div key={node} className="flex items-center gap-3">
                    <div className="w-32 text-xs font-medium text-slate-700 truncate">
                      {node}
                    </div>
                    <div className="flex-1 h-2 bg-slate-100 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-violet-500 to-violet-400 rounded-full transition-all duration-500"
                        style={{ width: `${pct}%` }}
                      />
                    </div>
                    <div className="w-16 text-right text-xs font-semibold text-slate-900">
                      {ms.toFixed(0)} ms
                    </div>
                  </div>
                );
              })}
          </CardContent>
        </Card>
      )}

      {/* CRAG vs Naive RAG */}
      {lastResult && lastResult.naive_rag_answer && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">CRAG vs Naive RAG</CardTitle>
            <CardDescription>
              Side-by-side comparison of the last query.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-sm font-semibold text-slate-900">
                    CRAG (Self-Correcting)
                  </span>
                </div>
                <p className="text-[11px] text-slate-600 mb-2">
                  Grounding: {lastResult.hallucination_score.toFixed(2)} | Decision:{" "}
                  {lastResult.relevance_decision}
                </p>
                <div className="rounded-lg border-l-4 border-violet-500 bg-gradient-to-br from-slate-50 to-violet-50/30 p-3 text-xs text-slate-800 whitespace-pre-wrap min-h-[180px]">
                  {lastResult.final_answer || "(empty)"}
                </div>
              </div>
              <div>
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-sm font-semibold text-slate-900">
                    Naive RAG (Baseline)
                  </span>
                </div>
                <p className="text-[11px] text-slate-600 mb-2">
                  No evaluation, no web search, no refinement
                </p>
                <div className="rounded-lg border-l-4 border-slate-400 bg-gradient-to-br from-slate-50 to-slate-100/50 p-3 text-xs text-slate-800 whitespace-pre-wrap min-h-[180px]">
                  {lastResult.naive_rag_answer || "(empty)"}
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Query history */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0">
          <CardTitle className="text-base">Query History</CardTitle>
          <Button size="sm" variant="outline" onClick={downloadCsv}>
            <Download className="h-3.5 w-3.5 mr-1" />
            CSV
          </Button>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-slate-200 text-slate-600 text-left">
                  <th className="py-2 pr-3">#</th>
                  <th className="py-2 pr-3">Query</th>
                  <th className="py-2 pr-3">Decision</th>
                  <th className="py-2 pr-3">Relevance</th>
                  <th className="py-2 pr-3">Grounding</th>
                  <th className="py-2 pr-3">Web Search</th>
                  <th className="py-2 pr-3">Latency (s)</th>
                </tr>
              </thead>
              <tbody>
                {history.map((h, i) => (
                  <tr key={i} className="border-b border-slate-100 hover:bg-slate-50">
                    <td className="py-2 pr-3 text-slate-500">{i + 1}</td>
                    <td className="py-2 pr-3 text-slate-900 max-w-xs truncate">
                      {h.original_query}
                    </td>
                    <td className="py-2 pr-3">
                      <Badge
                        className={
                          h.relevance_decision === "relevant"
                            ? "bg-emerald-100 text-emerald-800"
                            : h.relevance_decision === "irrelevant"
                              ? "bg-rose-100 text-rose-800"
                              : "bg-amber-100 text-amber-800"
                        }
                      >
                        {h.relevance_decision}
                      </Badge>
                    </td>
                    <td className="py-2 pr-3 text-slate-700">
                      {h.overall_relevance.toFixed(2)}
                    </td>
                    <td className="py-2 pr-3 text-slate-700">
                      {h.hallucination_score.toFixed(2)}
                    </td>
                    <td className="py-2 pr-3 text-slate-700">
                      {h.web_search_used ? "Yes" : "No"}
                    </td>
                    <td className="py-2 pr-3 text-slate-700">
                      {(h.total_latency_ms / 1000).toFixed(2)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

function MetricCard({
  value,
  label,
  color,
}: {
  value: string;
  label: string;
  color: string;
}) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-3 text-center">
      <div className={`text-xl font-bold ${color}`}>{value}</div>
      <div className="text-[10px] text-slate-500 uppercase tracking-wide mt-1">
        {label}
      </div>
    </div>
  );
}

"use client";

import { useState } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Sparkles,
  Send,
  Loader2,
  Globe,
  ShieldCheck,
  ShieldAlert,
  ShieldX,
  Clock,
  Target,
  CheckCircle2,
  FileText,
} from "lucide-react";
import type { CragResult } from "@/lib/crag-api";

interface ChatTabProps {
  onRunQuery: (q: string) => void;
  running: boolean;
  lastResult: CragResult | null;
  messages: { role: "user" | "assistant"; content: string }[];
}

const SAMPLE_QUESTIONS = [
  "What is retrieval-augmented generation?",
  "How does chain-of-thought prompting work?",
  "What is HyDE and when is it useful?",
  "What are the latest techniques to detect hallucinations in LLMs?",
  "What is the stock price of Apple today?", // forces web search
];

export function ChatTab({ onRunQuery, running, lastResult, messages }: ChatTabProps) {
  const [query, setQuery] = useState("");

  const handleRun = () => {
    if (!query.trim() || running) return;
    onRunQuery(query);
  };

  const handleSample = (q: string) => {
    setQuery(q);
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* LEFT: Input + conversation */}
      <div className="space-y-4">
        <Card>
          <CardHeader>
            <CardTitle className="text-lg flex items-center gap-2">
              <Sparkles className="h-5 w-5 text-violet-600" />
              Ask a Question
            </CardTitle>
            <CardDescription>
              Try questions about RAG, LLMs, transformers, hallucination, or agent workflows.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <details className="text-sm">
              <summary className="cursor-pointer text-slate-600 hover:text-slate-900">
                Try a sample question
              </summary>
              <div className="mt-3 space-y-2">
                {SAMPLE_QUESTIONS.map((q) => (
                  <button
                    key={q}
                    onClick={() => handleSample(q)}
                    className="block w-full text-left rounded-md border border-slate-200 bg-white px-3 py-2 text-xs text-slate-700 hover:border-violet-400 hover:bg-violet-50 transition"
                  >
                    {q}
                  </button>
                ))}
              </div>
            </details>

            <Textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="e.g., What is CRAG and how does it improve on naive RAG?"
              className="min-h-[100px] resize-none"
              onKeyDown={(e) => {
                if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
                  handleRun();
                }
              }}
            />
            <Button
              onClick={handleRun}
              disabled={!query.trim() || running}
              className="w-full"
            >
              {running ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Running CRAG pipeline...
                </>
              ) : (
                <>
                  <Send className="h-4 w-4 mr-2" />
                  Run CRAG
                </>
              )}
            </Button>
            <p className="text-[10px] text-slate-500 text-center">
              Tip: ⌘/Ctrl + Enter to run
            </p>
          </CardContent>
        </Card>

        {/* Conversation */}
        {messages.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Conversation</CardTitle>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-[400px] pr-4">
                <div className="space-y-4">
                  {messages.map((msg, i) => (
                    <div
                      key={i}
                      className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
                    >
                      <div
                        className={`max-w-[85%] rounded-2xl px-4 py-2.5 text-sm ${
                          msg.role === "user"
                            ? "bg-violet-600 text-white"
                            : "bg-slate-100 text-slate-900"
                        }`}
                      >
                        <div className="whitespace-pre-wrap break-words">{msg.content}</div>
                      </div>
                    </div>
                  ))}
                  {running && (
                    <div className="flex justify-start">
                      <div className="bg-slate-100 rounded-2xl px-4 py-2.5 text-sm text-slate-500 flex items-center gap-2">
                        <Loader2 className="h-3 w-3 animate-spin" />
                        Thinking...
                      </div>
                    </div>
                  )}
                </div>
              </ScrollArea>
            </CardContent>
          </Card>
        )}
      </div>

      {/* RIGHT: Latest result */}
      <div>
        <ResultPanel result={lastResult} running={running} />
      </div>
    </div>
  );
}

function ResultPanel({
  result,
  running,
}: {
  result: CragResult | null;
  running: boolean;
}) {
  if (!result && !running) {
    return (
      <Card className="h-full min-h-[500px] flex items-center justify-center">
        <CardContent className="text-center py-20">
          <div className="mx-auto mb-4 h-12 w-12 rounded-full bg-slate-100 flex items-center justify-center">
            <Sparkles className="h-6 w-6 text-slate-400" />
          </div>
          <p className="text-sm text-slate-600">
            Run a query to see the CRAG pipeline in action.
          </p>
        </CardContent>
      </Card>
    );
  }

  if (running && !result) {
    return (
      <Card className="h-full min-h-[500px] flex items-center justify-center">
        <CardContent className="text-center py-20">
          <Loader2 className="h-8 w-8 animate-spin mx-auto mb-4 text-violet-600" />
          <p className="text-sm text-slate-600">Running CRAG pipeline...</p>
          <p className="text-xs text-slate-500 mt-1">
            This may take 10-30 seconds (5+ LLM calls).
          </p>
        </CardContent>
      </Card>
    );
  }

  if (!result) return null;

  const hScore = result.hallucination_score;
  const hBadge =
    hScore >= 0.7
      ? { variant: "default" as const, icon: ShieldCheck, text: "Well-Grounded", className: "bg-emerald-100 text-emerald-800 border-emerald-200" }
      : hScore >= 0.4
        ? { variant: "secondary" as const, icon: ShieldAlert, text: "Partially Grounded", className: "bg-amber-100 text-amber-800 border-amber-200" }
        : { variant: "destructive" as const, icon: ShieldX, text: "Likely Hallucinated", className: "bg-rose-100 text-rose-800 border-rose-200" };
  const HIcon = hBadge.icon;

  return (
    <Card className="h-full">
      <CardHeader>
        <CardTitle className="text-lg flex items-center gap-2">
          Latest Result
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Stat cards */}
        <div className="grid grid-cols-4 gap-2">
          <StatCard
            icon={Clock}
            value={`${(result.total_latency_ms / 1000).toFixed(2)}s`}
            label="Latency"
            color="text-violet-600"
          />
          <StatCard
            icon={Target}
            value={result.overall_relevance.toFixed(2)}
            label="Relevance"
            color="text-amber-600"
          />
          <StatCard
            icon={ShieldCheck}
            value={result.hallucination_score.toFixed(2)}
            label="Grounding"
            color={
              hScore >= 0.7
                ? "text-emerald-600"
                : hScore >= 0.4
                  ? "text-amber-600"
                  : "text-rose-600"
            }
          />
          <StatCard
            icon={FileText}
            value={String(result.citations?.length ?? 0)}
            label="Citations"
            color="text-slate-700"
          />
        </div>

        {/* Decision pills */}
        <div className="flex flex-wrap gap-2">
          <DecisionPill decision={result.relevance_decision} />
          {result.web_search_used && (
            <Badge className="bg-amber-100 text-amber-800 border-amber-200">
              <Globe className="h-3 w-3 mr-1" />
              Web Search Triggered (DuckDuckGo)
            </Badge>
          )}
          <Badge className={hBadge.className}>
            <HIcon className="h-3 w-3 mr-1" />
            {hBadge.text}
          </Badge>
        </div>

        <Separator />

        {/* Answer */}
        <div>
          <h4 className="text-sm font-semibold mb-2 text-slate-900">Answer</h4>
          <div className="rounded-lg border-l-4 border-violet-500 bg-gradient-to-br from-slate-50 to-violet-50/30 p-4 text-sm text-slate-800 leading-relaxed whitespace-pre-wrap">
            {result.final_answer || "(empty)"}
          </div>
        </div>

        {/* Evaluator reasoning */}
        {result.evaluator_reasoning && (
          <div>
            <h5 className="text-xs font-semibold mb-1 text-slate-700 flex items-center gap-1">
              <Target className="h-3 w-3" /> Evaluator Reasoning
            </h5>
            <p className="text-xs text-slate-600 italic">{result.evaluator_reasoning}</p>
          </div>
        )}

        {/* Hallucination reasoning */}
        {result.hallucination_reasoning && (
          <div>
            <h5 className="text-xs font-semibold mb-1 text-slate-700 flex items-center gap-1">
              <ShieldCheck className="h-3 w-3" /> Hallucination Check
            </h5>
            <p className="text-xs text-slate-600 italic">{result.hallucination_reasoning}</p>
          </div>
        )}

        {/* Execution trace */}
        <div>
          <h5 className="text-xs font-semibold mb-2 text-slate-700">
            Execution Trace
          </h5>
          <div className="flex flex-wrap gap-1">
            {result.node_trace?.map((node, i) => (
              <span key={i} className="inline-flex items-center text-xs">
                <Badge
                  variant="secondary"
                  className="bg-violet-100 text-violet-800 border-violet-200"
                >
                  {node}
                </Badge>
                {i < result.node_trace.length - 1 && (
                  <span className="mx-1 text-slate-400">→</span>
                )}
              </span>
            ))}
          </div>
        </div>

        {/* Rewritten queries (if any) */}
        {result.rewritten_queries && result.rewritten_queries.length > 0 && (
          <div>
            <h5 className="text-xs font-semibold mb-2 text-slate-700">
              Query Rewriting (Multi-Query)
            </h5>
            <div className="space-y-1">
              {result.rewritten_queries.map((q, i) => (
                <div key={i} className="text-xs text-slate-600 bg-slate-50 rounded px-2 py-1">
                  {i + 1}. {q}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* HyDE document (if any) */}
        {result.hyde_document && (
          <div>
            <h5 className="text-xs font-semibold mb-2 text-slate-700">
              HyDE Document (hypothetical answer for retrieval)
            </h5>
            <div className="text-xs text-slate-600 bg-slate-50 rounded p-2 italic max-h-32 overflow-y-auto">
              {result.hyde_document}
            </div>
          </div>
        )}

        {/* Citations */}
        {result.citations && result.citations.length > 0 && (
          <div>
            <h5 className="text-xs font-semibold mb-2 text-slate-700">
              Citations ({result.citations.length})
            </h5>
            <div className="space-y-1.5 max-h-48 overflow-y-auto">
              {result.citations.map((c, i) => (
                <div
                  key={i}
                  className="rounded-md border border-slate-200 bg-white px-3 py-2 text-xs"
                >
                  <div className="flex items-start justify-between gap-2">
                    <div className="font-medium text-slate-900 truncate">
                      [{c.source}] {c.title}
                    </div>
                    <div className="flex gap-1 shrink-0">
                      <Badge variant="outline" className="text-[10px]">
                        sim: {c.score.toFixed(2)}
                      </Badge>
                      {c.relevance_score !== undefined && (
                        <Badge variant="outline" className="text-[10px]">
                          rel: {c.relevance_score.toFixed(2)}
                        </Badge>
                      )}
                    </div>
                  </div>
                  <p className="text-slate-600 mt-1 line-clamp-2">{c.snippet}</p>
                  {c.url && (
                    <a
                      href={c.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-violet-600 hover:underline text-[10px] mt-1 inline-block"
                    >
                      {c.url}
                    </a>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function StatCard({
  icon: Icon,
  value,
  label,
  color,
}: {
  icon: React.ElementType;
  value: string;
  label: string;
  color: string;
}) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-2.5 text-center">
      <Icon className={`h-3.5 w-3.5 mx-auto mb-1 ${color}`} />
      <div className={`text-base font-bold ${color}`}>{value}</div>
      <div className="text-[10px] text-slate-500 uppercase tracking-wide">{label}</div>
    </div>
  );
}

function DecisionPill({ decision }: { decision: string }) {
  const styles =
    decision === "relevant"
      ? "bg-emerald-100 text-emerald-800 border-emerald-200"
      : decision === "irrelevant"
        ? "bg-rose-100 text-rose-800 border-rose-200"
        : "bg-amber-100 text-amber-800 border-amber-200";
  return (
    <Badge className={styles}>
      <CheckCircle2 className="h-3 w-3 mr-1" />
      Retrieval: {decision.toUpperCase()}
    </Badge>
  );
}

"use client";

import { useEffect, useState, useCallback } from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Toaster } from "@/components/ui/toaster";
import { toast } from "sonner";
import {
  cragApi,
  type SystemStatus,
  type CragResult,
  type ArchitectureInfo,
  type Paper,
  type TopicCount,
} from "@/lib/crag-api";
import { ChatTab } from "@/components/crag/chat-tab";
import { ArchitectureTab } from "@/components/crag/architecture-tab";
import { EvalDashboardTab } from "@/components/crag/eval-dashboard-tab";
import { KnowledgeBaseTab } from "@/components/crag/knowledge-base-tab";
import { Sidebar } from "@/components/crag/sidebar";

export default function Home() {
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [lastResult, setLastResult] = useState<CragResult | null>(null);
  const [history, setHistory] = useState<CragResult[]>([]);
  const [messages, setMessages] = useState<{ role: "user" | "assistant"; content: string }[]>([]);
  const [running, setRunning] = useState(false);
  const [useQueryRewriting, setUseQueryRewriting] = useState(true);
  const [useBaseline, setUseBaseline] = useState(true);

  // Architecture tab data
  const [archInfo, setArchInfo] = useState<ArchitectureInfo | null>(null);

  // Knowledge base tab data
  const [papers, setPapers] = useState<Paper[]>([]);
  const [topics, setTopics] = useState<TopicCount[]>([]);

  // ---------- load system status on mount ----------
  const refreshStatus = useCallback(async () => {
    try {
      const s = await cragApi.health();
      setStatus(s);
    } catch (e) {
      console.error("Health check failed:", e);
    }
  }, []);

  useEffect(() => {
    refreshStatus();
    // Also load architecture (cached) and topics
    cragApi.architecture().then(setArchInfo).catch(console.error);
    cragApi.topics().then((t) => setTopics(t.topics)).catch(console.error);
  }, [refreshStatus]);

  // ---------- run a CRAG query ----------
  const runQuery = useCallback(
    async (query: string) => {
      if (!query.trim()) return;
      if (!status?.deepseek_key_set) {
        toast.error("DeepSeek API key not set", {
          description: "Add DEEPSEEK_API_KEY to .env and restart the backend.",
        });
        return;
      }
      setRunning(true);
      setMessages((prev) => [
        ...prev,
        { role: "user", content: query },
      ]);
      try {
        const result = await cragApi.query({
          query,
          use_query_rewriting: useQueryRewriting,
          use_baseline: useBaseline,
        });
        if (result.error) {
          toast.error("CRAG pipeline error", { description: result.error });
          setMessages((prev) => [
            ...prev,
            { role: "assistant", content: `Error: ${result.error}` },
          ]);
        } else {
          setLastResult(result);
          setHistory((prev) => [...prev, result]);
          setMessages((prev) => [
            ...prev,
            { role: "assistant", content: result.final_answer },
          ]);
          toast.success("CRAG pipeline complete", {
            description: `Decision: ${result.relevance_decision} | Grounding: ${result.hallucination_score.toFixed(2)}`,
          });
        }
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        toast.error("Query failed", { description: msg });
        setMessages((prev) => [
          ...prev,
          { role: "assistant", content: `Error: ${msg}` },
        ]);
      } finally {
        setRunning(false);
      }
    },
    [status, useQueryRewriting, useBaseline],
  );

  const clearAll = useCallback(() => {
    setMessages([]);
    setHistory([]);
    setLastResult(null);
  }, []);

  const loadPapers = useCallback(async (q?: string) => {
    try {
      const res = await cragApi.papers(q);
      setPapers(res.papers);
    } catch (e) {
      console.error("Failed to load papers:", e);
    }
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-slate-100">
      <div className="flex">
        <Sidebar
          status={status}
          useQueryRewriting={useQueryRewriting}
          setUseQueryRewriting={setUseQueryRewriting}
          useBaseline={useBaseline}
          setUseBaseline={setUseBaseline}
          onClear={clearAll}
        />
        <main className="flex-1 p-6 lg:p-8 max-w-6xl">
          <header className="mb-6">
            <h1 className="text-3xl font-bold tracking-tight text-slate-900">
              CRAG Agent
            </h1>
            <p className="mt-1 text-sm text-slate-600">
              Self-Correcting Retrieval-Augmented Generation — evaluates retrieval
              quality, falls back to web search, and checks its own answers for
              hallucinations.
            </p>
          </header>

          <Tabs defaultValue="chat" className="w-full">
            <TabsList className="grid w-full grid-cols-4 mb-6">
              <TabsTrigger value="chat">Chat</TabsTrigger>
              <TabsTrigger value="architecture">Architecture</TabsTrigger>
              <TabsTrigger value="eval">Eval Dashboard</TabsTrigger>
              <TabsTrigger value="knowledge">Knowledge Base</TabsTrigger>
            </TabsList>

            <TabsContent value="chat" className="mt-0">
              <ChatTab
                onRunQuery={runQuery}
                running={running}
                lastResult={lastResult}
                messages={messages}
              />
            </TabsContent>

            <TabsContent value="architecture" className="mt-0">
              <ArchitectureTab archInfo={archInfo} />
            </TabsContent>

            <TabsContent value="eval" className="mt-0">
              <EvalDashboardTab
                history={history}
                lastResult={lastResult}
              />
            </TabsContent>

            <TabsContent value="knowledge" className="mt-0">
              <KnowledgeBaseTab
                papers={papers}
                topics={topics}
                onLoadPapers={loadPapers}
              />
            </TabsContent>
          </Tabs>
        </main>
      </div>
      <Toaster />
    </div>
  );
}

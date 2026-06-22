"use client";

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import type { SystemStatus } from "@/lib/crag-api";

interface SidebarProps {
  status: SystemStatus | null;
  useQueryRewriting: boolean;
  setUseQueryRewriting: (v: boolean) => void;
  useBaseline: boolean;
  setUseBaseline: (v: boolean) => void;
  onClear: () => void;
}

export function Sidebar({
  status,
  useQueryRewriting,
  setUseQueryRewriting,
  useBaseline,
  setUseBaseline,
  onClear,
}: SidebarProps) {
  return (
    <aside className="w-72 shrink-0 border-r border-slate-200 bg-slate-50/50 p-4 min-h-screen">
      <div className="mb-6">
        <h2 className="text-lg font-bold text-slate-900">CRAG Agent</h2>
        <p className="text-xs text-slate-600 mt-1">
          Self-Correcting Retrieval-Augmented Generation
        </p>
      </div>

      <Separator className="mb-4" />

      {/* System Status */}
      <div className="mb-4">
        <h3 className="text-sm font-semibold text-slate-900 mb-3">System Status</h3>
        <div className="space-y-2 text-sm">
          <StatusRow
            ok={status?.deepseek_key_set ?? false}
            label="DeepSeek API Key"
            okText="Configured"
            failText="Missing — add to .env"
            hint="platform.deepseek.com"
          />
          <StatusRow
            ok={status?.faiss_index_built ?? false}
            label="FAISS Index"
            okText="Built"
            failText="Not built"
          />
          <StatusRow
            ok={status?.retriever_ready ?? false}
            label="Retriever"
            okText="Ready"
            failText="Not loaded"
          />
          <div className="text-xs text-slate-600 pl-1">
            Knowledge Base:{" "}
            <span className="font-medium text-slate-900">
              {status?.n_papers ?? 0} arXiv papers
            </span>
          </div>
        </div>
      </div>

      <Separator className="mb-4" />

      {/* Pipeline Settings */}
      <div className="mb-4">
        <h3 className="text-sm font-semibold text-slate-900 mb-3">Pipeline Settings</h3>
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div className="space-y-0.5 pr-2">
              <Label htmlFor="qr-switch" className="text-xs font-medium cursor-pointer">
                Query Rewriting
              </Label>
              <p className="text-[10px] text-slate-500">
                Multi-Query + HyDE
              </p>
            </div>
            <Switch
              id="qr-switch"
              checked={useQueryRewriting}
              onCheckedChange={setUseQueryRewriting}
            />
          </div>
          <div className="flex items-center justify-between">
            <div className="space-y-0.5 pr-2">
              <Label htmlFor="bl-switch" className="text-xs font-medium cursor-pointer">
                Compare vs Naive RAG
              </Label>
              <p className="text-[10px] text-slate-500">
                Baseline for eval dashboard
              </p>
            </div>
            <Switch
              id="bl-switch"
              checked={useBaseline}
              onCheckedChange={setUseBaseline}
            />
          </div>
        </div>
      </div>

      <Separator className="mb-4" />

      <Button
        variant="outline"
        size="sm"
        className="w-full"
        onClick={onClear}
      >
        Clear conversation
      </Button>

      <Separator className="my-4" />

      {/* Configuration */}
      <div className="mb-4">
        <h3 className="text-sm font-semibold text-slate-900 mb-3">Configuration</h3>
        <div className="space-y-1.5 text-[11px] text-slate-600">
          <ConfigRow label="LLM Provider" value="DeepSeek" />
          <ConfigRow
            label="LLM Model"
            value={status?.deepseek_model ?? "—"}
            mono
          />
          <ConfigRow
            label="Reasoning"
            value={status?.deepseek_use_reasoning ? "ON" : "OFF"}
          />
          <ConfigRow
            label="Embeddings"
            value={status?.embedding_model ?? "—"}
            mono
          />
          <ConfigRow label="Vector DB" value="FAISS" />
          <ConfigRow label="Top-K" value={String(status?.top_k ?? 5)} />
        </div>
      </div>

      <Separator className="mb-4" />

      <Card className="bg-white">
        <CardContent className="p-3 text-[11px] text-slate-600 leading-relaxed">
          Built with LangGraph, FAISS, BGE embeddings, and DeepSeek-hosted LLMs.
          The agent retrieves docs, evaluates them with an LLM-as-judge, and falls
          back to DuckDuckGo web search when retrieval is irrelevant.
        </CardContent>
      </Card>
    </aside>
  );
}

function StatusRow({
  ok,
  label,
  okText,
  failText,
  hint,
}: {
  ok: boolean;
  label: string;
  okText: string;
  failText: string;
  hint?: string;
}) {
  return (
    <div className="flex items-start gap-2">
      <span
        className={`mt-1 inline-block h-2 w-2 shrink-0 rounded-full ${
          ok ? "bg-emerald-500" : "bg-rose-500"
        }`}
      />
      <div className="min-w-0 flex-1">
        <div className="text-xs font-medium text-slate-900">{label}</div>
        <div className={`text-[11px] ${ok ? "text-emerald-700" : "text-rose-700"}`}>
          {ok ? okText : failText}
        </div>
        {hint && !ok && (
          <div className="text-[10px] text-slate-500">Get key: {hint}</div>
        )}
      </div>
    </div>
  );
}

function ConfigRow({
  label,
  value,
  mono,
}: {
  label: string;
  value: string;
  mono?: boolean;
}) {
  return (
    <div className="flex items-start justify-between gap-2">
      <span className="text-slate-500 shrink-0">{label}:</span>
      <span
        className={`text-slate-900 text-right break-all ${
          mono ? "font-mono text-[10px]" : ""
        }`}
      >
        {value}
      </span>
    </div>
  );
}

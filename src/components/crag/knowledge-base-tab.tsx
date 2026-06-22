"use client";

import { useEffect, useState } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { BookOpen, Search } from "lucide-react";
import type { Paper, TopicCount } from "@/lib/crag-api";

interface KnowledgeBaseTabProps {
  papers: Paper[];
  topics: TopicCount[];
  onLoadPapers: (q?: string) => void;
}

export function KnowledgeBaseTab({
  papers,
  topics,
  onLoadPapers,
}: KnowledgeBaseTabProps) {
  const [search, setSearch] = useState("");

  // Initial load
  useEffect(() => {
    if (papers.length === 0) {
      onLoadPapers();
    }
  }, [papers.length, onLoadPapers]);

  // Debounced search
  useEffect(() => {
    const t = setTimeout(() => {
      onLoadPapers(search);
    }, 300);
    return () => clearTimeout(t);
  }, [search, onLoadPapers]);

  const maxCount = Math.max(...topics.map((t) => t.count), 1);

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <BookOpen className="h-5 w-5 text-violet-600" />
            Knowledge Base
          </CardTitle>
          <CardDescription>
            Browse the arXiv ML/AI papers used as the CRAG agent&apos;s local knowledge base.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-slate-700">
            <span className="font-semibold">{papers.length}</span> papers loaded.
          </p>
        </CardContent>
      </Card>

      {/* Topic distribution */}
      {topics.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Topic Distribution</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            {topics.map((t) => (
              <div key={t.topic} className="flex items-center gap-3">
                <div className="w-48 text-xs text-slate-700 truncate">{t.topic}</div>
                <div className="flex-1 h-6 bg-slate-100 rounded overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-violet-500 to-violet-400 rounded transition-all duration-500 flex items-center justify-end pr-2"
                    style={{ width: `${(t.count / maxCount) * 100}%` }}
                  >
                    <span className="text-[10px] font-semibold text-white">{t.count}</span>
                  </div>
                </div>
              </div>
            ))}
          </CardContent>
        </Card>
      )}

      <Separator />

      {/* Search + paper list */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Papers</CardTitle>
          <div className="relative mt-2">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400" />
            <Input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Filter by keyword (e.g., RAG, transformer, hallucination)"
              className="pl-9"
            />
          </div>
        </CardHeader>
        <CardContent className="space-y-2 max-h-[600px] overflow-y-auto">
          {papers.length === 0 && (
            <p className="text-sm text-slate-500 text-center py-8">
              No papers match. Try a different keyword.
            </p>
          )}
          {papers.slice(0, 30).map((p) => (
            <details
              key={p.arxiv_id}
              className="rounded-md border border-slate-200 bg-white px-3 py-2 text-sm"
            >
              <summary className="cursor-pointer text-slate-900 font-medium hover:text-violet-700">
                {p.title}{" "}
                <span className="text-xs text-slate-500">· {p.arxiv_id}</span>
              </summary>
              <div className="mt-3 space-y-2 text-xs text-slate-700">
                <div className="flex flex-wrap gap-1.5">
                  <Badge variant="secondary" className="text-[10px]">
                    {p.topic}
                  </Badge>
                  {p.published && (
                    <Badge variant="outline" className="text-[10px]">
                      {p.published}
                    </Badge>
                  )}
                  {p.categories?.map((c) => (
                    <Badge key={c} variant="outline" className="text-[10px]">
                      {c}
                    </Badge>
                  ))}
                </div>
                <div>
                  <span className="text-slate-500">Authors:</span>{" "}
                  {p.authors?.slice(0, 5).join(", ")}
                  {p.authors && p.authors.length > 5 && ", et al."}
                </div>
                <p className="text-slate-700 leading-relaxed">{p.abstract}</p>
                {p.url && (
                  <a
                    href={p.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-violet-600 hover:underline text-[11px] inline-block"
                  >
                    {p.url}
                  </a>
                )}
              </div>
            </details>
          ))}
          {papers.length > 30 && (
            <p className="text-xs text-slate-500 text-center py-3">
              Showing first 30 of {papers.length} papers. Use the keyword filter to narrow down.
            </p>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

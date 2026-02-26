"use client";

import { useState } from "react";
import { AlertCircle, ChevronDown, ChevronRight, Copy, Check } from "lucide-react";
import { Button } from "@/components/ui/button";
import { DevError } from "@/lib/services";

interface DevErrorPanelProps {
  error: Error | null;
}

export function DevErrorPanel({ error }: DevErrorPanelProps) {
  const [expanded, setExpanded] = useState(false);
  const [copied, setCopied] = useState(false);

  if (!error) return null;

  const isDevError = error instanceof DevError;
  const hint = isDevError ? error.hint : undefined;
  const traceback = isDevError ? error.traceback : undefined;
  const detailText = traceback || error.message;

  async function copyToClipboard() {
    await navigator.clipboard.writeText(detailText);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }

  return (
    <div className="rounded-lg border border-destructive/40 bg-destructive/5 p-4 text-sm">
      {/* Header */}
      <div className="flex items-start gap-2">
        <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-destructive" />
        <div className="flex-1 min-w-0">
          <p className="font-medium text-destructive leading-snug">{error.message}</p>
          {hint && (
            <p className="mt-1 text-muted-foreground">{hint}</p>
          )}
        </div>
      </div>

      {/* Toggle */}
      <button
        onClick={() => setExpanded((v) => !v)}
        className="mt-3 flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
      >
        {expanded ? (
          <ChevronDown className="h-3 w-3" />
        ) : (
          <ChevronRight className="h-3 w-3" />
        )}
        {expanded ? "Hide" : "Show"} technical details
      </button>

      {/* Traceback panel */}
      {expanded && (
        <div className="mt-2 relative">
          <pre className="overflow-x-auto rounded-md bg-black/80 p-3 text-xs text-green-400 font-mono leading-relaxed whitespace-pre-wrap break-words max-h-64 overflow-y-auto">
            {detailText}
          </pre>
          <Button
            variant="ghost"
            size="icon"
            className="absolute top-2 right-2 h-6 w-6 text-muted-foreground hover:text-foreground"
            onClick={copyToClipboard}
            title="Copy to clipboard"
          >
            {copied ? (
              <Check className="h-3 w-3 text-green-400" />
            ) : (
              <Copy className="h-3 w-3" />
            )}
          </Button>
        </div>
      )}
    </div>
  );
}
